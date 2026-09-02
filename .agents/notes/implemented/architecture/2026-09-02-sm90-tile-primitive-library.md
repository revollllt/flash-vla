# Agent Note: shared SM90 tile primitive library

Status: implemented

## Problem

Each CUDA task loop carried its own copies of the same device mechanics:
raw `cp.async.bulk.tensor` strings, a local `smem_u32`, bulk-store commit
and wait pairs, a hand-picked CuTe wgmma atom spelled by instruction name,
and FlashMLA's `sm90::gemm` pulled in through a `-I` on the vendored tree.
That made every new kernel re-derive the choreography, left the vendored
references on the build path against the stated intent, and gave the
instruction choice no place to be reasoned about once (the FFN header even
named an atom the kernel had stopped using).

## Decision

- One header-only library at `src/flash_vla/hardware/nvidia/cuda/tile/sm90/`
  (namespace `flash_vla::sm90`) owns the tile-level primitives: g2s (TMA
  2-D/3-D with L2 hints and multicast, bulk 1-D, cp.async tiled copy), s2r
  (ldmatrix derived from the MMA), r2s (fragment conversion plus stmatrix or
  scalar), s2g (TMA store, bulk store, group completion), the mbarrier
  vocabulary, the swizzled smem tile layout, and host tensor-map encoding.
- The tensor-core tables are complete and explicit: every wgmma 64xNx16
  bf16/f16 atom and every 64xNx32 fp8 atom for N = 8..256 step 8 in SS and
  RS form, and the mma.sync m16n8k16 (bf16/f16) and m16n8k32 (all
  e4m3/e5m2 pairs) atoms, all with f32 accumulation. Entries name CuTe atoms
  so traits, partitioning and descriptors stay CuTe's.
- `MmaSelector` picks the instruction at compile time from (tile M, N, K,
  element types, operand locations, majors, thread count); `gemm()` issues
  with the choreography that instruction needs. The wgmma path carries the
  FlashAttention-3 / FlashMLA fence-arrive-commit-wait contract with
  attribution; mma.sync is a plain unrolled K loop. The selection rule and
  caller invariants are the normative content of `tile/README.md`.
- CUTLASS `v4.7.1` is pinned as `third_party/cutlass` and is the only build
  dependency; FlashMLA and DeepGEMM are reference-only and off the include
  path. Build wrappers default `CUTLASS_DIR` to the submodule and hash the
  tile headers into the build key.
- Both production task loops (FFN, attention) are expressed through the
  library. This is a structural change with a codegen-identity requirement,
  not an optimization.

## Alternatives considered

- Keep including FlashMLA's helpers and DeepGEMM's selector directly:
  rejected because the vendored trees are references, their APIs move with
  upstream, and neither covers copies, mma.sync or the fp8 tables together.
- Generate the atom table by arithmetic over N instead of an explicit list:
  rejected; an explicit table gives a compile error naming the missing entry
  and is what a reviewer can diff against the PTX ISA.
- A runtime instruction dispatch: rejected, the workload is fixed-shape and
  every kernel here compiles its geometry in.

## Consequences

- New kernels state geometry (`MmaSelector<...>`) and call `tile::gemm`,
  `tile::tma_load_*`, `tile::copy_s2r` and friends; instruction names and
  PTX strings no longer appear in kernel bodies.
- The C7518 fragment-alternation rule and the proxy-fence rules are stated
  once in `tile/README.md`; the library does not hide them behind helpers.
- The ldmatrix path pairs mma.sync atoms so each warp owns 16 x 16 per
  repetition; a warp-level tile narrower than 16 in N is rejected at compile
  time.
- A stale `.so` cannot survive a tile-header edit (hashed into the key).

## Verification

Login node, CUDA 13.1 module, GCC 13.3, `-arch=sm_90a`, SASS compared with
`cuobjdump -sass` after stripping address columns:

- CUTLASS bump alone (unchanged kernels, 4.5.1 -> 4.7.1): both task loops
  byte-identical to the pre-bump build.
- Migrated FFN task loop: byte-identical to the pre-migration build.
- Migrated attention task loop: same instruction count (10136 lines) and
  identical opcode histogram; 69 lines differ after normalising registers
  and addresses, all reordering of independent integer address arithmetic.
  No ptxas C7518/C7515 in either build.
- `eval/correctness/tile_sm90/primitives.cu` instantiates every table entry
  (static traits check over all N and each dtype / location / major family)
  and asserts the selector reproduces the production atoms; its SASS carries
  HGMMA (N = 32, 64, 128), QGMMA e4m3/e4m3 and e4m3/e5m2, HMMA bf16, LDSM,
  STSM x2 and x4, UTMALDG, UTMASTG, UBLKCP and LDGSTS.  fp8 mma.sync is
  lowered by ptxas to F2FP unpack plus f16 HMMA on sm_90a.

GPU node (SLURM job 585783, H100, `sbatch/pi05_cuda.sh` environment):

- `eval.correctness.tile_sm90.primitives_parity`: eight cases; bf16 wgmma
  and mma.sync at cosine 0.9999986 or better against float32 torch, fp8
  mma.sync exact, fp8 wgmma at 2.5e-4 and 1.4e-4 relative with cosine
  1.0000000 (Hopper's reduced-precision fp8 accumulator; the gate for those
  two cases is 1e-3 and the rerun is recorded in the PR).
- `eval.correctness.pi05.ffn_taskloop_parity --modes gu,dr,full
  --replay-check 2 --seed 7`: worst cosine 0.9999999, replay 1.0, PASS.
- `eval.correctness.pi05.attention_block_parity --impl fused
  --replay-check 2`: passed, worst cosine 0.99997 (its own gate).
