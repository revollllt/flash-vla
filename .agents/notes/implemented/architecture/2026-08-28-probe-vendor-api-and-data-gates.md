# Agent Note: Hardware probes reach instructions through the vendor selector and gate rates on delivered data

Status: implemented

## Problem

The `hardware-unit-test` probes named their instructions themselves. The tensor
core probe carried a hand-written table mapping each tile N to one
`MMA_64xNx16_F32BF16BF16_SS` atom, built its own GMMA descriptors, issued the
warpgroup fence/commit/wait as raw inline PTX, and decoded the accumulator to
`(row, col)` by hand. The TMA probe issued its own `cp.async.bulk.tensor` PTX
and had no correctness check at all.

Two consequences, both of which make a constant unfalsifiable rather than wrong:

- A probe that names its own instruction can measure an instruction the library
  would never have selected, so the constant describes the probe rather than the
  machine. Nothing in the build or the output distinguishes the two cases.
- A wrong descriptor, box geometry or coordinate walk delivers the wrong bytes at
  the right speed. Every row of every TMA sweep still looks reasonable, and the
  derived constants cannot be contradicted by anything the probe prints. The
  `mma` unit had already gated its rates on a numeric check; the copy engine,
  which is the larger half of `sm90/constants.yaml`, had no equivalent.

## Decision

Probes reach a hardware instruction through the vendor's own dispatch, and
assert the identity of what was selected. Probes that move data gate every rate
on a comparison against a host-computed reference.

For the tensor core, `GMMA::ss_op_selector` chooses the atom, `make_tiled_mma`
and `partition_A/B/C` build the operands and the accumulator mapping, and
`cute::gemm` plus the `warpgroup_*` helpers issue and retire it. A
`static_assert` per swept N pins the selector's choice to the atom the recorded
constants were measured on, so a CUTLASS dispatch change fails the build instead
of silently re-measuring a different instruction under the existing tags. The
`NGROUP` and `WAIT` axes keep their meaning: `cute::gemm` does not emit the
warpgroup fence, commit or wait, so pipeline depth remains the probe's to
control.

For the copy engine, a check kernel replays the rate kernel's own coordinate
arithmetic at ring depth 1 and copies each delivered frame out. The host
recompares the coordinates it expected, then the bytes: exactly for an
unswizzled descriptor, and as a multiset for the swizzled descriptor the sweeps
actually use. The gate runs before any sweep and aborts the job on failure.

The `mma.sync` and `ldmatrix` probes keep their inline PTX. Their point is the
raw warp-level instruction and its documented fragment layout, both already
gated on a numeric check; routing them through a `TiledMMA` would add
partitioning the measurement is meant to exclude.

## Alternatives considered

- Keeping the hand-written atom table and adding a comment naming the CUTLASS
  version it matched. Rejected: the drift it guards against is silent, and a
  comment does not fail a build.
- Inverting the 128 B swizzle on the host so the swizzled TMA check could also
  be byte-exact. Rejected: the probe would then assert the smem layout it is
  supposed to be testing, and a wrong hardcoded permutation fails on correct
  hardware. The unswizzled pass already pins strides, box and walk exactly; the
  swizzled pass covers the descriptor the sweeps use, and its limits are stated
  rather than papered over.
- Replacing the hand-rolled TMA ring with `cutlass::PipelineTmaAsync`. Rejected:
  the pipeline reintroduces producer/consumer semantics that this probe exists
  to strip, which is the isolation the constants depend on.

## Consequences

- Porting the tensor-core probe to another element type or another architecture
  is a template argument rather than a second table.
- `sm90/constants.yaml` is unchanged: the selector resolves to the same eight
  atoms, asserted at compile time, and the measured rates are unmoved.
- The TMA unit's constants are now falsifiable by the probe that produces them.
- The C ABI of both probes is additive (`tma_probe_check`), so the Python
  harnesses' existing bindings are untouched.

## Verification

See `references/protocol.md` for the rules this change enforces, and
`probes/{compute/mma_rate,memory/tma_ring}.{cu,py}` for the implementation.

Toolchain: CUDA 13.1 (V13.1.115), CUTLASS 4.5.1 (`2e602843`), `-gencode
arch=compute_90a,code=sm_90a`.

- Both probes build clean; ptxas reports no `C7515`, so the wgmma pipeline is
  not serialized and the rates are the pipelined ones.
- The emitted PTX contains exactly the eight
  `wgmma.mma_async.sync.aligned.m64n{8,16,32,64,96,128,192,256}k16.f32.bf16.bf16`
  shapes the constants name, and the `static_assert` block makes that a build
  condition rather than a one-time observation.

On `ACD1-13` / `ACD1-50`, H100 80GB HBM3, torch 2.13.0+cu130, clocks NOT pinned:

- Job `560701`, `tma_ring.py --sweeps A`: the Q0 gate passes on all six
  configurations -- `contig`, `stride2k`, `stride8k` x {byte-exact unswizzled,
  multiset SW128} -- with zero coordinate and zero data mismatches.
- Job `560700`, `mma_rate.py --sweeps M1,M2`: `--check` passes at every
  N in {8, 16, 32, 64, 128, 256} through the new `partition_C` mapping, worst
  relative error `1.10e-07`.
- The refactor is behavior-preserving. Cycles per wgmma at N = 8 / 16 / 32 / 64
  / 96 / 128 / 192 / 256 measure 18.9 / 20.7 / 24.7 / 33.0 / 48.9 / 65.0 / 97.1
  / 129.3 against `wgmma.issue.wg.ss`'s recorded 18.9 / 20.7 / 24.7 / 32.8 / 48.7 / 65.0
  / 97.1 / 129.3 -- six identical, the two others inside 0.9%, against a stated
  6% noise floor. `wgmma.stages.wg.knee`'s `wait_group 0` penalty reproduces at 20-30%
  (79.0 vs 42.0 cycles at one group, 55.6 vs 33.1 at two). `constants.yaml` is
  therefore unchanged.
