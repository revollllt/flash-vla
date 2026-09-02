# SM90 tile primitives

Header-only CUDA/CuTe building blocks shared by every hand-written H100
kernel in this repository: tile-level copies between global, shared and
register memory, the complete tensor-core instruction tables, and one GEMM
entry that picks the instruction at compile time. Kernels compose these;
they do not spell PTX or CuTe atom names.

Include root: `src/flash_vla/hardware/nvidia/cuda` (build wrappers pass it as
`-I`). Namespace: `flash_vla::sm90`. Umbrella: `tile/sm90/sm90.cuh`; host
tensor-map helpers: `tile/sm90/tma_host.cuh`. Requires CUTLASS/CuTe from
`third_party/cutlass` and `-arch=sm_90a`.

## Interface

| Header | Owns |
| --- | --- |
| `common.cuh` | element aliases (`BF16`, `F16`, `E4M3`, `E5M2`), `Operand` (smem / reg), `Major` (K / MN), thread-role helpers, proxy fences, named barriers |
| `barrier.cuh` | `FullBarrier` (transaction) and `EmptyBarrier` aliases, non-blocking parity test, per-slot `PhaseRing` |
| `smem_layout.cuh` | `SmemTileLayout<T, Rows, Cols, Major, SwizzleBytes, Order>`: the swizzled (MN, K) tile a TMA box, ldmatrix/stmatrix and a wgmma descriptor all agree on; `swizzle_bytes_for<T, InnerElems>()` |
| `copy_g2s.cuh` | global to shared: `tma_load_2d/3d<L2Hint>`, multicast, L2 prefetch, `bulk_load_1d`, per-thread `cp_async_16` with group commit/wait, cp.async tiled copy (`make_g2s_cp_async_copy`, `copy_g2s`), `TmaTile2D` (a tile as swizzle-span boxes plus its `expect` byte count) |
| `copy_s2r.cuh` | shared to registers through ldmatrix derived from the MMA: `make_s2r_copy_A/B`, `copy_s2r` |
| `copy_r2s.cuh` | registers to shared: `convert_fragment<TOut>`, stmatrix copy derived from the MMA (`make_r2s_copy_C`), scalar fallback, `copy_r2s` |
| `copy_s2g.cuh` | shared to global through the async proxy: `tma_store_2d/3d`, `bulk_store_1d`, bulk-group commit / wait / wait-read |
| `mma_sync.cuh` | warp-level `MmaSyncAtom<TA, TB>`: m16n8k16 for bf16/f16, m16n8k32 for every e4m3/e5m2 pair, f32 accumulate (fp8 mma.sync is lowered by ptxas to f16 HMMA on sm_90a; fp8 tensor-core rate needs wgmma) |
| `wgmma.cuh` | `WgmmaAtom<TA, TB, N, LocA, MajA, MajB>`: every 64xNx16 bf16/f16 atom and every 64xNx32 fp8 atom for N = 8..256 step 8, SS and RS |
| `gemm.cuh` | `MmaSelector<TA, TB, TileM, TileN, TileK, LocA, LocB, MajA, MajB, Threads, WarpsN>` and `gemm<ZeroInit, WgWait, Arrive, Commit>(mma, A, B, acc)` plus the `gemm_ss` / `gemm_rs` conveniences |
| `tma_host.cuh` | `encode_tensor_map_2d/3d<T>(...)` for the loads and stores above (host only, not graph-capture safe) |

## Selection contract

`MmaSelector` chooses the instruction from the tile and the operands alone:

- **wgmma** when `Threads` is a whole number of warpgroups, `TileM` is a
  multiple of 64 per warpgroup and B is in shared memory. A in shared memory
  gives the SS form, A in registers the RS form. The instruction N is
  `TileN` up to 256, otherwise the widest table entry dividing `TileN`.
  fp8 requires K-major A and B. Warpgroups stack along M.
- **mma.sync** otherwise. Both operands must be register fragments (staged
  by `copy_s2r`) and K-major; warps tile `TileM` x `TileN` with each warp
  owning 16 x 16 per repetition so the x4 ldmatrix fills both fragments.

A tuple outside the tables fails at compile time with a message naming the
constraint; there is no runtime fallback.

## Invariants the caller keeps

- One elected lane issues each TMA, bulk copy or bulk store; every
  transaction barrier is armed with exactly the bytes the loads it covers
  deliver (`TmaTile2D::kBytes` for a whole tile).
- The swizzle passed to `encode_tensor_map_*` equals the `SmemTileLayout`
  swizzle the consumer reads through, and a swizzled box row does not exceed
  the swizzle width (`TmaTile2D` splits a wider tile into boxes).
- Generic-proxy writes to shared memory are followed by
  `fence_proxy_async_shared()` in every writing thread and a barrier before
  one lane issues a TMA or bulk store from that region. Generic-proxy global
  data acquired from another CTA is followed by `fence_proxy_async_global()`
  in the issuing lane before its first TMA read of it.
- wgmma register A fragments must not be refilled while a batch that reads
  them is in flight: alternate two fragments and fully unroll the stage loop
  (kernel-design wiki, `c7518-wgmma-serialization`).
- `gemm` with `WgWait < 0` leaves the batch in flight; the caller retires it
  with `cute::warpgroup_wait<N>()` before reading the accumulator or
  releasing the operand frames.

## Validation

- Compile-time: every table entry is instantiated by the parity build.
- Device parity: `eval/correctness/tile_sm90/primitives_parity.py` runs one
  CTA per case through g2s, s2r, gemm, r2s and s2g against a float32 torch
  reference (SS/RS wgmma bf16 with K- and MN-major B, fp8 wgmma including a
  mixed pair, two stacked warpgroups, mma.sync bf16 and fp8 through cp.async
  and ldmatrix). Run on a GPU node with
  `sbatch sbatch/pi05_cuda.sh -m eval.correctness.tile_sm90.primitives_parity`.
- Production kernels: the FFN and attention task loops build on these
  headers; their parity scripts (`eval/correctness/pi05/`) are the
  regression gate for any change here, and the build wrappers hash these
  headers so an edit never reuses a stale binary.
