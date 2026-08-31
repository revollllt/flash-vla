# Spec → backend mapping

Read only the section for the backend the user chose. Each section maps spec
fields onto constructs, then lists what that backend **cannot** express — the
limits are the useful part, because they are where a spec quietly stops being
implementable and a deviation has to be recorded.

For API detail beyond the mapping, defer: `cutlass_skill` (CUTLASS / CuTe /
CuTeDSL), `cuda_skill` (PTX, mbarrier, profiling), `triton_skill` (Triton).

## Contents

- [TileLang](#tilelang)
- [CUTLASS / CuTe (C++)](#cutlass--cute-c)
- [CuTeDSL (Python)](#cutedsl-python)
- [Raw CUDA + inline PTX](#raw-cuda--inline-ptx)
- [Recording deviations](#recording-deviations)

## TileLang

| Spec field | TileLang |
|---|---|
| `grid.ctas`, `grid.shape` | `T.Kernel(bx, by, bz, threads=N)` — the block ids come back as the `as (...)` tuple |
| `grid.cta_tile` | `BLOCK_M` / `BLOCK_N` constants closed over by the builder |
| `grid.mode: persistent` | not a primitive; write the loop over tiles yourself inside `T.Kernel` sized to the SM count |
| `grid.rasterization` | index arithmetic on the block id, or `T.use_swizzle(panel_size)` |
| `grid.launch.threads` | `threads=` on `T.Kernel` |
| `mainloop` | `for k in T.Pipelined(trip_count, num_stages=depth)` |
| `mainloop.operands_per_iter` | `T.copy(Global[...], Shared_buf)` — lowered to TMA on sm90 |
| `pipeline.depth` | `num_stages=` on `T.Pipelined` |
| `pipeline.staged_buffers` | `T.alloc_shared(shape, dtype)`; the pass multi-buffers it by `num_stages` |
| `pipeline.barriers` | **generated**, not written. The pipeline pass emits the mbarriers, arrive counts, and phase bits |
| `warp_groups` | **generated**. `T.copy` from global lowers to a producer/consumer split; disable with `TL_DISABLE_WARP_SPECIALIZED` |
| `math.unit`, `inst_shape`, `count_per_stage` | `T.gemm(A_shared, B_shared, C_local)` — the pass picks the wgmma atom and emits the instruction sequence |
| `math.acc.location: RF` | `T.alloc_fragment(shape, accum_dtype)` |
| `epilogue` | ordinary elementwise code on the fragment, then `T.copy(C_local, Out[...])` |
| `problem.dtypes` | `T.bfloat16` / `T.float8_e4m3` / `T.float32`, `accum_dtype` separate |

**What TileLang decides for you.** Barriers, phase bits, the producer/consumer
warp split, the wgmma atom selection, and the swizzle. That is the trade: the
spec's `pipeline.barriers` and `warp_groups` sections become *assertions to
verify against the generated code*, not code you write. Verify them — dump the
generated CUDA (`kernel.get_kernel_source()`) and check the stage count and
warp split match the spec. When they do not, that is a deviation.

**What TileLang cannot express.**

- **Asymmetric or cooperating warp groups.** TileLang's warp specialization is
  producer/consumer over `T.copy`. A seesaw or ping-pong schedule where two
  math groups exchange partial results through named barriers (FlashMLA, FA3)
  has no expression. If the spec's `warp_groups` has two `math*` entries with
  `inter_group_sync`, TileLang is the wrong backend — say so before writing
  code, not after.
- **Hand-placed barriers** between arbitrary program points.
- **Smem aliasing** across buffers with different lifetimes.
- **Per-warp-group register control** (`setmaxnreg`).
- **Fine-grained TMA/GEMM interleave** — issuing 9 separate TMA copies for one
  tile and starting the MMA after the first lands.

**Project note (flash-vla).** TileLang is pinned to 0.1.11; kernels write their
destination through a `T.Tensor` parameter rather than returning a new tensor,
so the wrapper does not pay a device-to-device copy per call. Warp
specialization is a per-kernel decision made through `pass_configs`
(`FAST_MATH` vs `NO_WARP_SPEC` in `kernels/base.py`) because below one wave the
producer warp sits idle and still costs warps and mbarrier traffic — if the
spec's `regime` is sub-wave, start from `NO_WARP_SPEC`. `pass_configs` is part
of the compile cache key.

## CUTLASS / CuTe (C++)

| Spec field | CUTLASS |
|---|---|
| `grid.cta_tile`, `mainloop.step` | `TileShape = Shape<_128,_128,_128>` (M, N, K per mainloop iteration) |
| `grid.cluster` | `ClusterShape = Shape<_2,_1,_1>`; multicast follows from it |
| `grid.mode`, `rasterization` | `TileScheduler` (`PersistentScheduler`, `StreamKScheduler`) and its rasterization order |
| `pipeline.depth` | `StageCount<N>`, or `StageCountAutoCarveout` sized from the smem budget |
| `pipeline.barriers` | `PipelineTmaAsync<Stages>` — `producer_acquire` / `consumer_wait` / `consumer_release`, `PipelineState` carries stage index and phase |
| `warp_groups` | the `KernelSchedule` tag: `KernelTmaWarpSpecializedCooperative` (2 math WGs on one tile), `...Pingpong` (2 math WGs alternating tiles) |
| `math.unit`, `inst_shape` | the MMA atom picked by `CollectiveBuilder` from arch + dtype + TileShape |
| `math.a_source`, `b_source` | `GmmaMajorA/B` and whether the atom is `SS` or `RS` |
| `mainloop.operands_per_iter` | `GmemTiledCopy` = `SM90_TMA_LOAD` / `SM90_TMA_LOAD_MULTICAST` |
| `epilogue` | `CollectiveEpilogue` + an EVT tree for fused ops |
| `epilogue.path` | `SM90_TMA_STORE` via a smem staging tile |

**What CUTLASS cannot express** without dropping below the collective layer:
anything the builders do not have a `KernelSchedule` tag for. `Cooperative` and
`Pingpong` cover the two standard GEMM warp splits; a custom split means
writing your own collective mainloop against the CuTe layer, which is a
different (larger) job than instantiating a builder. Say which layer you are
writing at before you start.

The builders also silently normalise things: a `StageCount` that does not fit
gets carved out, a cluster shape may be adjusted. After compiling, read back
the actual stage count and cluster from the kernel's traits and check them
against the spec.

## CuTeDSL (Python)

Same conceptual mapping as CUTLASS — `TiledMma`, `TiledCopy`, `Layout`,
`make_tiled_mma`, explicit `cute.gemm` calls — with the schedule written out in
Python instead of selected by a builder tag. That makes it the right choice
when the spec's `warp_groups` and `inter_group_sync` are custom: you write the
`cute.arch.barrier` calls and the wgmma issue order yourself, so an asymmetric
schedule is expressible.

Map `math.count_per_stage` onto the explicit `for k_block in range(...)` around
`cute.gemm`, exactly as the spec states it. `pipeline.barriers` become
explicit `mbarrier` init / arrive / wait with the arrive counts from the spec.
`warp_groups[].regs` become `cute.arch.warpgroup_reg_alloc/dealloc`.

Nothing in the template is inexpressible here; the cost is that everything the
CUTLASS builders would have derived — swizzle atoms, descriptor construction,
smem layout — is now yours to get right, and a wrong smem layout fails as
silently wrong numbers rather than a compile error.

## Raw CUDA + inline PTX

Everything is expressible; nothing is provided. Use it when the spec's
`warp_groups` or `pipeline` sections are irregular enough that neither
TileLang's generated split nor a CUTLASS schedule tag fits.

| Spec field | Construct |
|---|---|
| `grid.launch.threads`, `cta_per_sm` | `__launch_bounds__(threads, cta_per_sm)` |
| `grid.cluster` | `__cluster_dims__` or the launch-attribute API; `cute::block_rank_in_cluster()` |
| `pipeline.staged_buffers` | `extern __shared__ __align__(1024) uint8_t smem[]` with hand-computed offsets |
| `pipeline.barriers` | `cutlass::arch::ClusterTransactionBarrier`, or raw `mbarrier.init` / `.arrive.expect_tx` / `.try_wait.parity` |
| `pipeline.stage_index`, `phase` | the explicit `{iter % depth, (iter / depth) & 1}` pair |
| `warp_groups` | `warp_idx = threadIdx.x / 32` branch, plus `warpgroup_reg_alloc<N>()` / `reg_dealloc<N>()` |
| `warp_groups[].elected` | `cute::elect_one_sync()` |
| `inter_group_sync` | `cutlass::arch::NamedBarrier` with an enum of barrier ids |
| `mainloop.operands_per_iter` | `cp.async.bulk.tensor` with a `__grid_constant__ CUtensorMap`, or `cp.async` on sm80 |
| `math.unit` | `wgmma.mma_async` between `warpgroup_arrive()` / `commit_batch()` / `wait<0>()`, with `warpgroup_fence_operand` around the accumulator |
| `math.a_source: smem-desc` | a matrix descriptor built from the smem address, leading-dim byte offsets, and swizzle mode |
| `epilogue.path` | `st.shared` → `tma_store_fence()` → `SM90_TMA_STORE` → `tma_store_arrive/wait` |

The failure modes are concentrated in three places, all of which the spec
already pins down — check the code against the spec on each: barrier **arrive
counts**, the **phase bit**, and **smem offsets and alignment**. A wrong arrive
count or phase deadlocks; a wrong swizzle mode or offset produces wrong numbers
with no diagnostic.

## Recording deviations

When the backend cannot honour the spec, append to the spec's `deviations`
block before finishing:

```yaml
deviations:
  - field: pipeline.depth
    spec: 5
    actual: 4
    reason: TileLang carved a stage out for the epilogue buffer; generated
      source confirms num_stages=4.
    impact: ~8% less latency hiding on the K loop; re-tune if the mainloop
      shows up as memory-stalled.
```

Then say it in the hand-back message too. A deviation recorded only in the file
gets found by whoever debugs the kernel next, which is too late.
