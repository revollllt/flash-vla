# Vocabulary — every name traces to an authority, or is marked as ours

[`naming.md`](naming.md) governs constant TAGS. This governs every identifier in
the probes: kernel parameters, struct fields, Python variables, JSON keys.

The rule has two halves, and the second is what makes the first honest:

1. **A name for a machine quantity must come from an authority** — the CUDA
   driver API, the PTX ISA, or CUTLASS — cited to a file and line.
2. **A name that is ours must look like ours.** Sweep letters, run modes,
   repetition counts and the coordinate-walk arithmetic have no upstream
   equivalent. Inventing a word there is fine; what is not fine is inventing one
   that *reads* like a machine quantity. `frame` did, and cost a retraction.

## Why this is not tidiness

`frame_b` named three quantities that are already different in this suite:

| | `tma_ring` | `pipeline_ws` |
|---|---|---|
| one TMA box | `frame_b` | A box and B box, different sizes |
| bytes on one barrier | `frame_b` | both boxes summed |
| bytes of one smem stage | `frame_b` | one stage holding both |

It read as sufficient only because `tma_ring` is the degenerate case where the
three coincide. A second box on the same barrier makes the word describe nothing.

## Machine vocabulary

Authorities, all present on this machine:

- **DRV** — `$CUDA_HOME/include/cuda.h`, `cuTensorMapEncodeTiled` parameter names
- **CUTLASS-P** — `include/cutlass/pipeline/sm90_pipeline.hpp`
- **CUTLASS-M** — `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`
- **PTX** — the instruction mnemonic itself

### Descriptor and box geometry

| Use this | Authority | Was |
|---|---|---|
| `tensor_map` | DRV `CUtensorMap *tensorMap` | `map` |
| `tensor_data_type` | DRV `CUtensorMapDataType tensorDataType` | `dtype` |
| `global_dim[]` | DRV `cuuint64_t *globalDim` | `inner`, `outer` |
| `global_strides[]` | DRV `cuuint64_t *globalStrides` | `row_b` |
| `box_dim[]` | DRV `cuuint32_t *boxDim` | `box_inner`, `box_outer` |
| `element_strides[]` | DRV `cuuint32_t *elementStrides` | — |
| `swizzle` | DRV `CUtensorMapSwizzle swizzle` | unchanged |
| `oob_fill` | DRV `CUtensorMapFloatOOBfill oobFill` | — |

`box_dim[0]` is the fastest-varying extent in ELEMENTS. Bytes are always derived
and always named as bytes: `box_bytes = box_dim[0] * box_dim[1] * elem_bytes`.

### Cooperative launch

Authority `RT` = CUDA Runtime API 13.1, `CG` = `cooperative_groups` (CUDA
Programming Guide, "Cooperative Groups"). Every name below is the vendor's.

| Use this | Authority | Not |
|---|---|---|
| `grid_group` | CG `cooperative_groups::grid_group` | `grid`, `coop_group` |
| `this_grid()` | CG `cooperative_groups::this_grid()` | — |
| `grid_sync` | CG `grid_group::sync()` | `gridbar`, `global_barrier` |
| `num_blocks` | CG `grid_group::num_blocks()` | `n_ctas` in this unit only |
| `cudaLaunchCooperativeKernel` | RT, the only legal launch for `grid_sync` | `<<<>>>` |
| `max_active_blocks_per_sm` | RT `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `occ` |
| `cudaDevAttrCooperativeLaunch` | RT, the support query | — |

`grid_sync` is a *device-wide* barrier and is only defined when every block is
co-resident, which is why the block count is bounded by
`max_active_blocks_per_sm * sms` and why the launch API is a separate call that
can refuse. `cluster` names stay distinct: a cluster barrier synchronises a
cluster, not the grid.

### Pipeline and staging

| Use this | Authority | Was |
|---|---|---|
| `stages` | CUTLASS-P:277 `static constexpr uint32_t Stages` | `depth` |
| `stage` (an index) | CUTLASS-M:383 `int write_stage = smem_pipe_write.index()` | `s` |
| `transaction_bytes` / `txn_bytes` | CUTLASS-P:293 `uint32_t transaction_bytes` | `frame_b` |
| `stage_bytes` | derived from `Stages` | `frame_b` |
| `num_producers` | CUTLASS-P:297 `uint32_t num_producers` | `n_warps`, `n_prod` |
| `num_consumers` | CUTLASS-P:296 `uint32_t num_consumers` | — |
| `smem_pipe_write` / `smem_pipe_read` | CUTLASS-M:313, mainloop | — |
| `phase` | CUTLASS-P `PipelineState::phase()` | `ph` |
| `k_tile_count` | CUTLASS-M:316 `int k_tile_count` | `trip` |
| `k_tile_iter` | CUTLASS-M:316 | — |
| `K_PIPE_MMAS` | CUTLASS-M:264 | `WAIT` |

### Instructions

| Use this | Authority | Was |
|---|---|---|
| `warpgroup_arrive` / `_commit_batch` / `_wait` / `_fence_operand` | CUTLASS `cute/arch/mma_sm90_gmma.hpp` | hand-written asm |
| `ss_op_selector`, `rs_op_selector` | CUTLASS `cute/arch/mma_sm90.hpp:366` | `AtomFor<N>` macro table |
| `expect_tx`, `try_wait`, `arrive` | PTX `mbarrier.*` | — |
| `cp_async_bulk_tensor` | PTX `cp.async.bulk.tensor` | `tma_2d` |

## Our vocabulary — legitimate, but must not read as machine words

These have no upstream equivalent. Keep them, and keep them obviously ours by
scoping them to the harness rather than to a kernel parameter that looks like a
hardware quantity.

| Ours | What it is | Why no authority exists |
|---|---|---|
| `sweep`, sweep letters | which experiment | a harness concept |
| `mode` | which side of a unit runs | a harness concept |
| `reps`, `regime`, `l2_ratio`, `touched_b` | measurement policy | this skill's own protocol |
| `mask0/shift0/step0/mask1/step1` | the coordinate walk's shift-and-mask arithmetic | a probe-internal way to avoid an IDIV on the issue path; no upstream analogue |
| `cfg` | index into a unit's config table | a harness concept |
| `dbg`, trap `site` | watchdog reporting | this skill's rule 9 |

The walk arithmetic is the honest borderline case: it computes tensor
COORDINATES, which are a machine concept, but the shift/mask encoding is ours.
The coordinates themselves take the machine name — `coord[0]`, `coord[1]`,
matching the PTX operand order in `cp.async.bulk.tensor.2d ... [{%2, %3}]`.

## Banned

Never appears in probe code, in a tag, or in a JSON key:

| Banned | Because | Use |
|---|---|---|
| `frame` | invented; conflates box / transaction / stage | one of the three |
| `trip` | invented | `k_tile_count` |
| `depth` for a ring | inconsistent with `Stages` | `stages` |
| a cache name as a quantity | `tma.bw.dev.l2` named a cache, not what was measured | the quantity, with the source as a condition |
| a PTX mnemonic as a constant name | spelling the instruction names what was run, not what was measured | the quantity |

## How this is kept

By this file, read before naming something -- not by a script. A checker would
have to be taught which comments explain a banned word and which use it, and
that exception list is a second thing to maintain for a rule one page can state.

So: **a name that is genuinely new goes in the table above, with its citation,
before it goes in the code.** If no authority can be cited, it belongs in the
"ours" table instead, and the act of writing it there is the moment to ask
whether it reads like a machine quantity. `frame` would not have survived that
question.

Current debt, to be paid with the restructure rather than as its own pass --
`frame_b` splits into three fields, so the signatures change anyway:

| Word | Uses left | Becomes |
|---|---|---|
| `depth` | 124 | `stages` |
| `frame_b` | 95 | `box_bytes` / `txn_bytes` / `stage_bytes` |
| `n_warps` | 70 | `num_producers` |
| `trip` | 69 | `k_tile_count` |
| `frame` | 35 | as `frame_b` |

Counted 2026-08-29, **code only**. Every documented surface is paid: constants
and their prose, the unit and category pages, `SKILL.md`, and the protocol.
Three constant tags were retired a second time in the same pass --
`tma.depth.warp.knee`, `wgmma.depth.wg.knee`, `mma.depth.warp.knee` -- because
they contradicted `pipeline.stages.wg.knee` while naming the same quantity.

What remains is confined to the six files still awaiting migration:
`probes/units/tma_ring/tma_ring.{cu,py}`, `probes/units/overlap/overlap.{cu,py}`,
`probes/units/mma_rate/mma_rate.{cu,py}` and `probes/units/pipeline_ws/pipeline_ws.{cu,py}`.
The count rises rather than falls against the earlier estimate because that one
counted a single spelling per site; these are whole-word counts across both
languages.

Two matches in migrated code are NOT debt and must not be "fixed": `stack
frame` is the compiler's term for a local-memory allocation, and the retired
words appear in `hut/unit.hpp` and `hut/abi.py` inside the sentences that
prohibit them. A blanket rename in this pass turned `round trip` into `round
k_tile_count` and `stack frame` into `stack box` in five places before review
caught them -- rename with word boundaries AND read the diff.
