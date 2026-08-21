# Spec schema

Field reference for `assets/spec-template.md`. Read this before the first
interview question — the **arch capability table** at the bottom decides which
sections apply at all.

## Contents

- [Status and the gate](#status-and-the-gate)
- [0. problem](#0-problem)
- [1. grid / CTA](#1-grid--cta)
- [2. mainloop](#2-mainloop)
- [3. pipeline](#3-pipeline)
- [4. warp_groups](#4-warp_groups)
- [5. math (iters)](#5-math-iters)
- [5b. non_mma](#5b-non_mma)
- [6. epilogue](#6-epilogue)
- [7. checks](#7-checks)
- [8. handover](#8-handover)
- [The prose body](#the-prose-body)
- [Arch capability table](#arch-capability-table)

## Status and the gate

`status` has exactly three values and only one transition an agent may make:

| Value | Meaning | Who sets it |
|---|---|---|
| `draft` | Being written; fields may be `TODO` | agent, on creation |
| `review` | Complete and self-consistent; waiting on a human | agent, after all checks pass |
| `approved` | A human signed off; Phase 2 may begin | human, or agent transcribing an explicit human approval into `approved_by` |
| `reference` | Documents a kernel that already exists. No sign-off to record, no Phase 2 to unblock; `open_questions` may stay non-empty. **The only correct status for a reverse-engineered spec** | agent |

An agent never moves `draft` → `approved`. An agent may move `review` back to
`draft` when review turns up something that needs rework.

## 0. problem

| Field | Required | Notes |
|---|---|---|
| `arch` | yes | The compile target, not the marketing name. `sm90a` (not `sm90`) whenever wgmma or TMA is used — the `a` suffix gates those instructions. `sm100a` likewise for tcgen05. |
| `problem.op` | yes | One line of math. Include the scaling and masking, since those are where fused kernels differ. |
| `problem.dims` | yes | Use the names the mainloop will use. `dynamic` lists which are runtime values. |
| `problem.dynamic` | yes | `[]` is a real answer and an important one — all-static shapes unlock unrolling and constant folding. |
| `problem.dtypes` | yes | Separate `acc` from `d`: accumulating in fp32 and storing bf16 is the common case and must be visible. |
| `problem.layouts` | yes | Enough to determine the MMA's major-ness. `row(M,K)` / `col(N,K)` / `paged(page=64)` / `k-major` / `mn-major`. |
| `problem.regime` | yes | What the kernel is tuned for. A kernel tuned for M=1 decode and one tuned for M=8192 prefill share no tile sizes; without this the reviewer cannot judge any other number. |

## 1. grid / CTA

| Field | Required | Notes |
|---|---|---|
| `grid.mode` | yes | `wave`: one CTA per output tile, grid sized by the problem. `persistent`: grid sized by the machine, each CTA loops over tiles. Persistent is what lets a CTA amortise prologue cost and keep weights resident; it also forces you to specify the scheduler. |
| `grid.ctas` | yes | The formula, not a number, when shapes are dynamic. |
| `grid.shape` | yes | What each `blockIdx` component means. Multi-dimensional grids are common in attention (`(m_blocks, kv_heads, splits)`). |
| `grid.cta_tile` | yes | The **output** tile one CTA owns. Not the smem tile — those differ when the reduction axis is tiled separately. |
| `grid.rasterization` | yes | How a linear CTA id becomes a tile coordinate. This is an **L2-locality decision, not bookkeeping**, and it must carry an argument, not just a name. State: which operand consecutive CTAs share, how much of it is co-resident in L2 at once, and what the resulting DRAM read order is over the weight slab. `row-major` is an answer only if you can say why it beats the alternatives here. |
| `grid.l2_schedule` | when the workload is static | **A fully static graph makes tile order an offline optimisation, not a runtime heuristic.** Fixed shapes, a captured graph and a fixed layer order mean the CTA→tile map, and the order kernels touch the weight slab, can be solved once and baked in — as index arithmetic derived offline, or a table in constant memory. Say whether it was solved or defaulted; "defaulted to linear" is an acceptable answer and an unacceptable omission. |
| `grid.persistence` | when `mode: persistent` | The four things a persistent grid has to answer, and none of them is optional. **(a) Grid size vs residency**: a grid smaller than `SM_count x cta_per_sm` is not persistent in the useful sense — the scheduler spreads it one CTA per SM and the extra capacity is never used. State the intended CTAs/SM and the grid that realises it. **(b) The tile scheduler**: how a CTA gets its next tile, and in what order (this is also `grid.rasterization`). **(c) Whether it needs a residency guarantee**, i.e. cooperative launch — required only if CTAs wait on each other. **(d) The ordering mechanism** between phases: grid barrier, per-tile semaphores, or none. |
| `grid.cooperative` | when residency is required | `true` only when CTAs block on each other, because it is expensive in constraints. **Combined with a cluster it needs an explicit placement check.** A cooperative launch is rejected with `cudaErrorCooperativeLaunchTooLarge` when the grid exceeds what can be co-resident, and with a cluster that ceiling is `cudaOccupancyMaxActiveClusters x cluster_size` rather than `SM_count x cta_per_sm`. Deep pipelines hit it routinely (measured: cluster 8 at 207360 B gives 15 placeable against 16 needed); small clusters do not. It is a budget, not a prohibition. Cooperative launch *is* capturable into a CUDA graph; so is a plain cluster launch. |
| `grid.cluster` | sm90+ | Shape and which operand is multicast. Multicast is why clusters exist in a GEMM: one TMA load feeds every CTA in the cluster, halving A (or B) traffic. Delete on sm80 with a reason. |
| `grid.launch.threads` | yes | Must equal `sum(warp_groups.threads)` and match `__launch_bounds__`. |
| `grid.launch.cta_per_sm` | yes | Usually 1 for a Hopper warp-specialized kernel — smem and registers leave no room for a second. State it anyway; it is what makes the occupancy check meaningful. |
| `grid.launch.smem_B` | derived | From `pipeline`. |
| `grid.launch.max_regs_per_thread` | yes | The cap the launch bounds imply, before any `setmaxnreg` redistribution. |

## 2. mainloop

The CTA's serial loop over the reduction axis. One iteration = one pipeline
stage.

| Field | Required | Notes |
|---|---|---|
| `mainloop.axis` | yes | `K` for GEMM, `kv_seqlen` for attention. When two axes could be reduced, saying which one is tiled in the mainloop and which is unrolled inside a stage is a real design decision. |
| `mainloop.step` | yes | Extent consumed per iteration. For FP8 GEMM with per-128-channel scaling this is forced to 128 by the scale granularity — note the constraint when there is one. |
| `mainloop.trip_count` | derived | `ceil(extent / step)`. |
| `mainloop.tail` | yes | `predication` / `pad` / `none-needed (extent % step == 0)`. The most commonly skipped field, and a correctness bug when skipped. |
| `mainloop.operands_per_iter` | yes | Every tile that moves per iteration, with bytes. `via` decides the barrier kind: TMA needs a transaction-count mbarrier, `cp.async` needs a commit-group. |
| `mainloop.loop_carried` | yes | What survives an iteration. For GEMM, the accumulator. For flash attention, accumulator **plus running max and running sum** — forgetting these is how attention specs silently become wrong. |
| `mainloop.per_iter_math` | yes | A pointer, not a description: name the `non_mma` entries that fire each iteration. The detail lives in section 5b, because a one-line string is how a computation worth a quarter of the kernel's cycles hides in a spec. |

## 3. pipeline

| Field | Required | Notes |
|---|---|---|
| `pipeline.depth` | yes | Number of buffered stages. Bounded above by smem, below by the load latency you need to hide (below 3 there is nothing to overlap). |
| `pipeline.stage_index` | yes | Usually `iter % depth`. Write it explicitly — it is what the code indexes with. |
| `pipeline.phase` | sm90+ | mbarrier waits take a phase bit that flips each time the barrier completes a full cycle: `(iter / depth) & 1`. Getting this wrong deadlocks, and it is invisible in a spec that omits it. |
| `pipeline.prologue` | yes | How many stages are filled before the steady state, and by whom. |
| `pipeline.staged_buffers` | yes | Per-stage buffers with shape, dtype, bytes, and swizzle mode. Swizzle follows from tile width × dtype (128B swizzle for a 128-byte row) — derive it, do not ask. |
| `pipeline.non_staged_buffers` | yes | Carries `swizzle` too — an unswizzled epilogue buffer is a common bank-conflict source. Single-instance smem: epilogue staging, scale scratch, reduction workspace. When any of it **aliases** a staged buffer (FlashMLA overlaps its output buffer with K), say which and why it is safe — that is a real correctness argument, not a note. |
| `pipeline.barriers` | yes | Each barrier: kind, how many, initial arrive count, who arrives, who waits. The arrive count is the field people get wrong: an empty barrier released by N consumer warp groups across a cluster of C CTAs initialises to `N × C`, not 1. |

## 4. warp_groups

Delete on sm80 with a reason (`# no warp specialization: sm80 has no async
barriers or setmaxnreg`). On sm90+ this section is where the kernel's shape
actually lives.

Two distinct patterns, and conflating them is the most common spec error:

**Producer/consumer.** One warp group does nothing but issue TMA loads; the
others do nothing but MMA. The producer runs at minimum registers
(`setmaxnreg` dealloc to 24–40) so the math groups can take 232–240. Barriers
are full/empty pairs on the staged buffers. This is DeepGEMM, and most CUTLASS
Hopper collectives.

**Cooperating math groups.** Every warp group does MMA; they divide the *work*,
not the *role*, and synchronise through named barriers at points where one
group's output feeds the other's input. Registers are symmetric. This is
FlashMLA's seesaw, and FA3's ping-pong. TMA is issued by whichever group is
free at the right moment rather than by a dedicated group.

| Field | Required | Notes |
|---|---|---|
| `id` | yes | `producer` / `math0` / `math1` / `epilogue`. |
| `warps`, `threads` | yes | wgmma requires whole warp groups: 4 warps / 128 threads. |
| `regs` | sm90+ | Post-`setmaxnreg`. Omit when registers are not reconfigured, but then say so. |
| `role`, `issues` | yes | `issues` is the concrete instruction list — `cp.async.bulk.tensor` + `mbarrier.arrive.expect_tx`, or `wgmma.mma_async` + `st.shared`. |
| `elected` | yes | `true` when one elected thread issues for the whole group (TMA and `tcgen05.mma` both work this way). Changes the code shape enough to be worth a field. |
| `inter_group_sync` | yes | Named barriers between groups and what each one orders. For cooperating math groups this *is* the algorithm. |

## 5. math (iters)

One entry per distinct MMA step. A GEMM has one; flash attention has at least
two (QKᵀ and PV) with different shapes, different operand sources, and
different accumulators.

| Field | Required | Notes |
|---|---|---|
| `group` | yes | Which warp group runs it. |
| `stage_phase` | when >1 | Where in the stage this fires, for stages with several ordered math steps. |
| `unit` | yes | `wgmma.mma_async` (sm90, warp-group), `mma.sync` (sm80, warp), `tcgen05.mma` (sm100, CTA-pair). |
| `inst_shape` | yes | The **hardware** instruction shape, not the tile. wgmma is `m64nNk16` for 16-bit and `m64nNk32` for 8-bit, N a multiple of 8 up to 256. Getting M=64 wrong is why a spec's register math stops balancing. |
| `count_per_stage` | derived | `mainloop.step / inst_shape.K`. **This is the "iter" count** — the number the whole format exists to make explicit. |
| `a_source`, `b_source` | yes | `smem-desc` (wgmma reads smem through a matrix descriptor), `rf` (A only, the `rs` variants), `tmem` (sm100). Decides register pressure and whether a smem round-trip is needed for an intermediate. |
| `acc.name` | yes | The identifier the L3/L4 nest uses. Without it a stage-local accumulator appears only in prose and `traceability` cannot close on it — found by walking the DeepGEMM example. |
| `acc.location` | yes | `RF` on sm80/sm90, `TMEM` on sm100. |
| `acc.elems_per_thread` | derived | `inst M × inst N / 128` per wgmma for a warp group. Cross-check against `checks.acc_registers`. |
| `acc.cleared` | yes | Whether the first iteration zeroes the accumulator (`ScaleOut::Zero`) or accumulates onto a carried value. |
| `accumulate_across_iters` | yes | Whether the accumulator carries across mainloop iterations, or is drained each iteration (DeepGEMM drains into a separate fp32 accumulator every K-block so it can apply per-block scales). |
| `after_batch` | yes | What runs after `warpgroup.commit_batch` + `wait`: the barrier arrival that frees the stage, the dequant promotion, the softmax. Ordering here is what overlaps CUDA cores with tensor cores. |

## 5b. non_mma

Every computation that is not an MMA, one entry each. A plain GEMM writes
`non_mma: []`; an attention kernel has at least the online rescale; a fused
kernel usually has three or four and they are collectively where its CUDA-core
time goes.

| Field | Required | Notes |
|---|---|---|
| `id`, `where`, `kind` | yes | `where` is the schedule slot, and it is the field that decides whether this work can be overlapped at all. Work in `mainloop.per_iter` competes with the MMA for issue slots; work in `epilogue` does not. |
| `over` | yes | The axis and its extent. A row reduction over `BLOCK_N=64` and one over `BLOCK_N=256` are different kernels. |
| `span` | yes | `lane` / `warp` / `warpgroup` / `cta` / `cluster`. This is what picks the primitive and sets the cost — a lane-local reduction is free, a cluster-wide one costs a barrier (see the arch table). Getting `span` wrong is the most common way a reduction's cost is underestimated by an order of magnitude. |
| `primitive` | yes | `shfl.bfly` / `redux.sync` / smem tree / DSMEM / atomics. Name it; "a reduction" is not an implementation. |
| `loop_carried` | yes | What survives the iteration. Online softmax carries a running max **and** a running sum, and the accumulator rescale that keeps them consistent is itself a `non_mma` entry — specs that list only the accumulator are the ones that turn out wrong. |
| `dtype` | yes | The compute dtype **and where each rounding lands**. This is the numerical contract: the same algebraic operation in bf16 in smem and in fp32 on the accumulator are different functions, and the parity reference must mirror whichever the kernel does. |
| `cost` | yes | Ops per thread and the unit that runs them. This is the number that populates L3's CUDA-core column; without it the timeline is decorative. |
| `touches` | yes | Buffers read and written, with their layout. A reduction's access pattern is usually *not* the MMA's, so the swizzle that gives the MMA zero bank conflicts can give this entry many — that cross-check is the point of the field. |
| `on_critical_path` | yes | Does it sit between a load and the MMA that consumes it? If yes, the L3 timeline must show what the copy engine is doing during it, and the answer had better not be "nothing". |

### Recurring computations get a name

`online_softmax`, `row_rms` and `split_reduce` are contracted in
`references/primitives.md`. Reference the name plus its parameters; the
primitive fixes the algorithm and its hazards, the schedule stays yours.

```yaml
  - id: softmax
    primitive: online_softmax
    params: {rows: BLOCK_M, block: BLOCK_N, span: lane, first_iter: specialised,
             masked_rows: clamped, p_cast: bf16}
    where: mainloop.per_iter
    loop_carried: [m, l]
    on_critical_path: "yes -- acc_o *= alpha must precede this iteration's P@V"
```

Referencing a primitive does not excuse the per-kernel fields: `where`, `cost`,
`touches`, `span` and `on_critical_path` still have to be filled in. The
primitive fixes the *algorithm* and the *hazards*; the schedule is yours.

## 6. epilogue

| Field | Required | Notes |
|---|---|---|
| `position` | yes | `after-mainloop` / `fused-per-iter` / `split-k-partial`. |
| `math` | yes | Scaling, bias, activation, residual, cast — in order. |
| `path` | yes | `rf -> smem -> TMA-store` is the Hopper default (TMA store needs the data in smem); `rf -> gmem` skips smem at the cost of uncoalesced stores. |
| `output` | yes | Tile and dtype. |
| `split_reduction` | yes | `none` / `atomics` / a named combine kernel. Split-K and split-KV both need this, and the combine kernel needs its own spec. |

## 7. checks

Fill each with the computed value **and** a verdict, not just `ok` — the number
is what a reviewer scans for. `smem: 230808 B / 232448 B cap, 1640 B spare —
PASS (tight)` is useful; `smem: ok` is not.

The full check list and the formulas are in SKILL.md. Two worth expanding:

**acc_registers.** For an fp32 accumulator held in RF: `elems_per_thread =
cta_tile.M × cta_tile.N / math_threads`. A 128×128 tile over 256 math threads
is 64 registers per thread just for the accumulator; add operand fragments,
descriptors, and addressing, and 255 arrives fast. If the number exceeds ~200,
say so — that is the register-spill cliff, and spills on a Hopper mainloop cost
more than the tile size gains.

**arithmetic_intensity.** `2·M·N·K / bytes_moved` for one CTA tile versus the
arch ridge point (peak FLOP/s ÷ peak byte/s; ~295 FLOP/byte for H100 SXM5 bf16
dense at published peaks). A tile below the ridge is memory-bound *by
construction* and no amount of pipelining will fix it — that finding belongs in
the spec review, not in a profiler run three days later.

### `open_questions` in reverse-engineering mode

The "must be empty before `status: review`" rule applies to a **new** kernel,
where an open question is an undecided design choice. When reverse-engineering,
the term means "the source does not settle this" — those entries are findings
and may survive into `approved`. Mark them `[I]` and say what would settle them.

## 8. handover

| Field | Required | Notes |
|---|---|---|
| `verification.reference` | yes | What the kernel is checked against, concretely. |
| `verification.tolerance` | yes | fp8 and bf16 accumulation do not reproduce fp32 bitwise; naming the tolerance up front prevents an argument later. |
| `verification.perf_target` | yes | Metric, number, and how it is measured. "Fast" is not a target. |
| `open_questions` | yes | Must be `[]` before `status: review`. |
| `deviations` | Phase 2 | Where the generated code could not honour the spec, and why. |

## The prose body

Below the YAML the spec carries four prose sections. They are not decoration —
the YAML is what an agent checks, the prose is what a human reads.

**`## Loop nest`** — the L1/L2/L3/L4 pseudocode: iteration space, then the
tiled mapping, then the per-stage engine schedule, then the lowered
instructions with their per-thread access. Notation and rules
are in SKILL.md. This is the section a reviewer reads first, so it must stand
alone: someone who has not read the YAML should be able to see the interface,
the tiling, the loop bounds, and the instruction sequence from it. Every `range`
states start, stop, and step with the trip count in a comment — a bare
`for k in range(N)` hides exactly the two numbers a reviewer is checking. Every
extent here must also appear in the YAML; the block is a readable projection of
the fields, never a second source of truth that can drift.

**`## Warp-group choreography`** — only for kernels whose warp groups **interact**: a seesaw, a
ping-pong, any hand-off through smem under a named barrier. It shows the
ordering the loop nest cannot — who waits on whom, in what sequence, through
which barrier. When the split is plain producer/consumer the L2 nest already
shows every ordering that exists, and the section should be deleted with that
reason stated (see `example-deepgemm.md`). When the groups cooperate, this
section *is* the algorithm (see `example-flashmla.md`).

**`## Why these numbers`** — one short paragraph per non-obvious choice. Make
the reasoning attackable rather than asserting the conclusion; this is the
section the reviewer disagrees with, and a disagreement here is cheap while the
same disagreement after the kernel exists is not.

**`## Known risks`** — whatever `checks` reported as tight, plus tail behaviour
and register pressure. "Nothing is tight" is a valid entry and worth writing.

## Arch capability table

Ask only what the target arch has.

| | **sm80 / sm86 / sm89** (Ampere, Ada) | **sm90a** (Hopper) | **sm100a** (Blackwell) |
|---|---|---|---|
| Bulk load | `cp.async` (LDGSTS), per-thread, ≤16 B each | TMA `cp.async.bulk.tensor`, one elected thread moves a whole tile; multicast to a cluster | TMA, plus `tcgen05.cp` smem→TMEM |
| Load sync | `cp.async.commit_group` / `wait_group N` | mbarrier with transaction count + phase parity | same as sm90 |
| MMA | `mma.sync.m16n8k16`, per **warp**, A and B in RF | `wgmma.mma_async.m64nNk16/32`, per **warp group**, A in smem-desc or RF, B in smem-desc | `tcgen05.mma`, per **CTA pair**, operands in smem/TMEM |
| Accumulator | RF | RF | TMEM (`tcgen05.ld` / `.st` to move it) |
| Register control | none | `setmaxnreg` alloc/dealloc per warp group | same as sm90 |
| Warp specialization | rare — no async barriers to build it on | the norm: producer/consumer or cooperating math groups | the norm: load / MMA / epilogue groups, MMA issued by one elected thread |
| Cluster / DSMEM | no | yes, ≤16 CTAs, distributed smem, multicast | yes, plus 2-CTA MMA pairs |
| Smem per CTA | **163 KB** sm80 / **99 KB** sm86,sm89 (164/100 KB are the per-SM figures) | 227 KB (232448 B; the 233472 B figure is per SM, not per CTA) | 227 KB (232448 B; the 233472 B figure is per SM, not per CTA) |
| Registers per SM | 65536 × 32-bit | 65536 × 32-bit | 65536 × 32-bit |

Consequences for the interview:

- **sm80**: no `warp_groups`, no `cluster`, no `phase`. Pipeline depth is bounded by 163 KB per CTA and by `cp.async` in-flight limits. `mma.sync` is per-warp, so `inst_shape.M = 16` and the tile is split across warps in both M and N — the M-split check in SKILL.md needs the per-warp tiling stated instead.
- **sm90a**: everything in the template applies. The `a` suffix is mandatory; `sm90` alone will not compile wgmma or TMA.
- **sm100a**: `acc.location` is `TMEM`, which changes the register math completely — the accumulator no longer competes for RF, so larger tiles become possible. Add TMEM allocation/deallocation to `pipeline.barriers`, and note whether MMA is 1-CTA or 2-CTA.
