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
| `contracts` | yes | The axis **this** MMA reduces, as a name from `problem.dims` or `name=extent`. For a GEMM it is `mainloop.axis` and the field looks redundant; in a fused kernel it routinely is not, and that is the case it exists for. FlashMLA's QKᵀ contracts `d_k`=576 while the mainloop walks `kv_seqlen`, so `count_per_stage: 36` is correct and looks wrong against `mainloop.step`. Without the field the `mma_k` check has to be waived by hand, which is how a real mismatch would get waived too. |
| `count_per_stage` | derived | `extent(contracts) / inst_shape.K`. **This is the "iter" count** — the number the whole format exists to make explicit. |
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
| `span` | yes | `lane` / `warp` / `warpgroup` / `cta` / `cluster`. This is what picks the mechanism and sets the cost — a lane-local reduction is free, a cluster-wide one costs a barrier (see the arch table). Getting `span` wrong is the most common way a reduction's cost is underestimated by an order of magnitude. |
| `primitive` | yes | A **name from `references/primitives.md`** (`online_softmax`, `row_rms`, `split_reduce`), or `none` when the computation is bespoke. Fixes the algorithm, its state and its hazards. |
| `mechanism` | yes | *How* it is computed: `shfl.bfly` / `redux.sync` / smem tree / DSMEM / atomics. Name it; "a reduction" is not an implementation. **Distinct from `primitive`** — these were one key until a spec was parsed and the earlier value turned out to be silently discarded. |
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
    primitive: online_softmax          # WHAT -- the contract in primitives.md
    mechanism: "lane-local max and sum; no shfl -- the fragment keeps a row in one lane"
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

Do not evaluate these by hand. `python3 scripts/budget.py <spec.md> --sms N`
computes every arithmetic row and prints the number beside its verdict;
`--gate` additionally fails on `SKIP`, which is what `status: review` must
clear. Rows it reports as `MANUAL` are the ones that are not arithmetic — L3's
bubble check, traceability, the rasterization argument — and they still need a
human.

**`l4_accesses`** sits beside `checks` and names the YAML file holding the
layout and thread-value maps for every per-thread touch. `scripts/tv_check.py`
turns those two maps into L4's table, so the widths and conflict counts are
computed rather than claimed; `budget.py` follows the field and runs it.
Format in `references/l4-access.md`, worked files in
`references/accesses-deepgemm.yaml` and `accesses-flashmla.yaml`. `none` is
legal when nothing in the kernel is touched per-thread — a pure TMA + wgmma
mainloop — but say so rather than deleting the field.

The full check list and its formulas follow — this is the canonical version,
and `budget.py` implements it. The rows it reports `MANUAL` (L3's bubble check,
traceability, the rasterization argument) are the ones that are not arithmetic
and still need a human.

| Check | Must hold |
|---|---|
| smem | `depth × per_stage_bytes + non_staged_bytes ≤ arch smem/CTA` (232448 B = 227 KB per CTA on sm90/sm100 -- the per-SM figure is 233472 B and is a different number, 164 KB sm80) — and if aliasing is used, say which buffers alias and why that is safe |
| threads | `sum(warp_groups.threads) == launch threads`, each group a multiple of 128 when it issues wgmma |
| acc_registers | fp32 accumulator elems/thread `= cta_tile.M × cta_tile.N / math_threads`; plus operands and addressing must fit 255 (sm90 RF) — or live in TMEM (sm100) |
| mma_k | `iters_per_stage × inst_shape.K == extent of the axis this MMA contracts`, named in `math[].contracts`. That axis is usually `mainloop.axis`, and in a fused kernel often is not: attention's QKᵀ contracts the head dim while the mainloop walks `kv_seqlen`, so without the field a correct `count_per_stage` looks wrong |
| mma_m | `inst_shape.M × num_math_groups == cta_tile.M` (or state the split rule that replaces it) |
| mma_n_legal | `inst_shape.N` is a legal atom for the arch/dtype (wgmma N ∈ {8,16,…,256} step 8) |
| trip_count | `mainloop.trip_count × mainloop.step ≥ problem reduction extent`, and the tail policy is named |
| output_coverage | grid tiles × cta_tile covers the output exactly; predication/masking named for ragged edges |
| occupancy | `cta_per_sm` implied by smem and registers matches what the spec claims; for persistent grids `ctas == num_sms × cta_per_sm` |
| barrier_arrivals | every stage buffer has a full and an empty barrier (or the stated equivalent); arrival counts match the number of arriving warps/CTAs; the phase-flip rule is written down |
| traceability | every bound and extent at L4 traces to an L2 loop, every name at L2 traces to an L1 dim, and the shared inner name of each `@` is `mainloop.axis` — an untraceable number means a missing field |
| loop_bounds | every `range` in the nest states start, stop, and step, and its trip count matches the corresponding YAML field (`mainloop.trip_count`, `math.count_per_stage`) |
| arithmetic_intensity | tile arithmetic intensity `≈ 2·M·N·K / (bytes moved)` versus the arch ridge point — flags a tile that cannot reach peak no matter how good the code is |
| **floor** | every `perf_target` ≥ the Phase 0 floor for that kernel, computed as `a + MB/b` from measurement 2 **plus one launch cost per kernel launch** — a target below its own floor makes the whole spec unfalsifiable |
| **reference** | measurement 4 is used to *calibrate the floor model*, never as a bound. See Phase 0 measurement 4 |
| **acceptance** | the spec names the *one* measurement that decides acceptance, and it is the one the kernel will ship under. Designing against an isolated cold benchmark and shipping against an in-graph profile is a 2× discrepancy waiting to happen |
| **falsifiability** | every performance claim in `Why these numbers` names the measurement that would refute it. A claim with no refuting measurement is an assumption wearing a number |
| **concurrency** | L3's bubble check is filled in: for a steady-state stage, each of the three engines is either busy or its idle time is named and accepted. |
| **vectorisation** | every gmem and smem touch at L4 states bits/thread and transactions; each is the widest legal access (128 b where alignment allows), coalesced, and bank-conflict-free — or the exception is named with its reason. **Computed, not asserted**: `l4_accesses` names the access file and `tv_check.py` produces the table. Report conflicts against their *ideal*, since two words per bank is optimal for a 64-bit access, not a 2-way defect |
| **addressing** | per-iteration address arithmetic is counted. Anything loop-invariant is hoisted to the prologue and carried in a register; if the mainloop recomputes a base each stage, say why |
| **register_budget** | `threads_per_cta × regs_per_thread × cta_per_sm ≤ 65536`, computed. This is what forces `cta_per_sm` in practice, and `acc_registers` (per thread) does not cover it. **But that product is the wrong one as soon as `setmaxnreg` is in play**: the launch bound's `max_regs_per_thread` is then a pre-redistribution ceiling, not the allocation. DeepGEMM's 384 × 255 "needs" 97920 registers while the kernel allocates 24/240/240 → 64512 and fits. When any warp group states `regs`, the binding sum is `Σ warps × 32 × regs ≤ 65536 / cta_per_sm` |
| **residency** | `cta_per_sm` is stated with the smem *and* register arithmetic that produces it, and for a latency-bound kernel a value of 1 is justified against the 2-or-4 alternative rather than inherited from the warp-specialisation idiom (see `references/residency.md`) |
| **persistence** | if `mode: persistent`, the grid is at least `SM_count x cta_per_sm` or the shortfall is named; `cooperative` is `true` only where CTAs block on each other, and when combined with a cluster the grid is checked against `cudaOccupancyMaxActiveClusters x cluster_size`; any semaphore is self-resetting under graph replay |
| **tile_order** | `grid.rasterization` carries an L2 argument, not just a name, and on a static graph `grid.l2_schedule` says whether the map was solved offline or defaulted |
| **non_mma_accounting** | every `non_mma` entry appears in L3's CUDA-core column with its `cost`, and every name in its `loop_carried` also appears in `mainloop.loop_carried`. Any entry with `on_critical_path: yes` shows what the copy engine is doing during it |
| **rounding_contract** | every `non_mma.dtype` says where each rounding lands, and the parity reference in `verification` mirrors it. An algebraically identical op at a different precision is a different function and will not compare |

When a check fails, the spec is wrong, not the check. Fix the spec or ask.

Two rows worth expanding:

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

- **sm80**: no `warp_groups`, no `cluster`, no `phase`. Pipeline depth is bounded by 163 KB per CTA and by `cp.async` in-flight limits. `mma.sync` is per-warp, so `inst_shape.M = 16` and the tile is split across warps in both M and N — the `mma_m` check (§7) needs the per-warp tiling stated instead.
- **sm90a**: everything in the template applies. The `a` suffix is mandatory; `sm90` alone will not compile wgmma or TMA.
- **sm100a**: `acc.location` is `TMEM`, which changes the register math completely — the accumulator no longer competes for RF, so larger tiles become possible. Add TMEM allocation/deallocation to `pipeline.barriers`, and note whether MMA is 1-CTA or 2-CTA.
