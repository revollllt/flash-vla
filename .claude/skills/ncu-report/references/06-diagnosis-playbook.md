# Diagnosis Playbook — Pattern → Cause → Fix

For each observed NCU signal: what it typically means, what to try first, and when it is a false alarm. Patterns **A–N** are the generic catalogue (from the upstream skill, with Hopper cross-references); patterns **O–V** are the sm90 / this-repo additions — warp-specialized persistent task loops fed by TMA and wgmma — including the cases where the rule engine's advice must be **overruled**.

Read this after gathering the metrics (`05-analysis-dimensions.md`). Machine numbers are cited by their `hardware-unit-test` tag — read the value and validity range with `python3 .claude/skills/hardware-unit-test/scripts/constants.py --tag <t>`; restating them here would let two copies drift.

---

## How to use this doc

For each *observation* below:

- **Signals** — the metric values that flag it.
- **Why** — the underlying cause.
- **First-line fix** — the cheapest change to try.
- **Deeper fixes** — when first-line isn't enough.
- **Exceptions** — kernel types where the pattern is *expected* and should be left alone.

Most kernels match 2–4 patterns at once. **Rank by magnitude** using `Est. Speedup` and the stall-sample breakdown; fix the biggest first. Never sum the estimates.

---

## Pattern A — Small grid / SM idle

**Signals:** `launch__waves_per_multiprocessor < 0.5`; `launch__grid_size < 132`; rule *"The grid for this launch is configured to execute only N blocks, which is less than the M multiprocessors"* with `Est. Speedup` 50–90 %.

**Why:** each CTA occupies at most one SM; with fewer CTAs than SMs, SMs idle throughout.

**First-line fix:** more CTAs — split-K, split across heads/rows, a grid-stride loop if work units are cheap.

**Deeper fixes:** persistent kernel (one CTA per SM dequeuing tasks); fuse with neighbours so one launch carries more work.

**Exceptions:** decode-shaped work (one query token) is fundamentally small — split-K over the KV length is the standard mitigation; the last stage of a multi-level reduction is naturally small — fuse it into the producer. **A persistent task loop with grid == 132 and waves == 1 is not this pattern** (see O).

**Cross-ref:** Hopper principle 1 (`../hopper-cuda-programming.md`).

---

## Pattern B — Tail effect (variable-length or uneven work)

**Signals:** per-SM active cycles spread widely (`WorkloadImbalance` rule: min far below average); timeline shows a long gradual tail; `launch__waves_per_multiprocessor` just over an integer; input work distribution max/avg > 3.

**Why:** a few CTAs (or a few task-queue items) keep running after everyone else finished.

**First-line fix:** sort or pack inputs by length; split long items across more CTAs with a small post-reduction.

**Deeper fixes:** chunkwise processing with a stitch step; classify-and-dispatch (short items on the simple path, long ones chunked). On a task loop: give the reduction its own task kind, let the epilogue owner start early (`kernel-design` wiki `reduction-own-task-kind`, `serial-epilogue-owner`).

**Exceptions:** kernels under ~10 µs where the partial wave is absolutely small; workloads already pre-packed.

**Cross-ref:** Hopper principle 11.

---

## Pattern C — Uncoalesced global loads (LSU path)

**Signals:** `sectors/request > 5` (ideal 4); rules *"uncoalesced global accesses resulting in N excessive sectors"* / *"only Y of the 32 bytes transmitted per sector are utilized"*; `long_scoreboard` on the offending load line.

**Why:** lanes access non-contiguous addresses; hardware fetches sectors only a few lanes use.

**First-line fix:** flip the lane ↔ data mapping (`x[lane*K + i]` → `x[lane + i*32]`); AoS → SoA.

**Deeper fixes:** stage through shared memory; vectorize to 128-bit loads (`float4`, `uint4`); on sm90 move the stream to TMA, which coalesces by construction.

**Exceptions:** gather/scatter by random index; graph traversal. **sm90 — on a TMA-fed kernel these rules describe only the scalar side path**; check the volume first (Pattern R).

**Cross-ref:** Hopper principles 2, 13.

---

## Pattern D — Sparse writes (low store efficiency)

**Signals:** `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio < 16` (ideal 32); `L2 Global Store Access Pattern` rule; code with `if (lane < K) out[...] = ...`.

**Why:** only some lanes write, so store sectors flush half-empty.

**First-line fix:** pack the write — produce the K values collectively (shuffle / smem), then K contiguous lanes store K consecutive elements; batch several iterations into one vector store.

**Deeper fixes:** write the tile into smem and publish it with one bulk store (`cp.async.bulk.global.shared` / TMA store) — also the sm90 answer to a scattered epilogue that then needs a release fence (wiki `bulk-store-publish`).

**Exceptions:** histogram / scatter (Pattern G).

---

## Pattern E — Latency-bound (long-scoreboard-dominated)

**Signals:** `long_scoreboard` > 40 % of samples or ratio > 3; `dram__bytes_read % < 10`; hotspot lines are loads (or a counter poll).

**Why:** a load is issued and its consumer waits before the next independent load goes out — too few requests in flight, too little occupancy or ILP.

**First-line fix:** more independent requests in flight — unroll the load loop, deepen the pipeline, raise occupancy where the design allows (Pattern J).

**Deeper fixes:** software pipelining; move the reused data to smem; **sm90:** put the stream on TMA with an mbarrier ring, sized from the TMA unit's latency-coverage stages (`[tma.*]`), not by feel. If the hot `long_scoreboard` line is a *global counter poll* (a flag-barrier or release/acquire protocol), the cost is one observe hop `[atom.lat.dev.hop]` per wait — ordering, not throughput, is usually the bug (Pattern T).

**Exceptions:** pointer chasing.

**Cross-ref:** Hopper principles 7, 15.

---

## Pattern F — Compute-bound but not on tensor cores

**Signals:** `sm__inst_executed_pipe_fma > 50 %`; `sm__pipe_tensor_cycles_active = 0 %`; matmul-shaped workload; `math_pipe_throttle` stalls.

**Why:** scalar FMA instead of the tensor pipe.

**First-line fix:** `wgmma` (sm90, warpgroup-level, operands from smem via TMA), or TileLang / CUTLASS, which are already tuned. Below the crossover tile (`[mma.xover.n.wgmma]`) the warp-level `mma.sync` is the right primitive, not wgmma.

**Deeper fixes:** relayout to the MMA tile constraints; smem + TMA staging; FP8 where the accuracy budget allows.

**Exceptions:** non-matrix work; tiny M/N/K.

**Cross-ref:** Hopper principle 10; `kernel-design` templates for the spelling.

---

## Pattern G — Atomics contention

**Signals:** `long_scoreboard` samples concentrated on `ATOM` / `RED` SASS; L2 busy while SMs idle; `lg_throttle` on the atomic lines.

**Why:** many threads atomically updating few addresses → serialization at L2.

**First-line fix:** hierarchical reduction — shuffle within the warp, smem within the block, one atomic per block.

**Deeper fixes:** bucketed partials + merge; **sm90:** the layout lever dwarfs everything (`[atom.*]` tags): spread addresses, then widen to vector `red.add`; scope is free, one hot address is catastrophic.

**Exceptions:** communication kernels.

**Cross-ref:** Hopper principle 12.

---

## Pattern H — Shared-memory bank conflicts

**Signals:** `l1tex__data_bank_conflicts_pipe_lsu_mem_shared*.sum` a sizeable share of `l1tex__data_pipe_lsu_wavefronts_mem_shared*.sum`; `short_scoreboard` on smem load lines; regular strides aligned to banks.

**Why:** 32 banks; same-bank different-address accesses serialize.

**First-line fix:** padding (`tile[32][33]`).

**Deeper fixes:** XOR swizzle; on sm90 the TMA / wgmma smem layouts carry their own swizzle modes — match the descriptor (`tma-3d-box-row-major` in the wiki for the row-major trap).

**Exceptions:** broadcast reads; low smem volume.

**Cross-ref:** Hopper principle 4.

---

## Pattern I — Synchronization overhead

**Signals:** `barrier` > 20 % of samples; hotspot line is `BAR.SYNC`.

**Why:** `__syncthreads()` waits for the slowest warp; any per-warp imbalance is amplified.

**First-line fix:** warp-level primitives where only warp scope is needed; consolidate phases.

**Deeper fixes:** warp specialization with mbarriers instead of block barriers — and once you have that, the barrier stall becomes Pattern P.

**Cross-ref:** Hopper principle 16.

---

## Pattern J — Low achieved vs theoretical occupancy

**Signals:** `sm__maximum_warps_per_active_cycle_pct > 50` but `sm__warps_active...pct_of_peak_sustained_active ≪ 50`; rule *"difference between calculated theoretical (X %) and measured achieved occupancy (Y %)"*.

**Why:** stalls leave slots empty, imbalance empties some SMs, or the kernel is too short for warmup to amortize.

**Reading:** with Pattern B present, fixing imbalance closes the gap; otherwise the stall reasons (E, H, I) do.

**Exception:** theoretical ≈ achieved ≈ 10 % on a persistent kernel is Pattern O, not this.

---

## Pattern K — Register spill

**Signals:** `smsp__sass_inst_executed_op_local_{ld,st}.sum > 0`; `sass__inst_executed_register_spilling_mem_local > 0`; rule *"N bytes spilled to local memory"*; `launch__registers_per_thread` at 255.

**Why:** live state exceeds the register budget; local memory is DRAM-backed.

**First-line fix:** `__launch_bounds__`; **sm90:** `setmaxnreg` to give the math warpgroups the producer's registers.

**Deeper fixes:** fewer live accumulators, recompute instead of cache, split the kernel, per-thread arrays to smem.

**Exceptions:** none worth accepting here — well-tuned sm90 kernels run at 230+ registers with zero spill; any spill on a hot loop is a regression.

**Cross-ref:** Hopper principle 6.

---

## Pattern L — FP64 used unintentionally

**Signals:** `sm__pipe_fp64_cycles_active > 0` in a kernel that should be FP32/BF16.

**Why:** unsuffixed literals (`1.0`, `0.5`) promote to double.

**First-line fix:** `f` suffixes; `__expf` / `__logf` variants.

**Cross-ref:** Hopper principle 8.

---

## Pattern M — Pipeline bubbles (no compute/memory overlap)

**Signals:** timeline sawtooth (SM inst-executed ↕ DRAM alternating); `long_scoreboard` high *and* DRAM high in bursts.

**Why:** single-buffered load → compute → load.

**First-line fix:** double-buffer.

**Deeper fixes:** a 3–4-stage ring with TMA + mbarrier; depth from `[tma.*]` latency coverage; producers should release frames on retirement, not on the next fill (wiki `release-on-retirement`).

**Cross-ref:** Hopper principle 15.

---

## Pattern N — Warp divergence

**Signals:** `smsp__thread_inst_executed_per_inst_executed.ratio` well under 32; `smsp__sass_average_branch_targets_threads_uniform.pct` low; divergent lines are also stall hotspots.

**Why:** lanes take different paths; hardware serializes.

**First-line fix:** rearrange so warps are uniform; branchless selects when both sides are cheap.

**Exceptions:** tree-reduction tails; boundary handling; **warp-role branches in specialized kernels** (producer vs consumer) — structural, keep the ratio only slightly under 32, ignore unless the divergent lines are the hotspots.

**Cross-ref:** Hopper principle 5.

---

## Pattern O — "Increase occupancy" on a persistent warp-specialized kernel — overrule it (sm90)

**Signals:** rules `TheoreticalOccupancy` / `IssueSlotUtilization` / `LaunchConfiguration` / `HighPipeUtilization` with 50–90 % estimates; occupancy limited by registers *and* shared memory at 1 block/SM; `launch__grid_size == 132`, waves == 1; theoretical ≈ achieved ≈ 10 %.

**Why:** the kernel *is designed* as one resident CTA per SM holding a large smem ring with specialized producer/consumer warps (e.g. 224 threads, 230 registers, 230 KB smem). NCU sees 1.75 warps per scheduler out of 16 and extrapolates.

**Move:** none from these rules. Diagnose the kernel by Dimension 3 stall structure, Dimension 4 `gmma`/tensor activity, and the benchmark. Whether two smaller CTAs per SM would beat one big one is a `kernel-design` decision, priced against the occupancy-vs-residency findings of `hardware-unit-test`'s launch unit — not an NCU rule to obey.

**Exception that is real:** a *non-persistent* kernel with waves/SM < 1 — there Pattern A applies.

---

## Pattern P — Barrier-dominated task loop (sm90)

**Signals:** `barrier` the top aggregate stall; hotspots concentrated on one or two ring-wait PCs (`cutlass/arch/barrier.h` and the consumer's wait line) carrying a large share of all samples; `long_scoreboard` second, on the same or a counter-poll line.

**Why:** consumers outpace the producer (ring too shallow, one producer warp issuing too slowly, box too small) or task dependencies serialize.

**Move:** size the ring and producer against the TMA unit's delivery constants — per-warp issue interval `[tma.issue.warp]`, in-flight bytes needed per math warpgroup `[wgmma.bytes.wg.tma]`, DRAM-vs-L2 stage counts from the unit reference — then re-capture and expect the barrier ratio to move. If a deeper ring does not help, the frames are being released late (wiki `release-on-retirement`). If the barrier is a *counter* protocol wait, see Pattern T.

**Exception:** a barrier that is the *end-of-task* join on a reduction kind is Pattern B's task-schedule tail, not a ring problem.

---

## Pattern Q — `gmma` / `warpgroup_arrive` stalls high, tensor pipe underfed (sm90)

**Signals:** `gmma` prominent in the aggregate ratios and/or `warpgroup_arrive` at the fence lines in the samples; `sm__pipe_tensor_cycles_active` (`_active`) well under the mma unit's measured ceiling for the tile shape; `wait` samples inside `mma_sm90_gmma.hpp`.

**Why:** too few wgmma groups in flight (`wgmma.wait_group 0` after every issue), tile N below the efficient range, a second math warpgroup fighting the first for the pipe, or the K loop serialized by a dependency the compiler could not hoist (wiki `c7518-wgmma-serialization` for the build-flag trap).

**Move:** check the spec against the mma unit's constants (`[wgmma.issue.wg.ss]`, `[wgmma.stages.wg.knee]`, `[wgmma.ratio.sm.wg2]`, `[mma.xover.n.wgmma]`): N has a hard efficiency knee, `wait_group 0` is never right, one warpgroup already saturates the pipe at large N, and below the crossover tile `mma.sync` wins. These are design constants — fix the shape, don't chase the stall.

**Exception:** `wait` (not `gmma`) at the wgmma lines with the pipe near its ceiling is the healthy state of a saturated math warpgroup.

---

## Pattern R — TMA-fed kernel: the coalescing rules are looking at the wrong path (sm90)

**Signals:** `UncoalescedGlobalAccess` / `MemoryCacheAccessPattern` rules with 5–30 % estimates; LSU `sectors/request ≈ 1` and `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` tiny; `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` tens of MB; `smsp__sass_inst_executed_op_tma_ld.sum > 0`.

**Why:** the bulk traffic rides TMA and never touches the LSU counters; the rules are scoring the scalar side path — task descriptors, counters, a narrow epilogue.

**Move:** judge bulk movement by TMA bytes, `dram__bytes_*` and `lts__*`. Fix the side path only when its volume is real (an epilogue writing 16 of 32 bytes per sector across the whole output is real; a descriptor read is not). A scattered epilogue becomes a bulk store from smem (Pattern D).

---

## Pattern S — Genuinely bandwidth-bound

**Signals:** `dram__bytes_read %` approaching the *measured* cold-stream ceiling (`[ld.bw.dev.dram]`, or the TMA burst curve `[tma.bw.dev.burst]` for a cold weight stream — the constants are stated per source, say which); stalls spread thin; no single hot line.

**Move:** move fewer bytes — fuse, keep residents in smem/L2, revisit the fusion boundary (a pipeline / buffer question for the Target, not a kernel-local one; wiki `fusion-economics`). A kernel at the measured ceiling is *done*; the remaining lever is the plan around it. Below the ceiling with all knobs null on a cold stream: read `cold-burst-ceiling` before spending another job.

---

## Pattern T — Counter-protocol / atomics serialization (sm90 flag barriers)

**Signals:** `long_scoreboard` or `membar` concentrated on a global counter poll or its release fence; `sleeping` from backoff loops; L2 busy while SMs idle.

**Why:** a device-scope release/acquire hop costs `[atom.lat.dev.hop]` regardless of observers; a fence per store instead of per publish, or a wait placed before independent work, serializes the whole task graph.

**Move:** fence once per publish (bulk store + one release), place the wait as late as the dependency allows (wiki `prefetch-across-dependency`), spread hot addresses, widen to vector `red`. Reduce warp-first, block-second, device-last.

---

## Pattern U — Tiny kernels: don't profile what the launch dominates

**Signals:** kernel under ~20 µs; conclusions swing between captures; ramp visible in `gr__ctas_launched_realtime`.

**Why:** per-launch ramp `[launch.lat.dev.ramp]` and replay perturbation are comparable to the kernel itself; with clocks unpinned here, < 5 % deltas are noise.

**Move:** judge such kernels by CUPTI medians / mins from `benchmark-kernel` and by *counts* (instructions, sectors, stall mix, TMA bytes) from NCU — never by NCU durations. Fusion decisions at this scale are launch-cost arithmetic (PDL chains, wiki `pdl-placement`, `producer-fusion-pdl`), not microarchitecture.

---

## Pattern V — Cluster / DSMEM kernels (sm90)

**Signals:** `launch__cluster_size > 0`; `launch__occupancy_cluster_pct` low or `launch__cluster_max_active` small; `smsp__sass_inst_executed_op_dshared_*.sum` high; `barrier` at `cluster_sync` lines.

**Why:** clusters must co-schedule on one GPC — the scheduler caps how many clusters fit (`[cluster.count.max]`), and every `cluster_sync` costs `[cluster.lat.sync]`.

**Move:** place cluster barriers at the DSMEM reduction only, never per K step (wiki `cluster-barrier-placement`); prefer size-2 clusters for a shared operand; if the cluster only exists to share a weight tile, price it against L2 (which already serves 132 SMs).

---

## Ranking template for the final plan

Rank by `(expected speedup) × (ease)`. `Est. Speedup` is the magnitude estimator, your judgement the effort estimator, and a rule you have overruled (O) contributes nothing.

```
Priority 1: <pattern> — <concrete fix>
  Evidence: <metric = value>, <hotspot file:line samples>, <rule + Est. Speedup>
  Ceiling:  <hardware-unit-test tag that bounds the gain>
  Effort:   <low / medium / high>
  Confirm:  <the targeted-metrics recapture that would show it landed>

Priority 2: ...
```

At most 3–5 priorities. More dilutes the signal; priorities past five rarely contribute 5 % each.
