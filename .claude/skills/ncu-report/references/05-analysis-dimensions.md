# Six Analysis Dimensions

Every kernel report is ambiguous until you look at it through specific lenses. These six consistently matter. Choose the dimensions relevant to the question and broaden only when the evidence leaves a meaningful alternative unresolved.

For each dimension: **what you're answering**, **which metrics to read** (sm90 / Nsight Compute 2025.4.1 names, all verified on this host's reports — see `08-sm90-metric-names.md`), **how to read them**, and **which helper to run**. **sm90** marks Hopper-specific notes; this repo's kernels are warp-specialized persistent task loops fed by TMA and wgmma, so those notes are not optional here.

---

## Dimension 1 — SM occupancy & launch geometry

**What:** is the grid large enough to fill the GPU? Is occupancy limited by registers, shared memory, block size — or by design?

**Metrics:**
```
launch__grid_size / launch__block_size / launch__grid_dim_{x,y,z}
launch__waves_per_multiprocessor
launch__registers_per_thread            (launch__registers_per_thread_allocated: after allocation granularity)
launch__shared_mem_per_block            (+ _static / _dynamic / _driver)
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
launch__occupancy_limit_barriers        (sm90: named-barrier count can be the limiter)
device__attribute_multiprocessor_count  (132 on this H100)
sm__maximum_warps_per_active_cycle_pct  (theoretical occupancy %)
smsp__maximum_warps_avg_per_active_cycle (theoretical warps per scheduler, max 16)
sm__warps_active.avg.pct_of_peak_sustained_active   (achieved occupancy %)
launch__cluster_size / launch__cluster_dim_{x,y,z} / launch__occupancy_cluster_pct   (sm90: 0 = no cluster)
```

**Reading:**

- **Waves / SM < 1** on a *non-persistent* kernel: SMs sit idle the whole time. `Est. Speedup` from NCU often hits 50–90 % here. Parallelize another axis, split-K, or fuse.
- **Waves / SM in [1, 2)**: a partial last wave. Tail cost ≈ `(1 − last_wave_fill) × block_time / kernel_time`.
- **Waves / SM > 4**: the grid is plenty; scheduling averages out.
- **Theoretical 100 % but achieved ≪ 100 %**: stalls or imbalance, not launch config → Dimension 3 / 2.
- **Theoretical low, `occupancy_limit_registers` tightest**: cut live registers or `__launch_bounds__` — *if* more residency is actually wanted.
- **`occupancy_limit_shared_mem` tightest**: smaller tiles or a shallower ring — same caveat.
- **sm90 — persistent task loop**: grid == 132, 1 CTA/SM, registers and smem both at the limit, theoretical ≈ achieved ≈ 10 %. That is the design (one resident CTA holding a large smem ring with producer/consumer warps), not a defect. NCU's `TheoreticalOccupancy`, `IssueSlotUtilization` and `LaunchConfiguration` rules will estimate 50–60 % gains from it; **overrule them** (playbook O) and judge the kernel by Dimensions 3–4. Whether 2 smaller CTAs/SM would beat 1 big one is a `kernel-design` decision informed by `hardware-unit-test`'s launch unit, not an NCU rule to obey.

**Derived: wave math**

```python
blocks_per_sm = min(occ_limit_blocks, occ_limit_registers, occ_limit_shared_mem, occ_limit_warps, occ_limit_barriers)
wave_size     = blocks_per_sm * num_sms
num_waves     = ceil(total_blocks / wave_size)
last_wave_pct = (total_blocks - (num_waves - 1) * wave_size) / wave_size * 100
```

**Helper:** `report_query.py summary` prints the launch block; `analyze_reports.py` archives it under "Launch geometry".

---

## Dimension 2 — Thread-block balance (tail effect)

**What:** do CTAs finish together, or do a few outliers drag the kernel out?

**Metrics:**

There is no single imbalance metric — combine:
```
sm__cycles_active.{avg,max,min}                 # per-SM spread; the WorkloadImbalance rules quote it
smsp__cycles_active.{avg,max,min}               # per-scheduler spread
# rule engine: WorkloadImbalance (SMs / SMSPs / L1 slices / L2 slices)
# timeline: warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>, TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed
```

**Reading:**

- The rule text says it directly: *"One or more SMs have a much lower number of active cycles than the average. Maximum instance value is X % above the average, while the minimum instance value is Y % below."* X ≈ 7 %, Y ≈ 91 % (seen here) means one SM did almost nothing — on a 132-CTA persistent grid that is one CTA with no tasks, i.e. the task schedule, not the data.
- Render the timelines (`plot_timeline.py`). Shapes: **flat high → clean drop** ideal; **flat high → gradual tail** tail effect; **flat low** grid too small (Dim 1) or stall-bound (Dim 3); **sawtooth** no compute/memory overlap.
- On this fixed-workload host kernels are balanced by construction; the exception is a task-loop kernel whose queue drains unevenly (a reduction task kind waiting on the slowest sibling, a serial epilogue owner). Then the fix is in the task schedule (`kernel-design` wiki: `reduction-own-task-kind`, `serial-epilogue-owner`), not in the kernel body.

**Where imbalance typically comes from:** variable per-CTA work (a length-driven inner loop), early-exit branches, a work-stealing scheme that hands the heavy items to few CTAs.

**Fix direction:** chunk variable-length work; oversubscribe with work-stealing; on a task loop, reorder or split the task graph.

**Helper:** `plot_timeline.py` — a gradual slope on the right is the tail. Inspect the per-CTA work distribution from the input (task list) as well; ratios > 5× max/min predict a tail.

---

## Dimension 3 — Stall reason breakdown + per-line hotspots

**What:** when warps aren't issuing, what are they waiting for, and on which source line?

**Aggregate stall ratios (whole kernel):**
```
smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio
   reasons (19, sm90): barrier branch_resolving dispatch_stall drain gmma imc_miss lg_throttle
   long_scoreboard math_pipe_throttle membar mio_throttle misc no_instruction not_selected
   selected short_scoreboard sleeping tex_throttle wait
```

**Per-PC samples (also in `--set full` here):**
```
smsp__pcsamp_sample_count
smsp__pcsamp_warps_issue_stalled_<reason>           (+ <reason>_not_issued)
   reasons (19, sm90): as above, but warpgroup_arrive instead of gmma, and no_instructions (plural)
```

**Stall reasons you need to know:**

| Reason | Waiting on | Typical cause | Fix direction |
|---|---|---|---|
| `long_scoreboard` | L1TEX result (global/local/texture) | latency-bound loads, scalar loads where a bulk path belongs, ring wait implemented as a global poll | more requests in flight, move the traffic to TMA, deepen the ring |
| `short_scoreboard` | shared-memory / MIO result | bank conflicts, dependent smem chains | pad/swizzle, add ILP |
| `gmma` (**sm90**, aggregate only) | wgmma group completion | `wait_group` too strict, tile N below the efficient range, pipe genuinely saturated | check the shape against the mma unit's constants (playbook Q) |
| `warpgroup_arrive` (**sm90**, per-PC only) | `WARPGROUP.ARRIVE` / `WARPGROUP.WAIT` | same discipline as `gmma`, seen at the fence lines | as above |
| `barrier` | CTA / named barrier | producer/consumer imbalance, divergence before a bar, ring wait | size the ring and producer (playbook P) |
| `wait` | fixed-latency dependency | dependent math chains | independent work; normal on tight wgmma loops |
| `imc_miss` | immediate-constant cache | large kernel-parameter block (many `CUtensorMap`s), cold cbank | prologue noise unless sustained |
| `math_pipe_throttle` | FMA/ALU pipe full | genuinely compute-bound off tensor cores | move the math to wgmma, or accept |
| `mio_throttle` / `lg_throttle` / `tex_throttle` | MIO / LSU / TEX queue full | too many load/store instructions | vectorize, stage through smem, bulk copies |
| `membar` | memory fence | fence placement in a release/acquire protocol | fence once per publish, not per store |
| `sleeping` | `nanosleep` / yield | polling loops with backoff | shorten backoff, wait on mbarrier instead |
| `branch_resolving` | branch target | tight loops | usually minor |
| `no_instruction` | fetch starve / drain | prologue / epilogue, icache miss on a huge kernel | minor unless large |
| `dispatch_stall` / `misc` / `drain` | dispatcher, misc, end of kernel | rare | ignore |
| `not_selected` | eligible, lost arbitration | **good** — spare parallelism | ignore |
| `selected` | issuing | productive; the baseline (= 1.0) | ignore |

**Reading the ratio:** `..._stalled_long_scoreboard_per_issue_active.ratio = 12.2` means that per cycle with an issue, 12.2 warps on that scheduler sat in `long_scoreboard`. Rank the ratios; take the top one or two; then `hotspots` to find *where*. On a 4–7-warp persistent CTA the ratios are small in absolute terms (few warps exist) — read them relative to each other and to `selected`.

**Reading the samples:** normalize by `smsp__pcsamp_sample_count`. Rules of thumb: `long_scoreboard` > 40 % latency-bound (check Dim 6 next); `short_scoreboard` > 30 % smem chains / conflicts; `barrier` > 20 % synchronization or imbalance; `selected` < 10 % the kernel is stall-bound overall.

**sm90 signature to recognize:** `barrier` + `long_scoreboard` dominant and concentrated on one or two ring-wait PCs (`cutlass/arch/barrier.h` and the consumer's wait line) is the producer/consumer wait of a warp-specialized task loop — a pipeline-sizing question (playbook P), not a memory-system defect. `wait` at `mma_sm90_gmma.hpp` lines is the wgmma issue loop and is normal.

**Helper:** `report_query.py hotspots --top 15` (terminal), `extract_stall_hotspots.py` (per-reason top lines to `analysis/`).

---

## Dimension 4 — Tensor Core utilization

**What:** is the kernel using tensor cores at all? If yes, how close to the ceiling for its geometry?

**Metrics (sm90):**
```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_{elapsed,active}      # overall tensor pipe
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active         # FP16/BF16/TF32 pipe — wgmma is counted here
sm__pipe_tensor_op_{imma,dmma}_cycles_active.avg.pct_of_peak_sustained_active  # INT8 / FP64 pipes
sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active         # wgmma *instruction* issue share
sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active         # mma.sync issue share
smsp__sass_inst_executed_op_shared_gmma.sum                                    # wgmma instructions reading smem operands
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active                    # scalar FMA — the fallback path
```

**Reading:**

- **`sm__pipe_tensor_cycles_active` = 0 %** on a matmul-shaped kernel: no tensor cores at all — the finding by itself.
- **Low tensor + high `gmma` / `warpgroup_arrive`**: the issue side is fine, the pipe is saturated or drained wrongly — group discipline, tile N (playbook Q).
- **Low tensor + high `long_scoreboard` / `barrier`**: the pipe is starved — feed it (playbook P / E).
- **What "good" means is geometry-dependent** (tile N, warpgroup count, K depth). The ceilings and the N-vs-throughput curve are `hardware-unit-test`'s mma unit constants (`[wgmma.*]`, `[mma.xover.n.wgmma]`), not NCU's speed-of-light. Consult them before calling a number bad; below the crossover tile the warp-level `mma.sync` beats wgmma outright.
- `_elapsed` counts idle SMs and the ramp against you, `_active` does not — quote which.
- **sm90 trap:** `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off` reads **0 on a wgmma kernel** (10 % busy in `sm__pipe_tensor_op_hmma_cycles_active`). The ops-path counters do not see wgmma; there is no `ops_path_tensor_op_gmma`. Use the pipe-cycles and `inst_executed_pipe_tensor_op_gmma` metrics.

**Fix direction:** 0 % on a matmul shape → redesign around wgmma (or TileLang / CUTLASS), a major refactor worth 2–10× on compute-bound paths. Present but underfed → Dimension 6 and the pipeline depth.

---

## Dimension 5 — Utilization over time

**What:** how does utilization vary over the kernel's lifetime — flat, tail, sawtooth, ramp?

**Metrics (2025.4.1 full set — these replace the `pmsampling:` names of the B200 docs):**
```
warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>                      # 20 warp-state series (PmSampling_WarpStates)
TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
TPC.TriageCompute.sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed
TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed
FBSP.TriageCompute.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed
LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed
LTS.TriageCompute.lts__average_t_sector_hit_rate_realtime.pct
SM_A.TriageCompute.l1tex__throughput.avg.pct_of_peak_sustained_elapsed
FE_B.TriageCompute.gr__ctas_launched_realtime.avg.per_cycle_elapsed          # launch ramp
```
Each has ~150 timestamped instances at the sampling interval (1.5 µs default here), spanning a window longer than the kernel; `plot_timeline.py` trims to the workload window.

**Reading (timeline shapes):**

- **Flat high, clean drop**: ideal.
- **Flat high, long tail**: tail effect (Dim 2).
- **Flat low**: grid too small (Dim 1) or stall-bound (Dim 3).
- **Sawtooth (compute ↕ memory alternating)**: no overlap — single-buffered, or a ring too shallow to cover latency.
- **Slow ramp, flat middle, clean drop**: prologue (tensor-map prefetch, weight ring fill), then steady state. Usually fine; on a 12 µs kernel the ramp is a large fraction — `[launch.lat.dev.ramp]` says what is structural.

**Limits:** at 1.5 µs a 12–20 µs kernel yields under ten in-kernel samples (the `PMSamplingData` rule warns about it). Shorten with `--pm-sampling-interval` (≥ 1000 ns) when the shape matters; otherwise answer shape-over-time questions with nsys via `gpu-profiler-analysis` or in-kernel counters where a persistent kernel exposes them.

**Helper:** `plot_timeline.py` — look at SM inst-executed, tensor, DRAM and the top stall series side by side.

---

## Dimension 6 — Memory access pattern & cache efficiency

**What:** what actually moved, through which path, and how efficiently? Is DRAM busy, or just badly used?

**Metrics:**
```
# DRAM — what truly moved
dram__bytes_read.sum[.pct_of_peak_sustained_elapsed | .per_second]
dram__bytes_write.sum[.pct_of_peak_sustained_elapsed]
dram__sectors_{read,write}.sum

# L2 / L1
lts__t_sector_hit_rate.pct / lts__t_sector_op_{read,write}_hit_rate.pct
lts__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_{ld,st}_hit_rate.pct

# sm90 — the bulk path (TMA), invisible to the LSU counters below
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum          # bytes TMA loaded from L2/DRAM
l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_{st,red}.sum   # bytes TMA stored / reduced
smsp__sass_inst_executed_op_tma_{ld,st,red}.sum                  # TMA instructions
sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active      # TMA unit busy
smsp__sass_inst_executed_op_dshared_*.sum                        # DSMEM traffic in cluster kernels

# LSU path — coalescing (only the scalar side path on a TMA-fed kernel)
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum   # sectors/request, ideal 4
smsp__sass_average_data_bytes_per_sector_mem_global_op_{ld,st}.ratio   # useful bytes per 32 B sector
derived__memory_l2_theoretical_sectors_global_excessive               # what the UncoalescedGlobalAccess rule counts
smsp__sass_inst_executed_op_global_{ld,st}.sum

# shared memory
smsp__sass_inst_executed_op_shared_{ld,st}.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared{,_op_ld,_op_st}.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared{,_op_ld,_op_st,_op_ldgsts}.sum   # conflicts / wavefronts = conflict share

# register spill
smsp__sass_inst_executed_op_local_{ld,st}.sum
sass__inst_executed_register_spilling_mem_local{,_op_read,_op_write}

# divergence
smsp__thread_inst_executed_per_inst_executed.ratio      # 32 = none
smsp__sass_average_branch_targets_threads_uniform.pct   # branch efficiency
```

**Reading:**

- **DRAM % near the machine's measured ceiling** → bandwidth-bound; the ceiling is `hardware-unit-test`'s streaming constant for the right *source* (cold DRAM vs from-L2 differ hugely — say which), not the datasheet, and a kernel at it is done.
- **DRAM % ≪ 10 % and SM % low** → latency-bound (Dim 3 has the reason) — not "memory-bound".
- **sm90 — TMA-fed kernels barely touch the LSU metrics.** `sectors/request = 1.0` with `l1tex__t_sectors...ld.sum` tiny while `..._tma_ld.sum` is tens of MB means the bulk traffic rides TMA; the coalescing rules (`UncoalescedGlobalAccess`, `MemoryCacheAccessPattern`) then describe only the scalar side path (task descriptors, counters, epilogue stores). Judge bulk movement by TMA bytes, `dram__bytes_*` and `lts__*`; chase the side path only when it carries real volume.
- **`l1tex__t_sector_hit_rate` > 90 %**: L1 absorbs the reuse. **`lts__t_sector_hit_rate` < 50 %**: L2 is being blown through; on a weight stream that is expected once per layer.
- **sectors/request 4–5**: fine; **> 8**: serious non-coalescing on the LSU path.
- **store `bytes_per_sector` < 16**: half-empty sectors — partial-warp stores (`if (lane < K)`), or an epilogue writing a narrow tile; sm90 fix is a bulk store (`cp.async.bulk` / TMA store) from smem.
- **bank conflicts / wavefronts > ~10 %** with `short_scoreboard` up: pad or swizzle; decidable offline from the access pattern.
- **any spill on a hot loop** (`local_ld/st > 0`, `register_spilling_* > 0`): a regression, not a tolerable cost — well-tuned sm90 kernels sit at 230+ registers with zero spill.
- **threads/inst well under 32** only matters on the stall hotspots; warp-role and tail branches in specialized kernels keep it slightly under 32 structurally.

**Fix directions:** strided access → remap lanes / SoA; sparse writes → pack and bulk-store; spill → `__launch_bounds__`, fewer live accumulators, split; TMA-fed but starved → ring depth and producer issue rate against `[tma.*]` constants.

---

## Cross-dimension synthesis

After all six, write the one-line diagnosis. Name the top 3–4 signals, each tied to a dimension and a value:

> "The kernel runs at X % SM throughput and Y % DRAM (Dim 1, 6); occupancy is pinned at 1 CTA/SM by design (Dim 1, overruled). Stalls are dominated by `barrier` (Z % of samples) at the ring wait `file:line` and `long_scoreboard` at the consumer's counter poll (Dim 3); TMA moved N MB while the LSU path moved almost nothing (Dim 6). Tensor pipe at W % `_active`, `gmma` low — the pipe is starved, not saturated (Dim 4). The timeline shows a ramp of R µs then flat (Dim 5)."

Fill in the values from your own report. That sentence is the deliverable; everything else is evidence.
