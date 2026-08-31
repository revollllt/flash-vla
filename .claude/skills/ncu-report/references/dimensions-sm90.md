# The six lenses, with sm90-verified metric names

Walk all six on every report; on any one kernel only one or two dominate, but
you don't know which until you've looked. Names below exist on the reports
the host toolchain's ncu produces (`metrics-sm90.md` has the validity
conditions and the full lists).

## 1. Launch geometry and occupancy

```
launch__grid_size / launch__block_size / launch__waves_per_multiprocessor
launch__registers_per_thread / launch__shared_mem_per_block
launch__occupancy_limit_{blocks,registers,shared_mem,warps}   # blocks/SM by resource; min wins
device__attribute_multiprocessor_count                        # the host H100 reports 132
sm__maximum_warps_per_active_cycle_pct                        # theoretical occupancy %
sm__warps_active.avg.pct_of_peak_sustained_active             # achieved occupancy %
```

- `waves/SM < 1` on a **non-persistent** kernel: SMs sit idle; parallelize
  another axis or split-K. On a **persistent/task-loop** kernel, grid == SM
  count and 1 CTA/SM is the design — do not "fix" it (playbook 1).
- Theoretical high but achieved far lower: the limit is stalls or imbalance,
  not the launch config — go to lens 3.
- Theoretical itself low: the tightest `occupancy_limit_*` names the resource
  to shrink, *if* more residency is actually wanted.
- Wave math when tails matter: `blocks_per_sm = min(limits)`;
  `wave = blocks_per_sm × SMs`; a partial last wave costs roughly its
  emptiness times one block's runtime.

## 2. Balance across SMs

No single metric; combine the details-page distribution note ("minimum is X%
below average" on `sm__cycles_active`) with the input's per-CTA work spread.
On a fixed-workload host, kernels are usually balanced by construction; the
exception is task-loop kernels whose task queue drains unevenly — then the
imbalance is in the task schedule, not the data.

## 3. Stalls — aggregate, then attribute

Aggregate: `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`
= how many warps sat in that state per issued instruction; read the top few
relative to each other. Per-PC: `smsp__pcsamp_warps_issue_stalled_<reason>`
(paired `_not_issued` variants; note `no_instruction` aggregate vs
`no_instructions` pcsamp), attributed by `report_query.py hotspots`.

| Reason | Waiting on | First association |
|---|---|---|
| `long_scoreboard` | global/L2 data return | latency-bound loads; check lens 6 and ILP/ring depth |
| `short_scoreboard` | shared-mem or short dep | bank conflicts, dep chains |
| `gmma` | **wgmma group completion (sm90-specific)** | tensor pipe saturated or `wait_group` too eager — see the mma unit's pipeline findings |
| `barrier` | named/CTA barrier | producer/consumer imbalance, divergence before bar |
| `wait` | fixed-latency pipe result | dep chains on math; add independent work |
| `imc_miss` | immediate/constant cache | large param/constant traffic, cold cbank; usually prologue noise unless sustained |
| `math_pipe_throttle` | FMA/ALU pipe full | genuinely compute-bound off tensor cores |
| `mio_throttle` / `lg_throttle` / `tex_throttle` | LSU/L-G/TEX queue full | too many load/store instructions; vectorize |
| `membar` | memory fence | fence placement |
| `branch_resolving` | branch target | tight loops; usually minor |
| `no_instruction` | fetch starve / drain | prologue-epilogue; minor unless huge |
| `not_selected` | eligible, lost arbitration | *good* — spare parallelism |
| `selected` | issuing | productive; the baseline of the ratio |

Reading: rank the ratios, take the top one or two, then `hotspots` to find
*where*. On a warp-specialized task loop, barrier + long_scoreboard
dominating and concentrated on one or two ring-wait PCs is the
producer/consumer wait signature, not a memory-system defect.

## 4. Tensor pipe

```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_{elapsed,active}
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
```

Zero tensor activity on a matmul-shaped kernel is the finding by itself. Low
tensor + high `gmma` stalls = issue side fine, pipe saturated or drained
wrong; low tensor + high `long_scoreboard`/`barrier` = the pipe is starved —
feed it. What "good" means is geometry-dependent (tile N, warpgroup count);
the ceilings and the N-vs-throughput curve are the mma unit's constants, not
NCU's speed-of-light — consult `hardware-unit-test` before calling a number
bad. `_elapsed` counts idle SMs against you, `_active` does not; quote which.

## 5. Utilization over time

Needs PM sampling, which no capture here has requested yet — the sections
exist on 2025.4.1 but populated metrics are unverified, a **GAP**
(`metrics-sm90.md` has the recipe to close it). Until then, shape-over-time
questions (tail, sawtooth, ramp) are answered indirectly: nsys timelines via
`gpu-profiler-analysis`, or in-kernel counters where a persistent kernel
exposes them.

## 6. Memory pattern

```
dram__bytes_read.sum[.pct_of_peak_sustained_elapsed|.per_second]  # what truly moved
lts__t_sector_hit_rate.pct / l1tex__t_sector_hit_rate.pct
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ÷ l1tex__t_requests_...  # sectors/request
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio    # store fill, max 32
smsp__sass_inst_executed_op_local_{ld,st}.sum                      # >0 = register spill
smsp__thread_inst_executed_per_inst_executed.ratio                 # 32 = no divergence
```

- **TMA-fed kernels barely touch the LSU metrics** — a kernel whose real
  traffic rides TMA shows only its scalar side path there. Judge bulk
  movement by `dram__bytes_*` and `lts__*`; judge coalescing only for the
  side path (`sectors/request` ideal 4 for contiguous 128 B — and on a tiny
  side path a poor ratio is not worth chasing).
- DRAM % near the machine's measured ceiling → bandwidth-bound (the ceiling
  is the streaming constant in `hardware-unit-test`, not the datasheet).
  DRAM % low *and* SM % low → latency-bound; lens 3 has the reason.
- Spill counters over zero on a hot loop: cut live state or split. Well-tuned
  sm90 kernels run near the register wall with zero spill, so treat any spill
  as a regression, not a tolerable cost.
