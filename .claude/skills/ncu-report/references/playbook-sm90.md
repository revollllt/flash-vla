# Signal → cause → move (sm90, fixed-workload kernel styles)

Each entry: the signals that flag it, the usual cause, the first move,
and the exception that makes the signal a false alarm. Machine numbers are
cited by their `hardware-unit-test` tag — read the value and validity range
via `python3 .claude/skills/hardware-unit-test/scripts/constants.py --tag <t>`;
restating them here would just let two copies drift.

## 1. "Increase occupancy" on a persistent warp-specialized kernel — overrule it

- **Signals**: rules `TheoreticalOccupancy` / `IssueSlotUtilization` /
  `LaunchConfiguration` with huge estimates; occupancy limited by registers
  and shared memory at 1 block/SM; grid == SM count.
- **Cause**: the kernel *is designed* as one resident CTA per SM holding a
  large smem ring with specialized producer/consumer warps. On such a kernel
  NCU will happily estimate double-digit gains from what is a design
  property.
- **Move**: none from these rules. Diagnose such kernels by lens-3 stall
  structure, `gmma`/tensor activity, and the benchmark — the design question
  ("would 2 smaller CTAs/SM beat 1 big one") is a kernel-design decision,
  informed by the occupancy-vs-residency findings in `hardware-unit-test`'s
  launch unit, not an NCU rule to obey.
- **Exception that is real**: a *non-persistent* kernel with waves/SM < 1 —
  there the small-grid rule is right; split work or fuse.

## 2. Barrier-dominated task loop

- **Signals**: `barrier` the top aggregate stall; hotspots concentrated on
  one or two ring-wait PCs that together carry a large share of all samples.
- **Cause**: consumers outpace the producer (ring too shallow, producer
  under-provisioned) or task dependencies serialize.
- **Move**: size the ring and producer against the TMA unit's delivery
  constants — per-warp issue interval `[tma.issue.warp]`, in-flight bytes
  needed per math warpgroup `[wgmma.bytes.wg.tma]`, DRAM-vs-L2 stage counts
  from the unit reference — then re-capture and expect the barrier ratio to
  move. If the barrier is a *counter* protocol wait, the hop cost is
  `[atom.lat.dev.hop]` and observers are free; ordering, not throughput, is
  usually the bug.

## 3. `gmma` stalls high / tensor pipe underfed

- **Signals**: `gmma` prominent in aggregate stalls; tensor `_active` % well
  under the mma unit's measured ceiling for the tile shape.
- **Cause**: too few wgmma groups in flight (`wait_group` too strict), tile N
  below the efficient range, or a second math warpgroup fighting the first.
- **Move**: check the spec's N and in-flight group count against the mma
  unit (`[mma.*]` tags): N has a hard efficiency knee, `wait_group 0` is
  never right, one warpgroup already saturates the pipe, and below the
  crossover tile the warp-level `mma.sync` beats wgmma outright. These are
  design constants — fix the shape, don't chase the stall.

## 4. Latency-bound loads (`long_scoreboard`) with idle DRAM

- **Signals**: `long_scoreboard` top-two; `dram__bytes_read` % far below the
  measured streaming ceiling; hotspot lines are loads.
- **Cause**: not enough requests in flight — serial dependent loads, ring
  too shallow, or scalar loads where a bulk path belongs.
- **Move**: more independent loads in flight (unroll, deepen the pipeline,
  or move the traffic to TMA). Size "enough in flight" from the TMA unit's
  latency-coverage stages, not by feel. Remember lesson: on TMA-fed kernels
  the LSU metrics only see the side path — confirm which path stalls before
  redesigning the wrong one.

## 5. Genuinely bandwidth-bound

- **Signals**: `dram__bytes_*` % approaching the measured cold-stream
  ceiling (the streaming constants are stated per SOURCE — cold DRAM vs
  from-L2 differ enormously; say which you mean); stalls spread thin.
- **Move**: move fewer bytes (fuse, keep residents in smem/L2, revisit the
  fusion boundary — a pipeline/buffer question for the target, not a
  kernel-local one). A kernel at the measured ceiling is *done*; the
  remaining lever is the plan around it.

## 6. Scalar-path memory sins

- **Signals**: LSU `sectors/request` well above 4; store fill
  (`...bytes_per_sector...st.ratio`) far under 32; `mio/lg_throttle` stalls.
- **Cause**: strided per-lane indexing, AoS layouts, partial-warp stores.
- **Move**: re-map lanes to contiguous addresses, vectorize to 128-bit
  accesses, stage through smem when the natural layout is hostile. Only
  worth it when the scalar path carries real volume (see lesson 2).

## 7. Shared-memory conflicts / `short_scoreboard`

- **Signals**: `short_scoreboard` elevated with heavy smem traffic.
- **Move**: pad or swizzle the layout. Bank behavior is decidable offline
  from the access pattern before any capture; NCU is the after-the-fact
  check.

## 8. Register spill

- **Signals**: `smsp__sass_inst_executed_op_local_{ld,st}.sum > 0`, regs at
  the 255 wall, spill bytes named in Instruction Statistics.
- **Move**: `__launch_bounds__`, fewer live accumulators, or split the
  kernel. Well-tuned sm90 kernels hold high register counts with zero
  spill — treat any spill as a regression, not a tolerable cost.

## 9. Atomics serialization

- **Signals**: stalls concentrated on `ATOM`/`RED` SASS PCs; L2 busy while
  SM idles.
- **Move**: the layout lever dwarfs everything else (`[atom.*]` tags:
  spread addresses, then widen to vector `red.add`; scope is free, one hot
  address is catastrophic). Reduce warp-first, block-second, device-last.

## 10. Tiny kernels: don't profile what the launch dominates

- **Signals**: kernel under ~20 µs; conclusions swing between captures.
- **Cause**: per-launch ramp `[launch.lat.dev.ramp]` and replay perturbation
  are comparable to the kernel itself; and with clocks unpinned here, <5%
  deltas are noise.
- **Move**: judge such kernels by CUPTI medians/mins from `benchmark-kernel`
  and by *counts* (instructions, sectors, stall mix) from NCU — not by NCU
  durations. Fusion decisions at this scale are launch-cost arithmetic, not
  microarchitecture.

## 11. Divergence

- **Signals**: `smsp__thread_inst_executed_per_inst_executed.ratio` well
  under 32, concentrated on branchy lines. Warp-role and tail branches in
  specialized kernels keep the ratio only slightly under 32; that is
  structural.
- **Move**: only act when the divergent lines are also the stall hotspots;
  warp-role branches in specialized kernels are structural and fine.
