# Unit: launch — grid ramp, streaming bandwidth, clusters, occupancy

**Constants** `LAUNCH-RAMP`, `BW-CEIL`, `BW-CTA`, `CLUSTER-SYNC`,
`CLUSTER-PLACE`, `OCC-NOT-WARPS` · **Probe** none in this skill

This unit was measured before the skill existed, as Phase 0 of a decoder
project, through `flash_vla.runtime.cuda.graph_time_cold`. The constants are
recorded here so every floor traces to one place; the full worked run — the
0a/0b split, and why the library-comparison measurement is *not* a bound — is in
`sm90/example-phase0-run.md`, which moved here with the rest of the
measurement layer. **It carries no Slurm job ids**, only its date, harness and
toolchain: that is the weakest provenance in this skill and the reason
`unit-launch` has no probe of its own yet.

**Toolchain caution.** These were taken 2026-08-20 on torch 2.11.0+cu130, five
days and one torch minor before the tma unit. The only cross-check between the
two is that both independently land near 2.77 TB/s for streamed cold reads.

## The four numbers that change a design

**Every kernel starts ~1.24 µs in debt** (`LAUNCH-RAMP`), at 128 CTAs and above.
This is *grid ramp*, not host launch overhead — a CUDA graph does not remove it.
Empty kernel by grid size: 0.95 / 1.20 / 1.24 / 1.24 µs at 32 / 64 / 128 / 256
CTAs. **Launch count is therefore a first-class term in every fusion decision:**
six decoder launches floor at `6 × 1.85 = 11.1 µs` per layer before a byte moves.

**`bytes / peak_BW` is not a floor** (`BW-CEIL`). A cold read costs
`t_us = 1.85 + MB/2.77`; the *marginal* rate is 2.77 TB/s and average bandwidth
is far below it on small transfers:

| cold read, 128 CTAs | 4.19 MB | 8.39 MB | 16.78 MB | 33.55 MB |
|---|---:|---:|---:|---:|
| µs | 3.34 | 5.18 | 8.21 | 14.26 |
| GB/s | 1256 | 1619 | 2043 | 2353 |
| % of 3.35 TB/s | 37.5 | 48.3 | 61.0 | 70.2 |

cuBLAS lands on the fit (7.95 µs measured against 7.91 modelled at 16.78 MB),
which is what calibrates it. Two of this project's three kernel targets had been
set *below their own floors* by dividing by the datasheet peak.

**CTA count still matters past machine coverage** (`BW-CTA`). At 4.19 MB:
8.91 / 5.43 / 3.85 / **3.33** / 3.60 µs at 16 / 32 / 64 / 128 / 264 CTAs. 32
CTAs cost 1.63× what 128 do for identical bytes — that is what justifies split-K
over a coarser tile — and 264 is *worse* than 128, so more is not monotonically
better.

**Declaring a cluster is free; synchronising one is not** (`CLUSTER-SYNC`).
Cluster 2 / 4 / 8 all measure the same as no cluster; each `cluster_sync()` adds
0.65 µs at cluster 8 (0.51 at cluster 4) at *zero skew*. Placement dominates
that floor: hoisting one barrier from a kernel's reduction up beside its TMA
issue, to hide it under load latency, cost **5.49 → 6.95 µs**. A barrier at the
*end* only absorbs skew already being paid; at the *start* it adds fill skew to
the critical path. **Design the number of barriers, not the cluster size.**

## Two traps, both of which cost this project time

**Cluster placement follows the occupancy query, not the SM count**
(`CLUSTER-PLACE`). At 1 CTA/SM this machine places 66 clusters of 2, 30 of 4,
15 of 8 — so a 128-CTA grid at cluster 8 is not co-resident, and either
deadlocks or runs a second wave. Confirmed behaviourally: 120 blocks reach a
spin barrier, 128 time out. The lever is **smem**, not the grid, and
`__launch_bounds__`'s second argument is a *minimum* hint that cannot cap
occupancy. Cooperative launch and clusters are not available together on sm90.

**Occupancy capacity is not resident warps** (`OCC-NOT-WARPS`). 128 CTAs on 132
SMs land one per SM whatever the query says. Freeing smem to reach 3 CTAs/SM
buys nothing until the grid reaches `3 × SM_count` — measured: dropping one
kernel phase from 204288 to 72320 B took the query 1 → 3 and bought 0.74 µs,
because the warp count never moved. Read `Block Limit Registers` too.

## Open gaps

- **No probe lives in this skill for this unit.** The constants came from a
  project harness, so re-measuring on another machine currently means porting
  that harness. A self-contained probe belongs here — and would also fix the
  missing job ids, which no amount of rewriting this file can recover.
- **The grid-barrier cost is `[I, UNMEASURED]`** — assumed ≈ launch cost, and
  load-bearing for the decision to reject cooperative launch. It is the claim a
  reader should check first.
- **The 0.23 µs/MB of L2 re-read** was measured on one fused kernel at 26% DRAM
  and 32% L1 utilisation. It is a single-body observation, not a machine
  constant, and is recorded here only as a warning that utilisation counters do
  not prove headroom is free.
