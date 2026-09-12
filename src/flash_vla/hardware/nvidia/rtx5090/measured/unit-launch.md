# unit: launch — what a kernel costs before it moves a byte

The unit this repository is denominated in. Every fusion decision on `main`
prices a removed launch against sm90's `[ld.bw.dev.dram]`, `t_us = 1.85 + MB/2.77`.
This is the sm_120 replacement, and the answer splits: streaming bytes costs
about 1.8x more here, while a launch inside a CUDA graph costs 2.8x *less*.

Probe: `lab/sm120/launch_unit.cu`. Build and run:

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_120f,code=sm_120f -o /tmp/launch_unit lab/sm120/launch_unit.cu
/tmp/launch_unit 50 3      # reps=50 (median), sweeps=3 (for the spread)
```

## Claims, and what would have refuted them

**`[launch.lat.dev.ramp]` — a launch costs 0.45 µs in a graph and 2.05 µs in a
stream, and only the first of those is the deployed number.**

| CTAs | in a 64-node graph | back-to-back in a stream | graph/stream |
|---:|---:|---:|---:|
| 1 | 0.419 | 1.904 | 0.22× |
| 170 | 0.449 | 2.044 | 0.22× |
| 340 | 0.515 | 2.044 | 0.25× |
| 1024 | 0.802 | 2.050 | 0.39× |

*Isolation*: an empty kernel, so nothing but the launch is in the number. The
stream column brackets a 200-launch batch with one event pair, so per-launch
event cost is excluded; bracketing a single launch instead reads 2.34, and that
0.29 µs difference is the event cost, not the launch cost.

**This correction matters more than the number.** An earlier version of this
file quoted the 2.05 µs stream figure against sm90's `[launch.lat.dev.ramp]` of
1.24 µs and concluded launches here are 1.65× dearer. That comparison is
invalid: sm90's constant is explicitly the in-graph figure — its note says the
ramp is "not removed by graph capture" — and this repository captures its whole
forward into one graph and replays it, so a graph node is what a fusion decision
actually removes. On matched terms this part launches **cheaper**: 0.45 µs
against 1.24.

The second difference is qualitative. In a stream the cost is flat from one CTA
to a thousand, the fixed submission cost hiding everything. In a graph it
**rises with grid size**, 0.42 → 0.80 µs, which is the grid ramp becoming
visible once the submission cost is gone. And graph capture is worth 4.5× here,
where sm90's note says it removed nothing — that is a fact about that machine,
not a law.

**`[ld.bw.dev.dram]` — a cold read costs `3.35 + MB/1.524` µs.**

| MB | µs | GB/s | spread |
|---:|---:|---:|---:|
| 0.5 | 4.10 | 128 | 0.0% |
| 2 | 4.10 | 512 | 0.0% |
| 4 | 5.82 | 720 | 5.5% |
| 8 | 8.19 | 1024 | 0.0% |
| 16 | 14.02 | 1197 | 2.3% |
| 32 | 24.58 | 1365 | 0.0% |
| 64 | 45.06 | 1490 | 0.1% |
| 128 | 88.06 | 1524 | 0.0% |
| 256 | 169.98 | 1579 | 1.0% |

*Isolation*: no tensor core, no shared memory, no barriers. The read kernel
accumulates into a register and stores only on a condition that never fires, so
the loads cannot be eliminated and no store traffic enters the measurement. L2
here is 96 MB, so the flush writes 192 MB before every launch — a smaller
buffer would have left the set resident and measured a warm read.

*Falsifier*: the small sizes fitting the same line as the large ones. They do
not, and that is why the fit is taken over the linear region only. Over all
points it reads `3.50 + MB/1.533`; over ≥16 MB, `3.93 + MB/1.539`. Below ~8 MB
the cost is fixed-cost dominated and drags the intercept down.

*Second falsifier, and the one that changed the answer*: whether that 3.9 µs is
launch overhead. Re-timed as a **1-node CUDA graph replay**, with the L2 flush
outside the timed region, the same sweep fits `3.35 + MB/1.524` — agreeing with
the stream fit inside the 6% noise floor, at every size. So the fixed cost is
the memory system's own spin-up and not submission overhead, which is what an
event pair around a launch should measure in the first place: GPU-side kernel
duration. The graph figure is the one recorded, because it is the deployed
shape.

The marginal rate, 1.598 TB/s, is **89% of the 1.792 TB/s datasheet peak** —
this memory system is efficient. The problem is the 3.35 µs in front of it.

**`[ld.ctas.dev.knee]` — a cold read wants 2× the SM count.**

| CTAs | µs (16 MB) | GB/s | vs best |
|---:|---:|---:|---:|
| 42 (¼×SM) | 47.10 | 356 | 3.29× |
| 85 (½×SM) | 28.67 | 585 | 2.00× |
| 170 (1×SM) | 22.21 | 756 | 1.55× |
| **340 (2×SM)** | **14.34** | **1170** | **1.00×** |
| 680–2720 | 14.34 | 1170 | 1.00× |

*Falsifier*: the curve being flat, which would have made grid sizing a free
choice. It is not: one CTA per SM costs 1.55× what two do for identical bytes.

## What this says against sm90

| | sm90 (H100) | sm_120 (RTX 5090) | ratio |
|---|---:|---:|---:|
| per-launch cost, in a graph | 1.24 µs, rises with grid | **0.45 µs, rises with grid** | **2.8× better** |
| per-launch cost, in a stream | not recorded | 2.05 µs, flat | — |
| cold-read fixed cost | 1.85 µs | **3.35 µs** | 1.81× worse |
| cold-read marginal | 2.77 MB/µs (2.905 TB/s) | **1.524 MB/µs (1.598 TB/s)** | 1.82× worse |
| fraction of datasheet peak | 87% | 89% | — |
| cold-read CTA knee | 128 CTAs (0.97×SM) | 340 CTAs (2×SM) | — |

The consequence for this project is more mixed than it first looked, and worth
stating carefully because the two rows point opposite ways.

run-02 concluded that LingBot's expert is launch-bound because it is deep and
narrow: the same bytes moved by twice as many launches, each charged a fixed cost
before a byte moves. Two things happen to that here. The **per-launch** term gets
*cheaper* — 0.45 µs against 1.24, so the penalty for having many launches is 2.8×
smaller. The **per-phase** term gets dearer: every cold read pays 3.35 µs of
spin-up instead of 1.85, and streams its bytes at 1.82× the cost.

So a forward that is launch-count-bound should suffer less here than the H100
numbers suggest, and one that is bytes-bound should suffer about 1.8× uniformly.
Which of those LingBot's expert actually is at this shape is not settled by this
unit, and the arithmetic should be done with both terms rather than by
extrapolating either one.

## Noise floor

The worst spread across three separate sweeps of the same point was **5.5%**
(the 4 MB cold read); most points were under 2.9% and several were 0.0% at the
timer's 0.032 µs granularity. The machine block records 6%, matching sm90's
figure — not because it was assumed, but because it measured the same.

Clocks are not pinned. No number here was taken under a controlled clock, so
6% is an observed spread rather than a bound.
