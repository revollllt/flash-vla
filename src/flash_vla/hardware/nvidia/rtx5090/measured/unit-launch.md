# unit: launch — what a kernel costs before it moves a byte

The unit this repository is denominated in. Every fusion decision on `main`
prices a removed launch against sm90's `[ld.bw.dev.dram]`, `t_us = 1.85 + MB/2.77`.
This is the sm_120 replacement, and it is worse on both terms.

Probe: `lab/sm120/launch_unit.cu`. Build and run:

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_120f,code=sm_120f -o /tmp/launch_unit lab/sm120/launch_unit.cu
/tmp/launch_unit 50 3      # reps=50 (median), sweeps=3 (for the spread)
```

## Claims, and what would have refuted them

**`[launch.lat.dev.ramp]` — a launch costs 2.05 µs and grid size does not
change that.**

| CTAs | batched (µs/launch) | single launch + 2 events | delta |
|---:|---:|---:|---:|
| 1 | 1.962 | 2.336 | 1.19× |
| 170 | 2.050 | 2.336 | 1.14× |
| 340 | 2.050 | 2.336 | 1.14× |
| 1024 | 2.049 | 2.336 | 1.14× |

*Isolation*: an empty kernel, so nothing but the launch is in the number.
*Falsifier*: the batched and single-launch columns disagreeing by more than the
event cost, or the batched column rising with grid size. Neither happened — and
the second is the interesting negative. sm90's `[launch.lat.dev.ramp]` **rises**
with grid (0.95 µs at 32 CTAs, 1.24 at 256): that is the grid ramp. Here the
cost is flat from one CTA to a thousand, so there is no ramp to amortise and a
small launch is pure overhead.

The single-launch column exists to catch the methodological error, not to be
quoted: bracketing one launch with two events measures launch *plus* events.
The 0.29 µs difference is the event cost.

**`[ld.bw.dev.dram]` — a cold read costs `3.93 + MB/1.539` µs.**

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
not, and that is why two fits are reported. Over all points the fit is
`3.45 + MB/1.532`; over the linear region (≥16 MB) it is `3.93 + MB/1.539`. The
second is the one to quote, because below ~8 MB the cost is fixed-cost dominated
and drags the intercept down while barely moving the slope.

The marginal rate, 1.613 TB/s, is **90% of the 1.792 TB/s datasheet peak** —
this memory system is efficient. The problem is the 3.93 µs in front of it.

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
| per-launch cost | 1.24 µs, rises with grid | **2.05 µs, flat** | 1.65× worse |
| cold-read fixed cost | 1.85 µs | **3.93 µs** | 2.12× worse |
| cold-read marginal | 2.77 MB/µs (2.905 TB/s) | **1.539 MB/µs (1.613 TB/s)** | 1.80× worse |
| fraction of datasheet peak | 87% | 90% | — |
| cold-read CTA knee | 128 CTAs (0.97×SM) | 340 CTAs (2×SM) | — |

The consequence for this project is specific. run-02 concluded that LingBot's
expert is launch-bound because it is deep and narrow: the same bytes moved by
twice as many launches, each charged 1.85 µs before a byte moves. **On this
machine that charge is 3.93 µs.** A workload that was launch-bound on H100 is
more launch-bound here, not less — and the extra SMs (170 against 132) do not
help, because the cost does not scale with grid.

## Noise floor

The worst spread across three separate sweeps of the same point was **5.5%**
(the 4 MB cold read); most points were under 2.9% and several were 0.0% at the
timer's 0.032 µs granularity. The machine block records 6%, matching sm90's
figure — not because it was assumed, but because it measured the same.

Clocks are not pinned. No number here was taken under a controlled clock, so
6% is an observed spread rather than a bound.
