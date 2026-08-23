# Worked example — Phase 0 on an H100 SXM5

A real calibration run, so the phase has an instance rather than only a rule.
Target: a fused bf16 decoder chain at M=50 on H100 SXM5, torch 2.11.0+cu130,
CUDA 13.0. **Clocks could not be pinned** (`nvidia-smi -lgc` denied), so ~6% is
the noise floor and every comparison below was taken **side by side in one
process, cold**, never across jobs.

Harness: `flash_vla.runtime.cuda.graph_time_cold` — N launches captured in one
CUDA graph, cycling distinct weight buffers so reads miss a 50 MB L2. Any
equivalent works; what matters is that it is **the same harness the kernel will
be accepted under**. See the `benchmark-kernel` skill for a generic one.

## 0a — shape-independent, once per machine

### 1. Empty-kernel cost vs grid size

Empty body, real thread count, real dynamic smem.

| CTAs | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| us | 0.95 | 1.20 | **1.24** | 1.24 |

Rises then saturates: this is **grid ramp**, not host launch overhead, and a
CUDA graph does not remove it. Corroborated by three real kernels that do almost
no work — a 25-CTA norm at 1.33 us, a 200-CTA combine at 1.49, a 64-CTA
projection at 1.76. **Every kernel in the design starts 1.2-1.8 us in debt.**

### 2. Cold streaming bandwidth vs transfer size

Pure read, 128 CTAs, cold.

| MB | 4.19 | 8.39 | 16.78 | 33.55 |
|---|---|---|---|---|
| us | 3.34 | 5.18 | 8.21 | 14.26 |
| GB/s | 1256 | 1619 | 2043 | 2353 |
| % of 3.35 TB/s peak | 37.5 | 48.3 | 61.0 | 70.2 |

Fit, within 6% across the range (worst row 8.39 MB at 5.8%; the other three are 0.6 / 3.7 / 2.1%):

```
    t_us = 1.85 + MB / 2.77
```

Marginal bandwidth **2.77 TB/s = 83% of peak**, behind 1.85 us of fixed cost
(1.24 grid ramp + ~0.6 DRAM ramp). **`bytes / peak_BW` is not a floor.** One
point is not enough — the curve is steep, and extrapolating from 4.19 MB alone
would have understated the 33 MB case by 40%.

### 3. Bandwidth vs CTA count

4.19 MB, cold.

| CTAs | 16 | 32 | 64 | 128 | 264 |
|---|---|---|---|---|---|
| us | 8.91 | 5.43 | 3.85 | **3.33** | 3.60 |
| GB/s | 471 | 773 | 1090 | 1260 | 1165 |

32 CTAs cost **1.63x** what 128 do for identical bytes — this is what justifies
split-K over a coarser tile. Two refinements the rule alone would miss: returns
flatten past 64 (64→128 is only 1.16x), and **264 is worse than 128**, so more
CTAs is not monotonically better.

### 4. Best existing implementation, same shapes

| shape | cuBLAS | model `1.85 + MB/2.77` |
|---|---|---|
| `mm (50,1024)@(1024,8192)`, 16.78 MB | 7.95 us | 7.91 us |

**The model is calibrated.** cuBLAS lands on it, so a floor derived from it is
trustworthy. Had cuBLAS come in far below, the model would have been too
pessimistic and every target derived from it wrong in the safe-looking direction.

Not a bound: on a second shape a hand-written fused kernel measured **7.94 us
against cuBLAS's 8.65** — a specialised kernel is supposed to go below a library
call. And for a fused kernel the honest reference is the **composition it
replaces**: cuBLAS's 16.6 us for two GEMMs omits the norm launch, the
elementwise pass over the intermediate, and the residual — roughly 20-21 us all
in, against the 22.0 the fused kernel achieved.

### 5a. Cluster and barrier costs

Empty kernel, 128 CTAs, 85 KB smem, varying only the attribute:

| | no cluster | cluster 2 | cluster 4 | cluster 8 |
|---|---|---|---|---|
| us | 1.297 | 1.278 | 1.273 | 1.270 |

**Declaring a cluster is free.** Synchronising one is not:

| | +1 `cluster_sync()` | +2 |
|---|---|---|
| us | 1.888 | 2.578 |

**0.65 us per barrier at cluster 8** (0.51 at cluster 4) — at *zero skew*, in an
empty kernel. It is a floor, not a portable price: **placement dominates it.**
Relocating one barrier from the end of a real kernel to beside its TMA issue,
hoping to hide it under load latency, cost **5.49 → 6.95 us**. At the end a
barrier only absorbs skew already being paid; at the start it *adds* fill skew
to the critical path.

## 0b — re-run once L2 has fixed the tile

Measurement 5's occupancy queries need the real smem and register budget, which
does not exist until the tile is chosen. Run it after L2, before committing.

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` and
`cudaOccupancyMaxActiveClusters`, at two real configurations:

| smem/CTA | blocks/SM | clusters of 8 placeable | grid needs |
|---|---|---|---|
| 85248 B | 2 | 30 | 16 → PASS |
| 207360 B | 1 | **15** | 16 → **FAIL, second wave** |

The two disagree **only** through the occupancy query, which is the evidence
that the cluster placer follows the query and not the SM count. Confirmed
behaviourally: at cluster 8 and 1 block/SM, 120 blocks reach a spin barrier and
128 time out.

`__launch_bounds__`'s second argument is **not** the lever — it is a *minimum*
blocks-per-SM hint constraining registers so the occupancy is reachable, and it
cannot cap occupancy. **smem is the lever.** Declare it anyway as a register
ceiling, so a later change that spills cannot silently drop the query to 1.

## What this bought

| | spec's original floor | measured floor | original target |
|---|---|---|---|
| kernel A, 5.24 MB | 1.56 us | **3.74** | 4.0 |
| kernel B, 4.19 MB | 1.25 us | **3.36** | 2.2 ← below its own floor |
| kernel C, 25.17 MB | 7.51 us | **10.94** | 10.0 ← below its own floor |

Two of the three shown were unreachable by *any* implementation, and the specs
carrying them could not be falsified: every later "we missed by 2x" was
unarguable. **This is what an hour of Phase 0 prevents.**

The corollary that changed the design: the 1.85 us is **per launch**, so launch
count is a first-class term. Six launches floor at `6*1.85 + 34.6/2.77 = 23.6`
us per layer; four at `19.9`. Essentially everything the fusion buys at the
floor is launch-count reduction, and none of it is tiling.

## What Phase 0 does NOT give you

Per-engine throughput — wgmma cycles per instruction, CUDA-core issue rate, TMA
issue latency. **L3's bubble check needs those and none of the five yields them.**
See `references/schedule-l3.md` for what to do instead: ordering edges and which
column is empty are *structural* and need no cycles; cycle counts are `[I]`, must
be marked, and the criterion is the **ratio** between columns, never the absolutes.

## Where this skill's own numbers come from

This skill holds specs to `[D]` / `[I]` / `TODO — needs source`; it owes the
same. Every measured figure in SKILL.md and the references is from one project —
an H100 SXM5 Pi0.5 decoder on this cluster, torch 2.11.0+cu130, clocks
**unpinnable** so ~6% is the noise floor. One machine's numbers, not constants:
**re-measure before porting them.**

| tag | what | source |
|---|---|---|
| `[MEAS-A]` | 0.30 eligible warps/sched, `short scoreboard` 42.7%, transform 24-27% of cycles, the 0.8 us TMA hoist, 0.23 µs/MB of L2 activation re-read | `ncu --set full` plus A/B rebuilds of **one** fused FFN kernel. Where it appears twice, that is one finding cited twice, not two corroborating ones |
| `[MEAS-B]` | cooperative + cluster rejected on sm90 | probe kernel, cluster 8 at 207360 B: `cudaOccupancyMaxActiveClusters` returns 15 against the 16 the grid needs, and cooperative+cluster returns `cudaErrorCooperativeLaunchTooLarge`. Reproduced behaviourally — 120 blocks reach a spin barrier, 128 time out |
| `[MEAS-C]` | 7.94 µs against cuBLAS 8.65 µs | split-K gated residual, M=50 K=4096 N=1024 bf16, both timed in one process, cold |
| `[I, UNMEASURED]` | grid barrier ≈ launch cost | inference. Load-bearing for rejecting cooperative and **not measured** — the claim a reader should check first |
| `[I]` | `~700 ns` HBM latency in the L3 example | a round figure for illustration |

The depth knee (`4→8 = 10.38→6.67 us` on one body, `8→11 = 14.45→14.99` on
another) is two kernels from that project — which is the rule's own point: a
slope does not transfer between bodies.
