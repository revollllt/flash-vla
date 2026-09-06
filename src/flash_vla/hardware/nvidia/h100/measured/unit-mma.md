# Unit: MMA / wgmma — the tensor core's issue rate

**Probe** `probes/units/mma_rate/mma_rate.{cu,py}` · **Constants** `wgmma.issue.wg.ss`, `wgmma.stages.wg.knee`,
`wgmma.ratio.sm.wg2`, `wgmma.clock.sm`, `wgmma.bytes.wg.tma`, `overlap.eff.sm`, `mma.issue.warp`,
`mma.stages.warp.knee`, `mma.xover.n.wgmma`, `mma.feedtax.warp.ldmatrix`,
`pipeline.ratio.sm.dep`, `pipeline.stages.wg.knee`
**Second probe** `probes/units/pipeline_ws/pipeline_ws.{cu,py}` (the coupled
mainloop below), **third** `probes/units/overlap/overlap.{cu,py}` (contention)

Two instructions, measured in one job on the same SMs: the warpgroup
`wgmma.mma_async` and the warp-level `mma.sync`. They are not
interchangeable and the choice between them is a tiling decision.

```
# launcher and output path are this host's; the probe itself takes neither
sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/units/mma_rate/mma_rate.py \
    --json profiles/hardware-unit-test/mma.json
```

One warpgroup issues `wgmma.mma_async.m64nNk16.f32.bf16.bf16` back to back out
of resident shared memory — no TMA, no global traffic, no barriers in the loop.
**Every rate is gated on a correctness check** (`M0`): one wgmma on random data,
D written out through the accumulator mapping, compared against torch at every
N. Measured relative error 5e-8 to 1e-7. A rate measured on an instruction
nobody verified is a measurement of an unknown.

## The four things to take away

**1. N must be ≥ 64.** Below it a per-instruction floor of ~19 cycles dominates
and most of the tensor core is idle:

| N | 8 | 16 | 32 | **64** | 96 | 128 | 192 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cycles/instruction | 18.9 | 20.7 | 24.7 | **32.8** | 48.7 | 65.0 | 97.1 | 129.3 |
| architectural ideal | 3.8 | 7.7 | 15.3 | 30.7 | 46.0 | 61.4 | 92.1 | 122.8 |
| % of peak | 20% | 37% | 62% | **94%** | 95% | 94% | 95% | 95% |

At N = 8 the instruction costs 18.9 cycles to do 3.8 cycles of work. Above 64,
cycles scale exactly with FLOPs — **N is free**, so take the largest the
register budget allows. Identical at 1 CTA and 132 CTAs, so this is a per-SM
rate. Landing at 94–95% of the architectural 2135 MAC/cycle/SM is itself the
evidence that no operand-path effect is throttling the measurement.

**2. Never write `wgmma.wait_group 0` in a mainloop.** It drains the pipeline
every iteration and costs 20–30%:

| per group → | 1 | 1 | 1 | 2 | 2 | 4 | 4 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wait stages | 0 | 1 | 3 | 0 | 1 | 0 | 1 | 0 |
| in flight | 1 | 2 | 4 | 2 | **4** | 4 | **8** | 8 |
| cycles/instruction | 79.0 | 42.0 | 33.7 | 55.6 | **33.1** | 44.6 | **32.3** | 38.2 |

At *equal* in-flight count the configurations disagree — 4 in flight is 33.1
with `wait 1` and 44.6 with `wait 0` — so the `wait_group` setting is its own axis, and
it is the one that matters. **Four in flight with wait ≥ 1** is the knee. That
is also a register decision: at N = 64 each group in flight is 32 accumulator
registers per thread, so pipeline stages and register budget are one choice.

**3. One warpgroup saturates the tensor core.** A second exactly doubles the
cycles each spends per instruction (33.1 → 64.4, 65.0 → 128.7, 32.3 → 64.3,
64.4 → 128.5 — all 1.94–1.99×), so aggregate throughput does not move.

> **The TFLOP/s column says the opposite and it is wrong.** Those same 2-warpgroup
> runs measure 703.6 → 842.7 TFLOP/s, a 20% "gain", purely because they happened
> to sit at 1.57 GHz against 1.31. A probe reporting only achieved TFLOP/s would
> have concluded a second math warpgroup is worth 20%. This is the strongest
> argument in this skill for measuring **cycles** on a machine whose clocks
> cannot be pinned.

Add a second math warpgroup only to hold more accumulators than 255 registers
allow, or to overlap an epilogue — never on the theory that it feeds the tensor
core faster.

**4. Below tile N = 32, use `mma.sync` instead — it is worth up to 3.1×.**
Which is the opposite of the usual "reach for the newest instruction" instinct.
On one axis, FLOP per cycle per SM, both measured in the same job:

| wgmma tile N | 8 | 16 | **32** | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| wgmma | 871 | 1577 | **2623** | 3959 | 4036 | 4053 |
| best `mma.sync` | 2680 | 2680 | **2680** | 2680 | 2680 | 2680 |
| winner | sync **3.1×** | sync **1.7×** | **tie** | wgmma 1.5× | wgmma 1.5× | wgmma 1.5× |

The comparison is like-for-like in coverage: `wgmma m64nNk16` covers a 64×N tile
with one instruction, and the `mma.sync` column covers the same tile with 4
warps of `m16n8k16`. The N = 32 row differs by 2%, inside the noise floor, so it
is a tie rather than a win either way.

That the wgmma small-N penalty is real and not a probe artefact is settled by
this same table: `mma.sync`, measured in the same job on the same SMs through
the same clock source, is *flat* in tile size and beats wgmma by 3.1× exactly
where wgmma's per-instruction floor bites.

## The warp-level instruction

`mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`, operands in registers, verified
against torch at 1.1e-7 relative error before any rate was read.

| independent accumulators | 1 | 2 | **4** | 8 |
|---|---:|---:|---:|---:|
| cycles/instruction | 25.14 | 12.54 | **6.26** | 6.29 |

**Latency is ~25 cycles, the issue interval is 6.26, so a warp needs exactly 4
independent accumulator sets** (25.14 / 6.26 = 4.02) — 16 f32 registers per
thread. One set is a latency reading, not a slow throughput reading; two halve
it and four quarter it, which is the 1/n a latency-bound pipeline gives.

| warps per SM | 1 | 2 | **4** | 8 |
|---|---:|---:|---:|---:|
| FLOP/cycle/SM | 655 | 1310 | **2618** | 2680 |

Linear to 4 warps, then flat. **The ceiling is ~2680 FLOP/cycle/SM = 63% of the
architectural peak**, against wgmma's 95% — reaching for `mma.sync` costs a
third of the tensor core even when everything else is perfect. That the 63% is
the instruction and not the probe is settled by the same probe reaching 95%
through wgmma.

Its lower utilisation lets the GPU clock *higher* (1.49–1.61 GHz against
wgmma's 1.06–1.34), so its achieved TFLOP/s flatters it relative to its
FLOP/cycle. Read cycles, not rates — again.

**Every engine on this machine wants four things outstanding**: 4 wgmma in
flight (`wgmma.stages.wg.knee`), 4 accumulator sets per warp (`mma.stages.warp.knee`), a TMA ring
of stages 4 (`tma.stages.warp.knee`). A useful default to reach for, and a cheap one to
check.

## The ratio this unit exists for -- and why it is a BYTE count

An L3 timeline that claims the math column covers the copy column is asserting a
ratio. The ratio was long written as "a CTA needs ~4.2 producer warps per math
warpgroup", and that wording hid its own most important term:

```
one wgmma m64n128k16 : 65.0 cycles @ 1.32 GHz = 49.2 ns
  consumes A(64x16) + B(128x16) bf16 = 6144 B   ->  124.8 B/ns
one TMA producer warp: box / 248 ns                   [tma.issue.warp]
```

Warps needed = 124.8 x 248 / box -- so the answer depends entirely on the
box size, which "4.2 warps" never stated:

| box | B/ns per warp | warps, isolated | warps, with contention |
|---|---|---|---|
| 8 KB | 33.0 | 3.78 | 4.50 |
| 16 KB | 66.1 | 1.89 | 2.25 |
| 32 KB (cap) | 132.1 | **0.94** | **1.12** |

At the descriptor cap ONE warp is enough. The recorded 4.2 was 36 KB expressed
at an unstated 8 KB box. Since `tma.bw.cta.warps` already establishes that
warps and box are interchangeable currency, the invariant is the product:

**~36 KB of in-flight `num_producers x box` per math warpgroup**, with
`overlap.eff.sm`'s measured contention included; 30 KB without it.

More than one producer warp is needed only because `tma.bytes.txn.max` caps one
transaction at 32 KB, just under the 36 KB required -- not because of anything
about warps.

## The pipeline, coupled -- E5b · `pipeline.stages.wg.knee`, `pipeline.ratio.sm.dep`

`overlap.eff.sm` removed the barrier on purpose. Adding it back with CUTLASS's
own `PipelineTmaAsync` warp-specialized mainloop (`sm90_pipeline.hpp`, mirrored
from `sm90_mma_tma_gmma_ss_warpspecialized.hpp`):

| N | stages | prod alone | cons alone | coupled | / slower | dependency | consumer duty |
|---|---|---|---|---|---|---|---|
| 128 | 2 | 94771 | 65809 | 226965 | **2.39** | 1.92 | 29% |
| 128 | 3 | 89014 | 65813 | 118937 | 1.34 | **1.07** | 55% |
| 128 | 4 | 88488 | 65789 | 118443 | 1.34 | **1.07** | 56% |
| 128 | 6 | 89165 | 65815 | 118734 | 1.33 | **1.07** | 55% |
| 64 | 4 | 89235 | 40130 | 114562 | 1.28 | 1.03 | 35% |

Cycles, 132 CTAs, median of 7. Coupled, the producer's and consumer's measured
spans agree to **0.03%** -- they are one window, which is what says the pipeline
is really coupled and not two things running loose.

**Measured twice, and the source changed the answer.** The first run had the
producer re-reading a single 24 KB tile: 132 CTAs shared one working set, the
walk never left L2, and it was fed at 9.9-14.8 TB/s -- above the DRAM ceiling,
so not a DRAM measurement at all. Re-run with a walking k-tile coordinate over
302 MB (5.8x L2):

| N=128, stages | L2-resident | cold |
|---|---|---|
| 2 | 2.39 | 2.21 |
| **3** | **1.34** | **1.57** |
| 4 | 1.34 | 1.32 |
| 6 | 1.33 | 1.27 |

The two regimes agree at 2 and 4 and **cross over at 3**. L2-resident, three
stages look sufficient; cold they cost 24% more than six. That is
`pipeline.stages.wg.knee` = **4 stages cold**, and the earlier reading of three
came from the L2-resident run.

**Falsifier.** If three stages really sufficed, S=3 and S=6 would agree cold as
they do L2-resident. They differ by 24%. And if the source did not matter, the
two regimes would not cross over exactly at S=3. `tma.stages.warp.knee`
says the same thing about the copy ring alone -- 4 stages for DRAM, 2 for L2 --
so the pipeline is inheriting its stages requirement from the source's latency,
exactly as `tma.lat.warp` predicts.

**The hand-off is still nearly free** -- `pipeline.ratio.sm.dep` = **1.06**.
1.32x over the slower side at four stages, of which `overlap.eff.sm` explains
1.25x as engine contention, leaving the barrier at ~1.05x. Both numbers are now
cold, so the division no longer mixes regimes.

**Falsifier.** If the barrier were the dominant term rather than stages, S=4 and
S=6 would differ as much as S=2 and S=4 do. They differ by 4% against 67%. And
the consumer-only baseline reproduces `wgmma.issue.wg.ss` independently, so the
denominators are not the probe's own invention.

**Isolation.** One kernel, three modes on the same SMs: coupled, producer-only
(self-draining its own full barrier) and consumer-only (re-reading a resident
stage, no waits). Coupled, the two sides' spans agree to 0.03%, which is the
check that they are one window.

REGIME (rule 6b): cold on both sides -- the producer walks 302 MB (5.8x L2) and
`overlap.eff.sm`'s 1.25x divisor was itself measured on a cold 256 MB walk.
132 CTAs, N=128, BK=64, 12288 k-tiles, median of 7. Job 568758 (supersedes
567094), `probes/units/pipeline_ws/pipeline_ws.py`.

**The consumer is starved 54% of the time** at four stages -- worse cold than
the 44% measured in cache. Whether the copy path can feed one warpgroup at
N = 128 is the open question `wgmma.bytes.wg.tma` boxes, and this is the
strongest evidence yet that it cannot.

> **The N = 256 row is excluded.** Its producer is DRAM-bandwidth-bound, and
> there cycles are the wrong unit -- a concurrent consumer lowers the clock, the
> same wall-clock wait spans fewer cycles, and the row reads a spurious 0.72x
> "speedup" at 56.6% run-to-run spread. Protocol rule 10 was refined for exactly
> this, and re-measuring on 2026-08-30 confirmed the instability rather than
> resolving it: 62.9% spread pre-migration and 64.3% post, against <= 9.3% on
> every other row of the sweep. No constant rests on it. [rule 14b]

## The clock, and why cycles are the unit

Under sustained wgmma load this GPU runs at **1.05–1.58 GHz**, not the 1.755 GHz
the datasheet's 989 TFLOP/s assumes. So the architectural peak is reached *in
cycles* and missed *in wall clock*: best observed ≈ **850 TFLOP/s**. Target
cycles against `wgmma.issue.wg.ss`; treat 850 as the practical ceiling, the way `ld.bw.dev.dram`
replaces the 3.35 TB/s datasheet bandwidth.

## Three build traps, either of which yields a plausible wrong answer

- **`-arch=sm_90a` is not enough.** On this toolchain it resolves to virtual
  arch `compute_90`, and ptxas then rejects `wgmma.fence`,
  `wgmma.commit_group` and `wgmma.wait_group` outright. Use
  `-gencode arch=compute_90a,code=sm_90a`. (It fails loudly here — but only
  because the probe uses those instructions directly.)
- **Accumulators need `warpgroup_fence_operand`.** Without it ptxas emits
  `C7515` and *serialises* the wgmma pipeline; the loop still runs and still
  produces a number, and that number is the serialised rate. The probe treats
  C7515 as a **build failure**, not a warning.
- **The instruction must be selected, not spelled.** The wgmma path reaches its
  atom through `GMMA::ss_op_selector` — the same dispatch CUTLASS itself uses —
  so the probe cannot measure an instruction production code would never have
  issued. A `static_assert` per swept N pins that choice to the
  `MMA_64xNx16_F32BF16BF16_SS` atom the constants above were measured on, so a
  library update that changes the dispatch fails the build rather than quietly
  re-measuring a different instruction under these tags. The accumulator's
  `(row, col)` mapping comes from `partition_C` for the same reason: it is
  correct by construction at every N, where a hand-decoded m64nNk16 layout was a
  second table to keep in step.

The warp-level `mma.sync` and `ldmatrix` probes keep their inline PTX
deliberately: their subject *is* the raw instruction and its documented fragment
layout, both already gated on a torch comparison, and routing them through a
`TiledMMA` would add the partitioning the measurement exists to exclude.

## Open gaps

- **fp8 and f16 accumulate.** Only bf16-in/f32-out is measured, for both
  instructions. The published ridge point doubles with fp8, so a single bf16
  number would condemn every fp8 tile ever written — measure both or claim
  neither.
- **A from registers.** Only the smem–smem (`_SS`) form is measured; the `_RS`
  form's cost, and whether it is worth its register pressure, is untested.
- **Swizzled operand layouts.** The probe uses the unswizzled `Layout_K_INTER`
  atom. Landing at 94–95% of peak says the layout is not throttling *this*
  measurement, but it does not measure what a swizzled layout costs or saves.
- **Why the clock moves** with configuration in ways the work does not explain.
