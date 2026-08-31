# Roadmap — the units not yet measured, and the pairs that would settle them

Most of this is **planned, not measured**; rows marked **done** have landed in
`<arch>/constants.yaml` and are cited from there, not from here.

> **This table is a roadmap, not a record.** Values here go stale the moment a
> constant is re-measured -- `tma.bw.dev.dram` sat at "3.02 T" here for a day
> after it had been corrected twice in `constants.yaml`. Read the value from
> `scripts/constants.py`; read only the STATUS from this page. A tag on this page does not exist
in any `<arch>/constants.yaml` yet and must not be cited in a floor. It is here
because `protocol.md`'s "Adding a unit" starts with the questions and the
decisive pair, and those are worth reviewing before a probe is written rather
than after.

Naming follows [`naming.md`](naming.md). Status in the tables below:
**have** = already measured, listed for context · **bound** = only a bound
exists and needs bracketing · **new** = not measured.

## Why these three

The existing constants are ceilings measured in isolation, which `protocol.md`
rule 2 makes explicit: a real kernel gets at most the constant. Three questions
a kernel author actually asks are either unmeasured, or measured only up to the
point where the answer stops being useful.

1. **Bandwidth has four ceilings and one is measured.** `tma.bw.dev.dram` is
   the DRAM aggregate. There is no measured per-CTA term at all -- delivery is
   linear in bytes in flight with no plateau. The **per-SM** ceiling has never
   been measured, and `tma.bw.dev.l2`
   found no L2 ceiling either, only a lower bound at the largest CTA count
   tried. What still does not close is the shape of the frontier: delivery rises
   linearly per CTA with no plateau, yet the measured curve still climbs past
   `tma.bw.dev.dip` at 36-56 CTAs. Something between the CTA and the memory
   system is unaccounted for, and every "how few CTAs" answer rests on it.

2. **Instruction choice is measured in one regime only.** `wgmma.issue.wg.ss`,
   `wgmma.stages.wg.knee` and `mma.xover.n.wgmma` all describe a saturated tensor
   core with enough independent work -- the large-GEMM case. Nothing measures an
   output tile too narrow to fill the instruction's fixed shape, or a pipeline
   with too little independent work to hide dependent latency. The `_RS` wgmma
   form is an open gap.

3. **Warps and CTAs are treated as one currency.** `tma.bw.cta.warps` shows that
   for TMA *issue*; `sched.ctas.sm.knee` shows a grid must reach 3x the SM count
   before extra CTAs become resident warps. Neither says which *arrangement* of
   a fixed warp budget hides which *stall*.


### Unit `bw` — bandwidth by source and aggregation level

| Tag | Meaning | Unit | Exp | Status |
|---|---|---|---|---|
| `tma.bw.sm.max` | max TMA a single **SM** absorbs, any CTA split | GB/s | E1 | **open** -- needs a cold measurement |
| `tma.bw.sm.dram` | per-SM share at a full grid, DRAM-sourced | GB/s | E1 | **open** -- must be measured, not divided out of the device figure |
| `tma.bw.sm.cta.scale` | what the 2nd co-resident CTA is worth | ratio | E1 | **done** <=1.0 |
| `tma.bw.dev.dram` | device-wide ceiling, DRAM-sourced | GB/s | E2 | **done** <=3.17 T (upper bound) |
| `tma.bw.dev.l2` | device-wide ceiling, L2-resident source | GB/s | E2 | **done** 18.3 T |
| `tma.bw.sm.l2` | per-SM ceiling, L2-sourced | GB/s | E2 | **open** -- fell with `tma.bw.sm.max` |
| `tma.ctas.dev.dram.knee` | fewest CTAs that saturate DRAM | count | E3 | **done** 128 @8KB, 32 @32KB |
| `tma.ctas.dev.l2.knee` | fewest CTAs that saturate L2 | count | E3 | not reached in range |
| `tma.bind` | which of {cta, sm, l2, dram} binds, given (CTAs, bytes/CTA, source) | rule | E4 | **new**, derived |
| `tma.bw.sm.wgmma` | TMA delivered while wgmma runs | GB/s | E5 | **done** -- recorded as `overlap.eff.sm`, a slowdown not a rate |
| `wgmma.issue.wg.tma` | wgmma issue interval while TMA runs | cyc | E5 | **done** -- the 1.05x half of `overlap.eff.sm` |
| `overlap.eff.sm` | measured joint / product of the two isolated constants | ratio | E5 | **done** 1.25x |

`tma.ctas.knee.dev.*` is the direct answer to "how few CTAs saturate", which is
today only obtainable by reading a curve.

### Unit `mma-regime` — instruction choice by regime

| Tag | Meaning | Unit | Exp | Status |
|---|---|---|---|---|
| `wgmma.issue.wg.ss` | cycles per `m64nNk16`, operands from smem | cyc | E6 | have |
| `wgmma.issue.wg.rs` | same, A from registers | cyc | E7 | **new** |
| `wgmma.feedtax.wg.rs` | cost of getting A into registers, measured not assumed | ratio | E7 | **new** |
| `wgmma.regs.wg.rs` | register cost of the RS form | count | E7 | **new** |
| `wgmma.util.wg.m` | useful / issued FLOP at M = 1,8,16,32,64 (M is fixed at 64) | ratio | E6 | **new** |
| `mma.issue.warp` | cycles per `m16n8k16` | cyc | E6 | have |
| `ffma.issue.warp` | CUDA-core dot-product baseline | cyc/FLOP | E6 | **new** |
| `mma.xover.m.narrow` | M at which the winning instruction changes | count | E6 | **new** |
| `wgmma.lat.wg.ss` | **dependent** latency, one accumulator chained | cyc | E8 | **new** |
| `wgmma.lat.wg.rs` | same, RS form | cyc | E8 | **new** |
| `mma.lat.warp` | dependent latency of `m16n8k16` | cyc | E8 | **open** -- the ~25 was read off `mma.stages.warp.knee`s 1-accumulator row, never isolated |
| `sched.wave.quantum` | CTAs in one wave, as scheduled rather than as SM count | count | E9 | **new** |
| `sched.wave.cliff` | cost of the 132 -> 133 CTA step at fixed work | ratio | E9 | **new** |

`ffma.issue.warp` is the one that decides whether the tensor core is worth
entering at all in the narrow regime, and nothing here has it today.

### Unit `latency-hiding` — arrangement against stall kind

| Tag | Meaning | Unit | Exp | Status |
|---|---|---|---|---|
| `sched.hide.sm.ilp` | warps-per-CTA vs CTAs, against a dependent arithmetic chain | ratio | E10 | **new** |
| `sched.hide.sm.mem` | same, against a pointer chase | ratio | E10 | **new** |
| `sched.hide.sm.bar` | same, against a `__syncthreads` loop | ratio | E10 | **new** |
| `sched.warps.knee.sm.ilp` | warps per SM past which ILP stalls stop shrinking | count | E11 | **new** |
| `sched.warps.knee.sm.mem` | same, memory stalls | count | E11 | **new** |
| `sched.warps.knee.sm.bar` | same, barrier stalls | count | E11 | **new** |

This unit claims the **ordering** of arrangements per stall kind, never the
magnitudes — a stall generator is a kernel body, and `protocol.md` is explicit
that a slope measured on one body does not carry to another.

## How each experiment is run

Every entry names the pair that discriminates, per `protocol.md` step 2.

**E1 — per-SM vs per-CTA.** Extends `tma_ring`. Equal aggregate product, equal
per-SM product: 132 CTAs at 1/SM x 36 KB against 264 CTAs at 2/SM x 18 KB.
Ceiling per-SM -> the two match; per-CTA -> the 2/SM case delivers 2x per SM.
Co-residency is forced by smem under half of 232448 B and *verified* with
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` plus a resident-CTA counter in
the kernel, not assumed — `sched.ctas.sm.knee` is the standing warning that the
scheduler spreads before it stacks. Falsifier: if the ~133 GB/s reading were per-CTA, 264 CTAs would deliver ~2x the
aggregate of 132 at the same per-CTA product; the frontier says they do not.
That is settled: there is no per-CTA ceiling. What this experiment still owes is
the per-SM ceiling, which has never been measured cold.

**E2 — L2 ceiling.** Footprint already selects the source. Hold 4 MB
(L2-resident) and walk the product to 2x and 4x the point where DRAM saturated.
Flat -> the constant stays a **bound** and is recorded as one; a bend is the
ceiling. Repeat E1's pair at 4 MB to get `tma.bw.cta.l2` and `tma.bw.sm.l2`.

**E3 — knees per source.** For each source, walk per-CTA bytes at fixed grid
(gives `tma.bytes.cta.*.knee`), then walk CTA count at fixed per-CTA bytes (gives
`tma.ctas.dev.*.knee`). Both directions of the same frontier, which is what makes the
knee falsifiable rather than a rearrangement of one fit (rule 5).

**E4 — binding rule.** Pure derivation from E1-E3 into `scripts/frontier.py`;
no GPU. Ships as a function, not prose.

**E5 — overlap.** One kernel: N TMA producer warps plus one wgmma consumer
warpgroup, sweeping N. Read against `tma.bw.sm` and `wgmma.issue.wg.ss`
measured in the same job. `overlap.eff.sm` < 1 is contention and its slope in N
is how much. This retires `wgmma.bytes.wg.tma`, which is arithmetic across two kernels.

**E6 — narrow tiles.** Fix useful FLOPs; sweep M = 1, 8, 16, 32, 64. Three
implementations of the same product: wgmma `m64nNk16`, `mma.sync m16n8k16`,
and an FFMA dot. Report cycles per **useful** FLOP, so the fixed-shape waste
appears as the quantity it is. Falsifier: if shape waste dominated, wgmma's
useful throughput at M=1 would be 1/64 of its M=64 value.

**E7 — RS vs SS.** Same N, same in-flight stages, A from registers vs smem, plus
the load cost to fill those registers measured the way `mma.feedtax.warp.ldmatrix` measured
the `ldmatrix` tax. Gate on the existing `--check` path at every N.

**E8 — dependent latency.** One accumulator chained (latency) against many
independent (throughput), per instruction and form. `mma_rate.py` already has
this axis for `mma.sync`; extend it to wgmma SS and RS.

**E9 — wave quantisation.** Fixed total work at 66 / 132 / 133 / 264 CTAs. The
132 -> 133 step is the decisive one: it isolates quantisation from any smooth
occupancy effect.

**E10/E11 — arrangement against stall.** New probe, stall kind a runtime
parameter, each kind read against its own no-stall control. Fixed 264 warps as
33x8, 132x2, 264x1. Hypotheses to kill: same-SM warps should hide *instruction*
latency where extra SMs cannot; extra CTAs should hide *memory* latency from
either arrangement; a wider CTA should *hurt* under barriers where more CTAs do
not. Any one failing to separate is the result worth having.


## What each unit must clear before its constants are usable

- Every constant passes `scripts/constants.py --validate`, and its falsifier --
  a real sweep row, not a demonstration -- is recorded in the unit reference.
  The published table stores value/units/short/rule; the evidence lives in
  `<arch>/unit-<name>.md`. [protocol.md, "The shape of a unit"]
- E1 and E2 either produce a ceiling with points on both sides of the knee, or
  are recorded as **bounds with the range tried** -- never a ceiling inferred
  from a monotone fit (rule 5). `tma.bw.dev.l2` is the live example.
- Cycles, not TFLOP/s, wherever a clock difference could carry the result
  (rule 10). `wgmma.ratio.sm.wg2` is the standing example of what ignoring that
  costs.
- Each new probe is gated on a host-reference correctness check before any rate
  is read (rule 11).
- Nothing already answered is re-measured: E3 extends `tma.bw.dev.l2` rather
  than repeating it, and E7/E8 extend the existing `mma` sweeps.

## Known traps in the framing

- **"Memory bound" merges two different failures.** A narrow matrix has low
  arithmetic intensity, but at M = 1 the binding constraint is usually not
  bandwidth -- it is that wgmma's fixed M = 64 wastes 63 of 64 rows. The fixes
  diverge: more bytes in flight for the first, a different instruction for the
  second. E6 measures them apart, and the unit reference must name them apart or
  the constant gets spent wrongly.
- **"Latency bound" merges too little ILP with too few waves.** Deeper
  pipelining fixes the first; more CTAs or a persistent kernel fixes the second.
  E8 and E9 are separate questions for that reason.
- **`latency-hiding` may not transfer.** A stall generator is a kernel body, and
  `protocol.md` is explicit that a slope measured on one body does not carry to
  another. That unit claims the *ordering* of arrangements per stall kind, never
  the magnitudes.
- **E1 may not be separable.** If the scheduler refuses 2 CTAs/SM at any smem
  budget that still allows a useful box, the per-SM ceiling is unreachable by
  this method and the experiment must say so rather than report the per-CTA
  number twice.
