# Agent Note: The FFN GatedUp phase is at its cold-burst ceiling; warm the weights, do not retile the stream

Status: proposed

Date: 2026-09-02

## Problem

The persistent FFN task loop's GatedUp phase streams 16.8 MB of packed
gate/up weights per layer (128 tasks, one per CTA, each 4 x 32 KB TMA boxes
through a 3-deep ring from one producer warp) and was recorded as reaching
only ~40% of DRAM bandwidth (~13 GB/s per CTA against a 24 GB/s fair share),
with no mechanism identified. The rejected note
`.agents/notes/rejected/architecture/2026-09-02-ffn-taskloop-inflight-probes.md`
asked for a hardware-unit-test probe at exactly that geometry before any
further kernel change.

## Finding (measured, job 585478, ACD1-9, CUPTI)

The probe is `hardware-unit-test` sweep P (`tma.bw.dev.burst`): one cold
burst at the GatedUp geometry, ring depth, box and CTA count varied one axis
at a time, every launch L2-flushed.

- 128 CTAs x 1 warp x 4 x 32 KB, stages 3: **9.25 us = 1.81 TB/s** end-to-end
  (2.1 TB/s with the 1.2 us grid ramp removed). That IS the ceiling for a
  16.8 MB cold burst on this machine.
- Ring stages 2 / 3 / 4 / 6: 9.28 / 9.25 / 9.31 / 9.41 us -- depth is a null.
- 16 KB x 8 instead of 32 KB x 4: 9.44 us -- transaction count is a null.
- 132 CTAs: 9.54 us; 64 x 8 / 32 x 16 at the same bytes: 9.66 / 10.43 us --
  CTA count above ~32 is a null.
- The rate climbs only with the burst's bytes: 2.02 TB/s at 33.6 MB, 2.19 at
  67, 2.53 at 134, 2.80 at 268. The second 16.8 MB of a continuous stream
  costs 7.4 us where the first costs 9.3.

The kernel's GatedUp phase (~128 KB per CTA at ~13 GB/s = ~10 us per task;
NCU gu-alone 12.6 us replay-perturbed) is therefore at **roughly 90% of its
real copy ceiling**, not 40%. The "40% of DRAM" figure divided by the
steady-state `tma.bw.dev.dram` (3.02 TB/s), which a single 16.8 MB burst
cannot reach on this machine however it is issued. The mechanism is the
**cold-burst ramp**: DRAM delivery for one cold stream rises over its first
~100 MB, so a phase-sized burst averages ~60% of steady state.

## Finding 2 -- the warmth pair (jobs 588751 / 588754 / 588761, sweep Q, `tma.bw.dev.burst.warm`)

Acceptance criterion 1 was run before any kernel edit, with one correction to
its bar: ">= 3x faster than cold" was unreachable by construction, because
the L2 ceiling of the same 16.8 MB burst is itself 2.5x (3.7 us vs 9.3). The
pair is therefore read against that ceiling.

- A set brought into L2 by a **TMA load** reads at the ceiling (3.75 us) and
  keeps it through 10.5 and 21 MB of clean streaming traffic (3.74 / 3.90);
  it is half gone after 32 MB, and half gone after only 10.5 MB when L2 was
  full of dirty lines when the load ran.
- A set brought in by **`cp.async.bulk.prefetch.tensor.L2`** -- the
  instruction the proposal named -- reads at 6.6 us and is cold-like after
  21 MB. A plain SM streaming read leaves the burst cold (8.2 us).
- Collision control: an 8.4 MB DownResidual-sized burst is 5.7 us alone and
  5.9-6.1 us under a concurrent 16.8 MB read (inside the rows' spread).

So warming works, but only through real loads. The kernel candidate is
adjusted accordingly: the GatedUp-only CTAs (96 of 128, which have no
DownResidual slot and would exit) load layer L+1's gate/up column tiles into
their retired GatedUp weight frames and discard them, each also covering one
DownResidual owner's column; nothing is prefetched by instruction. Flag
`FFN_GU_NEXT_WARM`; the launch's legacy second packed pointer becomes the
next layer's set (the wrappers learn the cyclic layer order during the eager
warm-up forward that precedes capture).

## Proposal

Stop treating GatedUp's TMA geometry as the lever; it is measured flat. The
two levers the probe leaves open, in order:

1. **Warmth: prefetch layer L+1's gate/up weights into L2 during layer L's
   DownResidual phase.** DownResidual is a serial dependency chain reading
   DRAM at ~22%, i.e. ~78% slack over ~8 us -- about 24 MB of capacity for a
   16.8 MB set. With the set L2-resident the GatedUp burst runs against the
   L2 ceiling (`tma.bw.dev.l2`), ~2-3 us instead of ~9-10; the upside is
   ~6 us x 180 launches = ~1.1 ms of decoder time. This is the REVERSE of
   the rejected `c1a-dr-weight-l2-prefetch` experiment, which prefetched
   DownResidual weights during GatedUp -- into a phase that is already at its
   burst ceiling, where any extra request collides. Prefetching into the
   DownResidual window collides with nothing measured.
2. **Continuity: one weight stream per layer instead of two bursts.** Folding
   the DownResidual weight stream onto the tail of the GatedUp stream (same
   producer, no drain at the seam) saves the second burst's ramp, ~2 us per
   layer (~0.36 ms). This is the task-graph megakernel's continuity argument
   restated with a number, and it composes with (1).

Not proposed, measured null: ring depth 3 -> 4 for GatedUp, 16 KB boxes, a
second GatedUp producer warp, more or fewer GatedUp CTAs.

## Alternatives considered

- Retile GatedUp (2 CTAs/SM, deeper ring, wider boxes): the probe shows the
  ceiling does not move on any of those axes; rejected without a kernel
  candidate.
- Accept the phase as-is: leaves ~1 ms on the table that (1) can plausibly
  recover with a producer-side change only.

## Acceptance criteria

1. **HUT gate first (decisive pair):** extend sweep P with a warm variant --
   prefetch 16.8 MB into L2 (`cp.async.bulk.prefetch.tensor` or plain L2
   prefetch), run ~10 MB of unrelated streaming traffic (the attention chain's
   footprint between two FFN launches), then issue the same 128 x 4 x 32 KB
   burst. Pass if the warm burst reads >= 3x faster than the cold one; fail
   means L2 does not hold the set across the gap and (1) is dead before any
   kernel edit.
2. Kernel candidate for (1) behind an `FFN_NVCC_DEFINES` flag, issued from the
   reserved warp inside the DownResidual slot only; kernel gate
   `ffn_taskloop_parity.py --modes gu,dr,full`; e2e A/B/A via
   `sbatch/plan_e2e.sh` on both node generations, promotion bar -0.25 ms on
   the pdl plan decoder min (same bar as the PDL chain note).
3. The DownResidual phase's own time must not regress (its DRAM slack is the
   budget; NCU stall compare before/after).

## Risks

- L2 residency across the attention chain is unverified; criterion 1 is the
  guard, and it is a probe, not a kernel change.
- L2 prefetch traffic competes with DownResidual's own weight loads for the
  same DRAM channels; if DownResidual's chain is more latency-sensitive than
  its 22% suggests, the phase lengthens -- criterion 3.
- The burst constant is CUPTI kernel time with the grid ramp inside; inside a
  persistent kernel the ceiling is ~1.2 us lower, which strengthens rather
  than weakens the "already at ceiling" reading.
