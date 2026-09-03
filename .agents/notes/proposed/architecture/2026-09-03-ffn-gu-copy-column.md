# Agent Note: the GatedUp phase has two stacked floors, and the DRAM one binds

Status: proposed — the finding closes a family rather than opening a candidate

Date: 2026-09-03

## Problem

Two measurements of the decoder FFN's GatedUp phase contradicted each other,
and six consecutive candidate lanes had been rejected without the
contradiction being resolved.

- Deleting the math and the epilogue but keeping the copies left **9.6 us** of
  the phase's 11.9. That is 16.8 MB at 1.75 TB/s, within 3% of the machine's
  cold-burst delivery rate for a burst that size, which reads as
  "bandwidth-bound".
- Making the same weight stream free by construction — pinning every stage to
  its task's first K tile so the reads repeat one 32 KiB address — bought only
  **1.37 us**. If the phase were bandwidth-bound that should have been worth
  about 5.6 us, the distance to the machine's warm-burst figure.

A phase cannot be at its bandwidth ceiling and indifferent to its bandwidth.

## Finding

Both measurements are correct. The inference that joined them was not: the
phase has **two floors, stacked 20% apart**, and each measurement moved one
of them while the other took over.

Evidence, job 591174, one job and one node (ACD1-32) so the kernel and the
machine are directly comparable — this pairing is what previous lanes could
only do across jobs, where the baseline drifts 11.56-12.45 us:

| | us | GB/s |
|---|---:|---:|
| kernel copy column, real addresses (cold) | 9.563 | |
| machine, same geometry, cold (`tma_ring` sweep Q) | 9.41 | 1783 |
| kernel copy column, reads repeat one address, L2 allowed to keep them | 7.685 | |
| machine, same geometry, warm — the L2 delivery ceiling | 3.71 | 4520 |

The kernel's cold column sits **within 1.6% of the machine's cold delivery
rate**: in production GatedUp is at its cold-DRAM ceiling, and the original
`2026-09-02-ffn-gu-dram-ceiling` reading was right.

Underneath it sits a second floor at **~7.7 us** that no byte-side change can
go below. When the weight bytes are made maximally cache-friendly the phase
stops at 7.685 us while the machine can deliver the same geometry warm in
3.71 — a **3.97 us/layer gap that is the kernel's ring protocol**, the
empty/full barrier round trips the bare probe does not pay. The gap is
conservative: the kernel row touches 4 MB distinct where the probe re-reads
the full 16.8 MB from L2, so if anything the kernel had the easier job.

The two floors are 9.41 and ~7.7. Removing either one alone leaves the other
binding, which is why every lane that removed one measured almost nothing.

**This model predicts the ring-depth result that produced it, in both
directions.** Depth 3 -> 2 costs 1.5-2.1 us because it lifts the protocol
floor above the DRAM floor; depth 3 -> 4 and 3 -> 5 buy nothing because the
protocol floor is already below the DRAM floor and lowering it further changes
nothing. A one-sided gradient is what two stacked floors look like.

Two hypotheses are refuted along the way. The weight loads carry
`L2Hint::kEvictFirst`, so the pinned bound might never have been resident:
dropping the hint on the bound is worth only 0.36-0.38 us (jobs 591136,
591150), nowhere near the ~4 us that reading would need. And the agreement
between 9.6 us and the cold-burst constant is not a coincidence to be
explained away — it is the mechanism.

**Production finding, incidental but shipped:** removing the evict-first hint
makes GatedUp **slower by 1.12 us/layer** (11.89 -> 13.01, job 591136;
9.56 -> 10.54 on the column alone, job 591150), about 0.20 ms over the
decoder's 180 launches. The hint is load-bearing — it keeps the one-shot
weight stream from displacing the reused XFS working set, exactly as its
comment claims — and must not be removed as a simplification.

## Proposal

**Close the GatedUp optimization family.** In production the phase is within
2% of what this machine can deliver for its byte count and geometry, and the
only remaining lever is moving fewer bytes, which is a numerics or model
decision rather than a kernel one.

Specifically, do not build the "fewer, larger frames" candidate the accounting
lane pointed at. Halving the four trips to two 64 KiB frames would lower the
protocol floor, but the protocol floor is already ~1.7 us *below* the DRAM
floor that binds; the phase would not move. That candidate only becomes
interesting if the weight bytes themselves get cheaper, in which case the
protocol floor becomes the binding one and this note's 3.97 us gap becomes its
prize.

## Alternatives considered

- **Build fewer/larger frames anyway**, on the accounting lane's reading that
  in-flight frames are the pole: rejected here, because the pole it named is
  the lower of two floors and is not the one carrying the phase.
- **Attack the protocol floor first, then the bytes**: the ordering is wrong.
  Nothing in scope makes the bytes cheaper, so the protocol floor stays hidden.
- **Keep pursuing warmth** (prefetch, cross-layer continuity): already
  measured null twice, and this note explains why — warmth removes the DRAM
  floor and lands on the protocol floor 20% below it.

## Acceptance criteria

This note asks for a decision, not an implementation. It is accepted when the
GatedUp family is recorded as closed and the accounting note's "in-flight
weight frames is the pole" reading is annotated as superseded: the copy column
is 83% of the phase because it runs at the machine's cold delivery ceiling,
not because of frame count.

If the family is ever reopened, the entry condition is explicit: a change that
reduces GatedUp's weight bytes. Only then does the 3.97 us protocol gap become
addressable, and the first candidate is two 64 KiB frames (two 32 KiB boxes
under one barrier with a doubled expect-tx), which must clear 1.0 us/layer on
the fused column before earning a route job.

## Risks

- The probe-vs-kernel gap compares two code paths, not one kernel with a knob.
  They share geometry, node, job and clock regime, but not their consumers:
  the probe's ring free-runs where the kernel's is cycled by a math warpgroup
  with its wgmma deleted. The 3.97 us is therefore an upper bound on the
  protocol tax, and a lower bound on how much of the warm gap is structural.
- Single job per row, one median of 50 reps. The reconciliation does not turn
  on small deltas — the claims are a 1.6% agreement and a 2.07x gap — but the
  0.36 us hint-on-bound difference is at the resolution limit and is used only
  to refute, not to size anything.
- `tma_ring` sweep Q's warm rows were reproduced here at 3.71-3.74 us, matching
  the constant recorded from jobs 588751/588754/588761, so the machine side is
  not a one-off.
