---
id: pattern-serial-epilogue-owner
title: "Widening a tile lengthens the critical path when one worker owns the epilogue"
type: pattern
architectures: [sm90]
tags: [split-k, reduction, epilogue-fusion, tile-scheduling]
symptoms: [tile-widening-regression, epilogue-bound, latency-bound]
candidate_techniques: [technique-reduction-own-task-kind, technique-epilogue-fusion]
related: [pattern-fusion-latency-chain, pattern-epilogue-bound-short-k, pattern-wgmma-tile-n-floor]
sources: [note-2026-09-06-optimization-campaign-plan, note-2026-09-03-ffn-dr-wide-tile]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-03-ffn-dr-wide-tile
---

# Widening a tile lengthens the critical path when one worker owns the epilogue

## Symptom

A split-K phase where the partials are folded and the residual
read-modified by a single designated worker per output tile (split 0, or the
last arriver). Widening the output tile looked like a win, fewer tiles,
fewer joins, fewer counter observes, a wider MMA atom, and the phase got
slower.

## Likely Causes

A join's cost is the counter observe every owner pays, plus that owner's
serial epilogue. The fold and the residual read-modify-write are
proportional to output area, which the tile width does not change, so
doubling the tile doubles what one owner does while halving how many owners
run concurrently. Wall time follows one owner's serial chain and goes up.
Halving a join count does not halve a join cost when the joins already ran
in parallel. The MMA-atom gain applies only to the compute floor, which in a
latency-bound phase is a small fraction of its time: a 62% to 93% atom
efficiency on a 0.6 us floor inside a 10 us phase returns about 0.2 us and
cannot pay for a doubled chain.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Reduction as its own task kind](../techniques/reduction-own-task-kind.md) | Distributes the fold across the splits instead of one owner |
| [Epilogue fusion](../techniques/epilogue-fusion.md) | Distribute the epilogue across all splits; then widening pays |

## Diagnosis Checklist

```
1. Count the epilogue work per owner, not per phase.
2. Check which the phase is: epilogue distributed across all splits
   (widening pays) or owned by one worker (widening regresses regardless of
   how many joins disappear).
```

## Caveats

The reverse move (narrower tiles, more owners) is bounded by the counter and
launch costs it multiplies, and by the wgmma N floor
(`pattern-wgmma-tile-n-floor`).
