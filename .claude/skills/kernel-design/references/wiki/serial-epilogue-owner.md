---
id: serial-epilogue-owner
type: pattern
arch: sm90
tags: [split-k, reduction, tile-shape, epilogue, latency]
confidence: measured
---

# Widening a tile lengthens the critical path when one worker owns the epilogue

## Context

A split-K phase where the partials are folded and the residual read-modified
by a single designated worker per output tile (split 0, or the last
arriver). Widening the output tile looks like a win: fewer tiles, fewer
joins, fewer counter observes, and a wider, more efficient MMA atom.

## Move

Count the epilogue work **per owner**, not per phase. The fold and the
residual read-modify-write are proportional to output area, which the tile
width does not change — so doubling the tile doubles what one owner does
while halving how many owners run concurrently. Wall time follows one
owner's serial chain and goes up. Halving a join *count* does not halve a
join *cost* when the joins already ran in parallel.

Before widening, check which of these the phase is: if the epilogue is
distributed across all splits, widening pays; if one worker owns it,
widening is a regression regardless of how many joins disappear.

## Why it works

A join's cost is the counter observe every owner pays, plus that owner's
serial epilogue. Neither shrinks with fewer tiles. Meanwhile the MMA-atom
gain applies only to the compute floor, which in a latency-bound phase is a
small fraction of its time — a 62% -> 93% atom efficiency on a 0.6 us floor
inside a 10 us phase returns ~0.2 us and cannot pay for a doubled chain.

## Caveats

The reverse move (narrower tiles, more owners) is bounded by the counter and
launch costs it multiplies, and by the wgmma N floor. If the epilogue is the
binding term, the fix is to distribute it across the splits or give the
reduction its own task kind (`reduction-own-task-kind`), not to change the
tile width.
