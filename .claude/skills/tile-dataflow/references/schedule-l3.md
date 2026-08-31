# L3 — the schedule, and why a fused kernel cannot skip it

SKILL.md's four-levels table defines L3 as *one stage seen as three engines
running concurrently*. This file is why that level is not optional in a fused
kernel, the one rule that catches most of its defects, and where the cycle
numbers may and may not come from. Read it before filling a spec's `## Loop
nest` L3 block or the `concurrency` check.

## Why L3 is not optional in a fused kernel

L1 and L2 describe *dependency*. A fused kernel's whole thesis is *concurrency* —
three engines busy at once — and a spec naming only which warp group holds which
*role* has **asserted** the overlap, not specified it.

Observed, not imagined. A spec said "the producer warp group owns the A-tile
transform, which overlaps it with the next stage's TMA". What got built was:

```
    wait TMA(s) -> transform s on CUDA cores -> release -> MMA(s)
                                                ^ TMA(s+1) issued HERE
```

The copy engine idles for the whole transform. Hoisting `TMA(s+1)` above it — one
line, invisible at L1/L2 and at the instruction level — was worth 0.8 us `[MEAS-A]` on a
14 us kernel, and was found by accident in Phase 2. In the same kernel the
transform was 24-27% of cycles, top stall `short scoreboard` 42.7%, 0.30 eligible
warps per scheduler. All L3 information; none of it survives into an instruction
list.

One rule covers most of these: **find what actually gates the next copy, because
it is later than it looks.** Three forms, all found by writing the timeline down:

- the issue sits below CUDA-core work that does not gate it — hoist it;
- the *release* sits below that work — `empty[s].arrive()` before the promote,
  not after, since the release is what the copy engine waits on;
- the buffer is not dead yet. wgmma reads smem **asynchronously**, so a stage is
  reusable only once `wgmma.wait_group` has *retired* the last instruction
  reading it — not when its barrier fired. In a seesaw that can be several
  steps after the data was consumed logically.

So every stage gets a timeline plus its ordering edges — three columns, or more
when warp groups must be kept out of phase. **An empty column is a bubble the
spec shows you before the kernel exists.**

## Where the cycle counts come from, since Phase 0 does not measure them

None of Phase 0's five measurements yields wgmma cycles per instruction,
CUDA-core issue rate, or TMA issue latency. The `hardware-unit-test` skill measures
some of these and marks the rest GAP -- check it before assuming a cycle. So:

- **the ordering edges and which column is empty are structural** — they follow
  from the dependencies and need no cycles at all. That is the part that catches
  bubbles, and it is the part worth arguing over;
- **cycle counts are `[I]` and must be marked so.** Published peaks are an
  acceptable source *for these* precisely because the conclusion does not rest on
  their absolute values;
- **the criterion is the ratio between columns, never the absolutes.** "The copy
  column is 527 against the tensor column's 512, so they are balanced within 3%"
  survives both numbers being 20% wrong. "The copy column is 527 cycles" does not.
  **But state the rate once, with its derivation, and check which columns it
  actually scales** — a column counted in *instructions* does not move with an
  FLOP/cycle rate, so halving that rate doubles its share while leaving the
  tensor column's ratio intact. A ratio is only robust between columns derived
  the same way.

On sm80 there is no separate copy engine: `cp.async` consumes LSU issue slots in
the same warps that compute, so the three-column model's premise — different
units, no contention — does not hold. Draw two columns and say so.

See `example-shape.md`'s L3 block for a filled-in timeline with its ordering
edges and bubble check.
