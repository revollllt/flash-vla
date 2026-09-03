---
id: price-the-direction
type: pattern
arch: sm90
tags: [methodology, pipeline, ring-depth, ablation]
confidence: measured
---

# Price the direction you intend to move

## Context

An ablation measures what removing a resource costs — one less ring stage,
one fewer warp, a smaller tile — and the number is large. It is tempting to
read the gradient backwards and expect the same magnitude from adding one.

## Move

Do not. Measure the direction you intend to ship. A pipeline resource sized
at or just past its knee has a **one-sided** gradient: taking one away falls
off the cliff, adding one buys nothing, and the two numbers can differ by an
order of magnitude. Where the resource must be paid for by shrinking
something else, build the *unpaid* version first — even if its numerics are
invalid — so a null result cannot be blamed on the payment.

## Why it works

Knees are where a resource stops being the constraint. Below the knee the
system is starved and every unit is worth its full latency; at or above it
the next unit is dead weight, and the cost of managing it (a protocol, a
barrier, a frame that must be tracked) is charged anyway.

## Caveats

The rule cuts the other way too: a cheap upward probe can retire an expensive
plan. One bound row costs a single job and settles what a protocol
implementation would take a lane to learn.
