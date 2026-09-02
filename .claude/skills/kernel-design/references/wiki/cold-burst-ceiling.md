---
id: cold-burst-ceiling
type: pattern
arch: sm90
tags: [tma, dram, weight-stream, persistent, l2]
confidence: measured
---

# A phase that streams tens of MB cold is bounded by the burst curve, not by steady-state DRAM

## Context

A weight-streaming phase inside a persistent kernel reads a cold set of
~10-30 MB and reaches only ~40-60% of the steady-state DRAM constant. Ring
depth, TMA box size and CTA count are the obvious levers and none of them
moves it.

## Move

Divide by `[tma.bw.dev.burst]` (the size-dependent cold-burst curve), not
by `[tma.bw.dev.dram]`. A single cold stream's delivered rate climbs over
its first ~100 MB; a phase-sized burst averages ~60% of steady state on
its own, and a phase already at ~90% of the burst ceiling has no geometry
lever left. What does move it: **warmth** -- have the set already in L2
when the phase starts (prefetch it during an earlier phase that has DRAM
slack) -- or **continuity** -- one long stream across phase boundaries
instead of a fresh burst per phase.

## Why it works

Each cold burst pays its own ramp; ring depth only hides latency once the
stream is flowing, and box or CTA count only matter below the issue knee.
The ramp is per stream, so the cure is fewer, longer, or pre-warmed
streams.

## Caveats

Prefetching into a phase that is itself at its burst ceiling makes both
worse -- measure the target phase's DRAM slack first, and gate the L2
residency across the intervening traffic with a decisive pair before
touching the kernel.
