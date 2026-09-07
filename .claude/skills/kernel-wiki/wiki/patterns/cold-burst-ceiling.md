---
id: pattern-cold-burst-ceiling
title: "A cold weight stream is bounded by the burst curve, not by steady-state DRAM"
type: pattern
architectures: [sm90]
tags: [tma, weight-streaming, persistent-kernel, cache-policy]
symptoms: [dram-burst-ramp, weight-stream-below-floor, memory-bound]
candidate_techniques: [technique-prefetch-across-dependency, technique-pipeline-stages, hw-tma]
related: [pattern-memory-bound, pattern-stacked-floors, kernel-megakernel-forms]
sources: [doc-hardware-unit-test, note-2026-09-06-optimization-campaign-plan]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# A cold weight stream is bounded by the burst curve, not by steady-state DRAM

## Symptom

A weight-streaming phase inside a persistent kernel reads a cold set of about
10-30 MB and reaches only about 40-60% of the steady-state DRAM constant.
Ring depth, TMA box size and CTA count are the obvious levers and none of
them moves it.

## Likely Causes

1. **The phase is being divided by the wrong constant.** A single cold
   stream's delivered rate climbs over its first ~100 MB; a phase-sized burst
   averages about 60% of steady state on its own. Divide by the
   size-dependent cold-burst curve [tma.bw.dev.burst], not by
   [tma.bw.dev.dram]. A phase already at about 90% of the burst ceiling has
   no geometry lever left.
2. **Each cold burst pays its own ramp.** Ring depth only hides latency once
   the stream is flowing, and box or CTA count only matter below the issue
   knee. The ramp is per stream, so the cure is fewer, longer, or pre-warmed
   streams.
3. **The stream is not on the critical path at all.** In a kernel whose math
   and dependency structure already hide most of the stream, the curve
   overstates the prize.

## Candidate Techniques

| Technique | Effect |
|---|---|
| Warmth | Have the set already in L2 when the phase starts: prefetch it during an earlier phase that has DRAM slack. Only a real TMA load leaves a set resident [tma.bw.dev.burst.warm]; the L2 prefetch instruction warms it half as well, and a plain SM read not at all |
| Continuity | One long stream across phase boundaries instead of a fresh burst per phase |
| [Prefetch across the dependency](../techniques/prefetch-across-dependency.md) | Issue the weight frames before the dependency wait so the burst starts earlier |
| [Pipeline stages](../techniques/pipeline-stages.md) | Only below the issue knee; do not expect depth to move a burst-bound phase |

## Diagnosis Checklist

```
1. Bound the stream's exposed share first: pin every stage to one K tile so
   all but the first read hits L2 (numerics invalid, timing valid). That is
   the real ceiling, and it has come out at a fifth of what the curve
   suggested. One job retires warming, prefetch, retiling and continuity
   at once.
2. Divide the phase's bytes by [tma.bw.dev.burst] at the phase's size, not
   by the steady-state constant.
3. Measure the target phase's DRAM slack before prefetching into it;
   prefetching into a phase at its own burst ceiling makes both worse.
4. Gate warmth on an end-to-end A/B, not on the probe: a set that survives
   synthetic filler traffic can still be evicted by real inter-launch
   traffic, and an isolated 2.5x warmth win has measured null-to-negative
   end to end.
```

## Caveats

The prefetch also collides with whatever phase issues it. Gate the L2
residency across the intervening traffic with a decisive pair before
touching the kernel.
