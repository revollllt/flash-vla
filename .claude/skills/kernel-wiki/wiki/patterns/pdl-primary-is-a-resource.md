---
id: pattern-pdl-primary-is-a-resource
title: "On a PDL chain, the kernel in front of you is a resource"
type: pattern
architectures: [sm90]
tags: [pdl, kernel-fusion, persistent-kernel, fusion-pricing]
symptoms: [pdl-chain-regression, fusion-regression, launch-bound]
candidate_techniques: [technique-pdl-placement, technique-producer-fusion-pdl, hw-pdl-gdc]
related: [pattern-fusion-latency-chain, pattern-isolated-timer-overstates-fusion]
sources: [doc-cuda-programming-guide-pdl, note-2026-09-06-optimization-campaign-plan]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# On a PDL chain, the kernel in front of you is a resource

## Symptom

A chain of dependent launches runs under programmatic dependent launch. One
link is a tiny launch-bound kernel (a norm factor, a counter reset, a
combine) and folding it into its consumer looks like free money: one less
launch, one less grid ramp. The fold measures neutral or worse end to end.

## Likely Causes

PDL's value is ramp and prefetch overlap, which is a property of the pair,
not of either kernel. Under PDL the successor's grid ramps and prefetches its
weight ring under the primary's tail, so a 1-3 us kernel can be buying more
overlap than its own cost. Removing a launch changes who the primary is, and
the new pairing can be strictly worse: a small kernel with a wide,
early-triggering grid is the ideal primary, and a persistent loop is the
worst one.

Two corollaries: a persistent kernel is a bad primary (triggering at entry
makes the whole successor chain resident during the loop's last, most
latency-sensitive phase; triggering at completion gives the successor no ramp
at all; keep a cheap non-persistent kernel between them), and there is no
middle trigger (a programmatic launch waits for every CTA of the primary, so
any placement later than entry degenerates to the implicit completion
trigger).

## Candidate Techniques

| Technique | Effect |
|---|---|
| [PDL placement](../techniques/pdl-placement.md) | Price the fold as saved launch minus lost overlap; sweep the trigger of the new pair |
| [Producer fusion](../techniques/producer-fusion-pdl.md) | Fuse into the producer side of the chain rather than into the persistent consumer |

## Diagnosis Checklist

```
1. Measure the fold end to end on the whole chain, never per kernel.
2. Keep an upper-bound measurement: skip the kernel outright, accepting
   wrong numerics, so a fold is never built against a prize smaller than
   its own risk.
3. Ask what the small kernel does for its successor as a primary.
```

## Caveats

Fold anyway when the small kernel's output is on the consumer's critical
path in a way that forces a serialization the fold removes; then the win is
the dependency, not the launch.
