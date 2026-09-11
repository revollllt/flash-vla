---
id: pattern-one-sided-gradient
title: "Price the direction you intend to move"
type: pattern
architectures: [sm90]
tags: [measurement, ablation, pipeline-stages]
symptoms: [null-ablation, one-sided-gradient]
candidate_techniques: [technique-same-process-aba, technique-pipeline-stages]
related: [pattern-stacked-floors, pattern-persistent-kernel-timing-artifacts]
sources: [doc-hardware-unit-test, note-2026-09-06-optimization-campaign-plan]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# Price the direction you intend to move

## Symptom

An ablation measures what removing a resource costs, one less ring stage,
one fewer warp, a smaller tile, and the number is large. It is tempting to
read the gradient backwards and expect the same magnitude from adding one.

## Likely Causes

A pipeline resource sized at or just past its knee has a one-sided gradient:
taking one away falls off the cliff, adding one buys nothing, and the two
numbers can differ by an order of magnitude. Knees are where a resource
stops being the constraint. Below the knee the system is starved and every
unit is worth its full latency; at or above it the next unit is dead weight,
and the cost of managing it (a protocol, a barrier, a frame that must be
tracked) is charged anyway.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Same-process A/B/A](../techniques/same-process-aba.md) | Measure the direction you intend to ship, in one job |
| [Pipeline stages](../techniques/pipeline-stages.md) | The stage count is chosen from the measured knee [tma.stages.warp.knee], [wgmma.stages.wg.knee], not from the downward ablation |

## Diagnosis Checklist

```
1. Measure the upward direction, not the downward one.
2. Where the resource must be paid for by shrinking something else, build
   the unpaid version first, even with invalid numerics, so a null cannot
   be blamed on the payment.
3. A cheap upward probe can retire an expensive plan: one bound row costs
   a single job.
```
