---
id: pattern-persistent-kernel-timing-artifacts
title: "Persistent-kernel timing looks absurd; cross-job deltas do not reproduce"
type: pattern
architectures: [sm90]
tags: [measurement, persistent-kernel]
symptoms: [measurement-artifact, heavy-tail-timing]
candidate_techniques: [technique-same-process-aba]
related: [pattern-isolated-timer-overstates-fusion, pattern-one-sided-gradient, pattern-low-sm-utilization]
sources: [doc-benchmark-kernel, note-2026-09-06-optimization-campaign-plan, note-2026-09-07-pi0-expert-cuda-chain]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
---

# Persistent-kernel timing looks absurd; cross-job deltas do not reproduce

## Symptom

A persistent or cooperative candidate shows a heavy-tailed sample
distribution under profiler replay, a sub-microsecond win that appears in one
job and not the next, or an isolated number about 2x better than what the
captured pipeline delivers.

## Likely Causes

1. **Replay injection.** An L2-flush kernel injected before each replay
   delays co-residency of the persistent grid; the tail is the harness, not
   the kernel.
2. **Unpinned clocks** across jobs; the median moves with the node.
3. **Regime mismatch.** In isolation launches do not overlap and caches
   differ from the captured pipeline.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Same-process A/B/A](../techniques/same-process-aba.md) | Event timing over rotating buffer sets or single-replay records for any persistent kernel; `min` not `median`; interleaved same-process comparison; gate on the pipeline profile, not the isolated number |

## Diagnosis Checklist

```
1. Name the node and toolchain; treat sub-microsecond cross-job deltas as
   noise.
2. Screen the candidate with design-time arithmetic before running it; a
   candidate predicted below the noise floor is recorded as screened out.
3. Rerun an anomalously good result in the complete harness before
   believing it, in either direction.
```
