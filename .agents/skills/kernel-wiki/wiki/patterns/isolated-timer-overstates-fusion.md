---
id: pattern-isolated-timer-overstates-fusion
title: "An isolated cold timer overstates a fusion's prize"
type: pattern
architectures: [sm90]
tags: [measurement, kernel-fusion, cache-policy]
symptoms: [isolated-vs-in-graph-mismatch, measurement-artifact, fusion-regression]
candidate_techniques: [technique-same-process-aba]
related: [pattern-persistent-kernel-timing-artifacts, pattern-fusion-latency-chain, technique-pdl-placement]
sources: [doc-benchmark-kernel, note-2026-09-06-optimization-campaign-plan, note-2026-09-07-gemma-backbone-cuda-backend, note-2026-09-07-pi0-expert-cuda-chain]
confidence: measured
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-07-gemma-backbone-cuda-backend
  - evidence_type: reproduction
    source_id: note-2026-09-07-pi0-expert-cuda-chain
---

# An isolated cold timer overstates a fusion's prize

## Symptom

A per-kernel benchmark with rotating cold inputs says the route you want to
replace costs X, your fused kernel costs less, and the fusion looks like a
win. In the captured graph the same replacement is neutral or negative.

## Likely Causes

The cold-input timer charges the incumbent a DRAM read that the real
pipeline never pays: its operand was just written by the kernel before it and
is L2-resident. Activations are produced in place, one kernel earlier, and
are the operand a fusion is usually trying to keep in registers or shared
memory; charging them at DRAM prices credits the fusion with bytes nobody was
paying. A library attention kernel measured about 11 us cold and about 8.5 us
in place, which was the entire margin the fusion was built to capture. The
same applies in reverse: a fused kernel that removes a node does not
necessarily remove that node's gap.

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Same-process A/B/A in the graph regime](../techniques/same-process-aba.md) | Measure the incumbent in graph context, replay the captured graph and read the position out of the trace, and use that as the baseline; compare whole-stage wall time per layer, not the sum of kernel durations |

## Diagnosis Checklist

```
1. Ask what the predecessor in the graph leaves in cache.
2. Measure the incumbent in place; expect it materially cheaper than cold.
3. Compare whole-stage wall time per layer for candidate and incumbent.
4. Keep the cold timer only for weight-dominated kernels and for ranking
   configs of one implementation, where the bias is common to every row.
```

## Caveats

The cold rotating-buffer regime exists to make weight streams honest, and it
does. Switch regimes the moment the decision is "fuse or not".
