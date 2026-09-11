---
id: technique-same-process-aba
title: "Historical A/B/A measurement notes"
type: technique
architectures: [sm90]
tags: [measurement, ablation, persistent-kernel, cache-policy]
confidence: measured
reproducibility: snippet
prerequisites: [hw-pdl-gdc]
related: [pattern-isolated-timer-overstates-fusion, pattern-persistent-kernel-timing-artifacts, pattern-one-sided-gradient, pattern-stacked-floors, technique-pdl-placement]
sources: [doc-benchmark-kernel, doc-hardware-unit-test, note-2026-09-06-optimization-campaign-plan, note-2026-09-07-pi0-expert-cuda-chain]
evidence_basis:
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
  - evidence_type: reproduction
    source_id: note-2026-09-07-pi0-expert-cuda-chain
---

# Historical A/B/A measurements

This page records older experiment evidence. The current
[optimization workflow](../../../../../docs/optimization.md) owns timing:
independent processes, first capture, a graph fixed to its stream, and median
with raw samples. A/B/A is an optional drift diagnostic, not a required model loop.
The old unpinned-clock observations do not establish a universal noise threshold
or justify choosing min over median.

For kernel comparisons, match shapes, layout and cache state; judge a fusion at
its full call-site boundary. Persistent kernels and dependent launches can have
overlapping trace durations, so profiler sums do not substitute for latency.
Reproduce an anomalous gain with the deployed model before accepting it.

```bash
python -m benchmarks latency --target h100/pi05 --plan shipped --out artifacts/current.json
python -m benchmarks kernels --target h100/pi05 --site action_expert_attention --timer cudagraph
```
