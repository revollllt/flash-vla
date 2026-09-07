---
id: technique-same-process-aba
title: "Same-process A/B/A, min under unpinned clocks, and the in-graph regime"
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

# Same-process A/B/A, min under unpinned clocks, and the in-graph regime

The harness belongs to `doc-benchmark-kernel`; this page records the rules
that decide what a number means on a cluster where clocks cannot be pinned.

- **Compare same-process, interleaved A/B/A**, and with unpinned clocks read
  `min`, not `median`. Name the node and toolchain; treat sub-microsecond
  cross-job deltas as noise.
- **One variable per experiment; screen before you build.** Price a
  candidate with design-time arithmetic against the measured constants
  first; one predicted below the noise floor is recorded as screened out,
  not run.
- **Profiler-replay medians lie on a persistent launch.** An L2-flush kernel
  injected before each replay delays co-residency of the persistent grid, so
  the sample distribution grows a heavy tail that the kernel does not have.
  Read event timing over rotating buffer sets, or single-replay records, for
  any persistent kernel.
- **Switch to the in-graph regime the moment the decision is "fuse or
  not".** Keep the cold rotating-buffer timer for weight-dominated kernels
  and for ranking configs of one implementation, where the bias is common to
  every row; a fusion is judged on whole-stage wall time per layer inside
  the captured graph.
- **An anomalously good result that does not reproduce in the complete
  harness is not evidence**; rerun before believing it, in either direction.

## Commands

The in-graph verdict and the per-site kernel timing this repository uses:

```bash
python -m eval.gate --candidate <plan>          # A/B/A in the captured graph; pass | fail | blocked
python -m benchmarks kernels --site <site>      # per-call-site kernel timing, CUPTI by default
```

A `blocked` verdict is rerun, never read as a pass.

## Caveats

The two rates compared in `pattern-stacked-floors` must share job, node,
clock and geometry; that comparison turns on a few percent. A cheap upward
probe can retire an expensive plan (`pattern-one-sided-gradient`): one bound
row costs a single job.
