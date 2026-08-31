---
id: measuring-persistent-kernels
type: pattern
arch: sm90
tags: [benchmarking, cupti, persistent-kernel, ab-testing]
confidence: measured
---

# Measuring persistent kernels without fooling yourself

## Context

Benchmarking a persistent or cooperative candidate, or comparing
candidates on a cluster where clocks cannot be pinned. The harness itself
belongs to the `benchmark-kernel` skill; this entry records the traps
specific to persistent grids and unpinned clocks.

## Move

- **Profiler-replay medians lie on a persistent launch.** An L2-flush
  kernel injected before each replay delays co-residency of the
  persistent grid, so the sample distribution grows a heavy tail that the
  kernel does not have. Read event timing over rotating buffer sets, or
  single-replay records, for any persistent kernel.
- **With unpinned clocks, trust `min`, not `median`** — and only
  same-process, interleaved A/B/A comparisons. Name the node and
  toolchain; treat sub-microsecond cross-job deltas as noise.
- **One variable per experiment; screen before you build.** Price a
  candidate with design-time arithmetic first; one predicted below the
  noise floor is recorded as screened out, not run.
- **Isolated cold timing can overstate the in-graph win** — a kernel that
  looks ~2x better in isolation may deliver a fraction of that inside the
  captured pipeline, where launches overlap and caches differ. Gate on
  the pipeline profile, not the isolated number.
- **An anomalously good result that does not reproduce in the complete
  harness is not evidence** — rerun before believing it, in either
  direction.
