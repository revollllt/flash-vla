---
id: doc-benchmark-kernel
title: "benchmark-kernel: per-kernel timing method for this repository"
url: ../../../benchmark-kernel/SKILL.md
source_category: official-doc
architectures: [sm90]
tags: [gemm, attention]
retrieved_at: 2026-09-07
---

# benchmark-kernel

The sibling skill owns per-kernel GPU timing: CUPTI hardware timestamps by
default, CUDA-graph replay with rotating cold buffers when launch overhead
must be amortized, CUDA events for coarse checks. It states the rules the
wiki's measurement pages rely on: compare plans same-process and interleaved
(A/B/A), read `min` rather than `median` under unpinned clocks, name the node
and toolchain, and treat sub-microsecond cross-job deltas as noise.

The end-to-end regime, a captured graph replayed and read out of a trace, is
`python -m benchmarks latency` and the `eval.gate` verdict; the skill says
when a per-kernel number decides and when only the in-graph number does.
