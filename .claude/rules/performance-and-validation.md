---
paths:
  - "benchmarks/**/*.py"
  - "eval/**/*.py"
  - "tools/**/*.py"
  - "tests/**/*.py"
  - "src/flash_vla/**/*.py"
  - "src/flash_vla/**/*.cu"
  - "src/flash_vla/**/*.cuh"
---

# Performance and Validation

Follow [the optimization workflow](../../docs/optimization.md). Its top-down
profiling, independent first-capture timing, relevant correctness checks and
recording rules are the defaults. Optional legacy qualification policies do not
add mandatory Campaign, A/B/A, re-anchor or publication steps to that workflow.

Warm up compilation and allocations before capture; keep graph replay on its
owner stream. Use Torch Profiler/Nsight Systems for complete-forward timelines,
then selected-module attribution and Nsight Compute only where needed. Profiling
and uninstrumented latency measurements run in separate processes.
