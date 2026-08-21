---
paths:
  - "benchmarks/**/*.py"
  - "eval/**/*.py"
  - "src/flash_vla/**/*.py"
  - "src/flash_vla/**/*.cu"
  - "src/flash_vla/**/*.cuh"
---

# Performance and Validation

Performance claims require a reproducible baseline and an independent
correctness result. Prefer the cheapest tool that answers the question, and
state the tool in the report.

- **Use the production shape and execution regime.** Record device, driver,
  CUDA, Torch, compiler, git revision, and clock policy.

  ```text
  Good: H100 SXM / CUDA 13.1 / torch 2.x / CUDA Graph replay / shape B=...,
  with warmup=... and measured_iters=...
  Bad: "kernel is 20% faster" with no shape, timing backend, or revision.
  ```

- **Separate correctness and performance gates.** A faster result does not
  override a numerical mismatch. Run the relevant parity script before
  comparing latency.
- **Choose the timing tool by question.** CUDA Events (or the benchmark's
  graph timing) support latency claims; Torch Profiler and Nsight Systems explain
  the timeline; Nsight Compute explains selected kernel microarchitecture.
- **Warm up before capture.** Compilation, lazy initialization, and allocations
  must complete before CUDA Graph capture. A replay path must not allocate.
- **Treat profiler timings as diagnostic.** Nsight replay and profiler overhead
  can change scheduling; never substitute them for the benchmark baseline.
- **Keep artifacts reproducible.** Put Chrome JSON/JSON.GZ traces, reports, and
  native profiler outputs under the ignored artifact directory and record their
  metadata and command line.
