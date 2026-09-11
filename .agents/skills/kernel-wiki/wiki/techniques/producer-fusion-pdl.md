---
id: technique-producer-fusion-pdl
title: "Fuse the producer chain; overlap the consumer with PDL"
type: technique
architectures: [sm90]
tags: [pdl, griddepcontrol, producer-fusion, kernel-fusion, persistent-kernel]
confidence: measured
reproducibility: snippet
prerequisites: [hw-pdl-gdc]
related: [technique-pdl-placement, pattern-fusion-latency-chain, pattern-layout-production-cost, pattern-pdl-primary-is-a-resource]
sources: [doc-hardware-unit-test, doc-ptx-isa-sm90, doc-kernel-design-templates, note-2026-09-06-optimization-campaign-plan]
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-hardware-unit-test
  - evidence_type: benchmark
    source_id: note-2026-09-06-optimization-campaign-plan
version_sensitive:
  id: vs-tilelang-0.1-pdl-attribute
---

# Fuse the producer chain; overlap the consumer with PDL

A chain of small kernels sits between two heavy stages: an epilogue, a norm,
a layout producer, a counter reset, then a persistent consumer. Every
boundary pays a launch ramp [launch.lat.dev.ramp]; a grid barrier costs
[coop.lat.dev.sync]; and a relaunch costs about 1.3x a grid sync
[coop.ratio.dev.relaunch], so neither "more kernels" nor "one cooperative
kernel" wins by itself unless the total count drops.

Fuse the producer chain into one cooperative kernel with the grid sync
inside; fold the consumer's counter and state reset into the producer (the
graph must then not carry a standalone reset kernel); trigger PDL after the
sync; let the persistent consumer wait at kernel entry. An early trigger is
memory-safe: PTX `griddepcontrol.wait` guarantees the prerequisite grid
completed and its memory is visible, while `.launch_dependents` only
controls when the dependent may be scheduled. The overlap won is the
consumer's dependency-free preamble, its weight prefetch, running under the
producer's tail; that overlap only exists if the producer triggers early, and
correctness is carried entirely by the consumer's wait.

## Source-backed fragment

The two intrinsics and what each promises, from `sm90_common.cuh` in
`doc-kernel-design-templates`:

```cpp
//   cudaGridDependencySynchronize()           -> griddepcontrol.wait  ::: "memory"
//   cudaTriggerProgrammaticLaunchCompletion() -> griddepcontrol.launch_dependents ::: (none)

__device__ __forceinline__ void pdl_wait() { cudaGridDependencySynchronize(); }

__device__ __forceinline__ void pdl_trigger() {
  cudaTriggerProgrammaticLaunchCompletion();
}
```

## Caveats

- Trigger and wait deploy as a chain, never per kernel in isolation
  (`technique-pdl-placement`).
- A cooperative launch needs a fail-fast residency check.
- Version-sensitive (TileLang 0.1.x): the PDL launch attribute is driven by
  the presence of `pdl_sync` in the kernel body, `__ldg` is rejected in such
  kernels, and `__restrict__` is silently dropped from their parameters;
  measure before assuming it free.
- PDL survives CUDA-graph capture as a programmatic dependency edge.
