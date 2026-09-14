---
name: kernel-design
description: Design and implement one CUDA/TileLang kernel or fusion, validate correctness and local performance, and return a candidate to the model loop. Use after a hotspot has been identified.
---

# Kernel design

Own the kernel inner loop. The [model workflow](../model-optimization/SKILL.md)
owns hotspot selection, serial integration and deployed end-to-end measurements.

1. Read the selected call site's shapes, layout and numerical requirements,
   then search before writing: existing model and shared-component
   implementations, the vendored upstream libraries under `third_party/`
   (`cutlass`, `deepgemm`, `flashmla`), and [kernel-wiki](../kernel-wiki/SKILL.md)
   for a technique that already has a page. A conclusion that it has to be
   written from scratch states what was searched.
2. State one hypothesis and expected recoverable time. Consult
   [hardware-unit-test](../hardware-unit-test/SKILL.md) when a machine limit matters.
3. Implement the smallest change that tests it, against the measured bottleneck:
   tile shape, data layout, tensor-core utilization, memory access, pipeline
   overlap or explicit fusion. Fusions — normalization, activation, residual,
   epilogue — are hand-written CUDA or TileLang, to remove intermediate tensors
   and repeated traversals. GEMMs and GEMM-shaped mainloops go through
   CUTLASS/CuTe rather than being written from scratch: a vendored collective
   carries schedules such as stream-K and persistent tiling that are not worth
   re-deriving, and it is the baseline a hand-written one has to beat.
   Compiler-generated paths, `torch.compile` included, are references, not a
   delivery path.
4. Use existing [references](references/reference-tiers.md) and
   [tolerances](references/parity.md) for correctness, then
   [benchmark-kernel](../benchmark-kernel/SKILL.md) for local gain.
   Use [ncu-report](../ncu-report/SKILL.md) only when counters can resolve a question.
5. Return a correct, faster candidate with its diff, command and result. Failed or
   uncertain candidates stay in this loop. Distinct kernels can use parallel
   agents in separate worktrees; model integration remains serial.

## Example

> The model profile points to `action_expert_attention`. Check the existing
> model and shared-component implementations, then `third_party/`, before
> concluding anything has to be written — a scheduling symptom such as tail
> imbalance may already be a vendored collective's option. Then test one layout
> or fusion hypothesis, and return a candidate only after numerical checks and
> representative local timing pass.

A short experiment entry is enough. No separate contract, full-model gate or
publication step is required inside the kernel loop.
