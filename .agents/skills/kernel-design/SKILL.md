---
name: kernel-design
description: Design and implement one CUDA/TileLang kernel or fusion, validate correctness and local performance, and return a candidate to the model loop. Use after a hotspot has been identified.
---

# Kernel design

Own the kernel inner loop. The [model workflow](../../../docs/optimization.md)
owns hotspot selection, serial integration and deployed end-to-end measurements.

1. Read the selected call site's shapes, layout and numerical requirements.
   Reuse existing Pi0/Pi0.5 or shared-component implementations first.
2. State one hypothesis and expected recoverable time. Consult
   [kernel-wiki](../kernel-wiki/SKILL.md) for implementation patterns and
   [hardware-unit-test](../hardware-unit-test/SKILL.md) when a machine limit matters.
3. Implement a small CUDA/TileLang change: tiling, layout, memory reuse, pipeline
   overlap or explicit operator fusion. Existing compiler-generated paths may be
   references; new fusion work uses a manually written kernel.
4. Use existing [references](references/reference-tiers.md) and
   [tolerances](references/parity.md) for correctness, then
   [benchmark-kernel](../benchmark-kernel/SKILL.md) for local gain.
   Use [ncu-report](../ncu-report/SKILL.md) only when counters can resolve a question.
5. Return a correct, faster candidate with its diff, command and result. Failed or
   uncertain candidates stay in this loop. Distinct kernels can use parallel
   agents in separate worktrees; model integration remains serial.

## Example

> The model profile points to `action_expert_attention`. Check the existing
> Pi0/Pi0.5 implementations, test one layout or fusion hypothesis, and return a
> candidate only after numerical checks and representative local timing pass.

A short experiment entry is enough. No separate contract, full-model gate or
publication step is required inside the kernel loop.
