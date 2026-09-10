---
name: kernel-design
description: Implement or optimize a GPU kernel using existing references, targeted correctness checks and measured end-to-end benefit. Use for CUDA/TileLang kernels and fusions; follow docs/optimization.md for the enclosing model workflow.
---

# Kernel design

Follow [the optimization workflow](../../../docs/optimization.md). Start from its
identified bottleneck and reuse existing implementations/references where possible.
Record the proposed change and expected benefit in the experiment log; no separate
contract, approval receipt, draft-to-plan ceremony or mandatory full gate is needed.

Use [reference guidance](references/reference-tiers.md) and
[numerical tolerances](references/parity.md) for the affected computation. Extend
an existing check instead of creating duplicate references. A fusion must preserve
the required numerical behavior; add decomposition only when it helps diagnose it.
Do not edit the source a currently running compile/profile job is reading.

## Handoffs — this skill sequences, others own

Use whole-forward profiling first, then inspect selected kernels as described
in the default workflow. NCU is optional when counters can resolve a question.

| Need | Go to |
|---|---|
| a machine number, a floor, "is this target reachable" | `hardware-unit-test` — the measured table under the hardware axis; the ceiling cites tags, the roofline cites `spec.py`, neither is a target |
| per-kernel timing, comparing two implementations | `benchmark-kernel` |
| why a candidate is slow — the kernel-level profile: an ncu pass, capture plan through `REPORT.md`, read into a named bottleneck | `ncu-report` |
| where the pipeline's time goes — the stream-level profile: stage / launch / graph timeline with the torch profiler or `nsys`; also the capture runner behind an ncu pass | `gpu-profiler-analysis` |
| choosing the next optimization move, or what a mechanism costs and how it fails | `kernel-wiki` — `scripts/query.py --symptom <s>` from the report's stall reason, then the pattern page's candidate techniques |
| how a sm90 mechanism is actually spelled | `kernel-wiki` — the sm90 templates bundle (`artifacts/kernels/sm90-templates/variants`, `queries/by-template.md`); compilable, graded skeletons |
