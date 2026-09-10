---
name: target-onboarding
description: Add a model or model/hardware Target that is not yet registered in flash-vla. Use for model bring-up and compatibility; existing Target optimization belongs to docs/optimization.md.
---

# New Target bring-up

Own the boundary from an upstream model to a runnable Target. The kernel inner
loop begins after that model's semantics and correctness reference are understood.

1. Read [architecture](../../../ARCHITECTURE.md) and the upstream forward,
   checkpoint format and preprocessing. State the intended shapes, precision,
   inputs/outputs and unresolved compatibility questions.
2. Reuse runtime vocabulary and existing components to express the graph. Model
   details stay in the Target; a shared runtime change needs a model-independent
   reason. Use existing operators before designing a new kernel.
3. Load real assets and compare against the upstream reference with the existing
   numerical requirements. Establish a working benchmark of this workload;
   make no speedup claim against a mismatched checkpoint or execution mode.
4. Register the Target, assets and plans, then hand the identified bottlenecks to
   the [optimization workflow](../../../docs/optimization.md).

## Example

> Add the upstream VLA model on H100 with its existing checkpoint and input
> fixture. Map its forward to the runtime, reuse available components, verify
> outputs against upstream, and provide the command that measures the new Target.

Use the existing [numerical tolerances](../../../eval/tolerances.py) for numerical
requirements and [runtime architecture](../../../ARCHITECTURE.md) for graph mapping.
The [recorded onboarding procedure](references/workflow.md) and its schema apply
when a legacy `lab.onboarding` ledger/handoff is explicitly requested; creating
that ledger is not a prerequisite for ordinary Target work.
