---
name: target-onboarding
description: Add a model or model/hardware Target that is not yet registered in flash-vla. Use for model bring-up and compatibility; existing Target optimization belongs to the model-optimization skill.
---

# New Target bring-up

Own the boundary from an upstream model to a runnable Target. The kernel inner
loop begins after that model's semantics and correctness reference are understood.

1. Read [architecture](../../../ARCHITECTURE.md) and the upstream forward,
   checkpoint format and preprocessing. State the intended shapes, precision,
   inputs/outputs and unresolved compatibility questions.
2. Reuse runtime vocabulary and existing components to express the graph in
   plain PyTorch, preserving the upstream algorithm — this is the numerical
   reference every later optimization is checked against. Model details go in
   `src/flash_vla/models/<model>/` -- `definition.py` (a `ModelDefinition`),
   `graph.py`, and `ops.py` for call sites beyond the standard vocabulary --
   and the Target in `hardware/<vendor>/<device>/<model>/target.py` only
   composes that model with the device's layout, backends and plans. A shared
   runtime change needs a model-independent reason. Use existing operators
   before designing a new kernel.
3. Load real assets and compare against the upstream reference with the existing
   numerical requirements, descending to intermediate activations only where the
   outputs already differ. Establish a working benchmark of this workload;
   make no speedup claim against a mismatched checkpoint or execution mode.
4. Register the Target, assets and plans, record the upstream source and revision
   with the commands that run and check it, then hand the identified bottlenecks
   to the [optimization workflow](../model-optimization/SKILL.md).

Adding a GPU to a model that already has a Target reuses that reference: write
a new `Target` value over the existing model definition, with that device's
backends, and verify it runs. Do not re-derive the model, and do not import the
other device's Target or kernels (`tests/test_layering.py`).

## Example

> Add the upstream VLA model on H100 with its existing checkpoint and input
> fixture. Map its forward to the runtime, reuse available components, verify
> outputs against upstream, and provide the command that measures the new Target.

Use the existing [numerical tolerances](../../../eval/tolerances.py) for numerical
requirements and [runtime architecture](../../../ARCHITECTURE.md) for graph mapping.
