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
2. Write the model's end-to-end reference first:
   `src/flash_vla/models/<model>/reference.py`, the whole model -- vision
   encoder, backbone and the action expert's full denoising loop -- translated
   from upstream into plain torch, citing the source files and revision. Its
   parameters carry the official checkpoint's names below each part's prefix
   (`PREFIXES`, `models/official.py`), it imports nothing but torch, its
   model's `spec` and shared reference parts (`tests/test_layering.py`), it
   reproduces upstream's inference dtypes, and its module docstring declares
   every deliberate difference. Check its `official_schema` against the real
   checkpoint's names and shapes; it is the numerical reference every later
   optimization is checked against.
3. Express the graph over the runtime vocabulary. Model details go in
   `src/flash_vla/models/<model>/` -- `definition.py` (a `ModelDefinition`),
   `graph.py`, `ops.py` for call sites beyond the standard vocabulary, and
   `reference_view.py`, which pairs every declared stage output with the
   reference's -- and the Target in `hardware/<vendor>/<device>/<model>/target.py`
   only composes that model with the device's layout, backends and plans. The
   first backend can simply run the reference's own modules on the runner's
   weights (as GR00T's and LingBot's reference routes do). A shared runtime
   change needs a model-independent reason. Use existing operators before
   designing a new kernel.
4. Compare the Target with its reference (`python -m eval.model_reference`,
   on seeded random official-layout weights and on real ones), and load real
   assets to compare against upstream with the existing numerical
   requirements, descending to intermediate activations only where the
   outputs already differ. Establish a working benchmark of this workload;
   make no speedup claim against a mismatched checkpoint or execution mode.
5. Register the Target, assets and plans: `models/<model>/sources.py` resolves
   where the weights and fixture come from and their provenance
   (`runner_source`) and the official-layout weights the reference runs
   (`official_weights`), and one `TargetEntry` in `src/flash_vla/inference.py`
   names the Target, its sources and its reference view. Record the upstream
   source and revision with the commands that run and check it, then hand the
   identified bottlenecks to the [optimization workflow](../model-optimization/SKILL.md).

Adding a GPU to a model that already has a Target reuses that reference: write
a new `Target` value over the existing model definition, with that device's
backends, register it with the model's existing sources and reference view,
and verify it against the reference.
Do not re-derive the model, and do not import the other device's Target or
kernels (`tests/test_layering.py`).

## Example

> Add the upstream VLA model on H100 with its existing checkpoint and input
> fixture. Map its forward to the runtime, reuse available components, verify
> outputs against upstream, and provide the command that measures the new Target.

Use the existing [numerical tolerances](../../../eval/tolerances.py) for numerical
requirements and [runtime architecture](../../../ARCHITECTURE.md) for graph mapping.
