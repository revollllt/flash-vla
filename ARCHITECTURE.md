# Architecture

This repository builds fixed-workload VLA inference targets. Peak end-to-end
action latency on one fixed workload takes priority over a universal operator
abstraction, so the atomic unit of production is neither a model nor a kernel:

```text
Target = hardware x model revision x shape profile x precision policy
```

Everything specialized — buffers, fusion boundaries, tile configurations,
kernels — belongs to one Target. Two measurements describe the same thing only
when they carry the same Target.

![flash-vla architecture](docs/arch.png)

## The boundary

`src/flash_vla/runtime/` owns what is invariant across every Target: the op
vocabulary, the explicit computation graph and the API that builds it, the VLA
template, the one `ModelRunner`, the static arena and graph capture, plan
binding, and the identity every report carries. It knows no model, no backend
and no device.

The Target owns everything else: its model contract, its graph, its backends
and their kernels, and the plans that route call sites to them. A
specialization that helps one Target and not another lives in that Target.

## The template

Inference is three stages, in this order:

```text
images -> vision_encoder -> [host slot] -> llm_backbone -> action_expert -> actions
```

`vision_encoder` embeds the camera views, `llm_backbone` builds the prefix KV
cache, `action_expert` denoises the action chunk over that cache. A host slot
is optional host work between two stages, ordered by the program and hidden
behind the preceding stage's replay.

Onboarding a model means rewriting its original forward pass as an explicit
graph over the op vocabulary: every op names its inputs, outputs and weights,
and Python loops only construct nodes. This is the relation vLLM's model files
have to their transformers originals — the rewrite need not be written the way
the original was, and its correctness is judged only by the precision gate
against the original implementation.

## Dependency direction

```text
runtime/                              -> nothing model-, backend- or Target-specific
models/<model>/                       -> nothing hardware-specific
hardware/<vendor>/<device>/<model>/   -> models/ + runtime/ + its own backends/
eval/, benchmarks/                    -> ModelRunner + the Target registry
                                         (benchmarks/targets.py)

src/ never imports eval/, benchmarks/ or lab/.
No Target imports another Target's kernels.
The deployment path never imports lab/.
```

## The runtime interface

| module | owns |
|---|---|
| `runtime/ops.py` | `OpSpec`: one call site's parameter order, outputs, weights, FLOP formula |
| `runtime/graph.py` | `Graph`, `BufRef`, `WeightRef`: the forward pass as data, and the builder API |
| `runtime/vla.py` | `VLA`, `Input`, `STAGE_OUTPUTS`: the template a Target subclasses |
| `runtime/runner.py` | `ModelRunner`: materialization, execution, capture, attribution, costs |
| `runtime/registry.py` | `Registry`: how a Target's backends are named, routed and built |
| `runtime/binding.py` | route constraints and their validation against a plan |
| `runtime/engine.py` | the `Engine` protocol every harness is written against |
| `runtime/cuda/` | `StaticArena` (fixed addresses), `Program` (warmup, freeze, capture, replay) |
| `runtime/identity.py` | `Identity`: the axes, shape numbers, plan and revision on every report |
| `runtime/cost.py` | `Cost`, `Invocation`, `Ceiling`, `SegmentCosts`: the minimal traffic and math a floor divides, and a call site's declared measured ceiling |

## Invariants

- A plan is resolved and validated against every route constraint before
  capture, never at the first replay.
- Nothing allocates after the workspace allocator freezes; a request warmup did
  not cover raises instead of allocating inside a capture.
- Every op writes its outputs in place through the parameters its spec names;
  return values are ignored.
- Buffers, their padding and their alias views are declared data, not a
  consequence of running the graph.
- Every report carries an `Identity`. Two numbers are comparable only when
  target, hardware, model, shape profile and precision policy match and the
  plan is stated.
- The floor model is guidance, never an objective. Its ceiling divides only by
  tagged measured constants of the hardware axis's `measured/` table, its
  roofline only by the axis's `spec.py` peaks, and the registry's stop
  condition reads the headroom between measured and ceiling.

## What a Target is

One `target.py` holding the model contract (configuration, shape numbers,
weight schema and loader), the shipped plan, the reference plan and the backend
registry; one `pipeline.py` whose `build` writes the graph; its `backends/`;
a factory entry in `benchmarks/targets.py`; and an acceptance entry in
`eval/acceptance.py`. There is no engine, buffer plan or cost table to write.

## Deployment configuration

Each Target carries exactly one shipped plan and one reference plan (its
correctness oracle route). Everything else — candidate plans, ablations,
per-kernel trials — lives in `lab/`, the optimization workspace, which is
tracked in git, may import the deployment path, and is never imported by it.

## Evaluation

`python -m benchmarks {latency,profile,kernels,floor}` take the Target as an
input, as do `python -m eval.correctness` and `python -m eval.gate`, which
turns the registry's checks and an A/B/A into one verdict. `python -m
eval.smoke` checks every Target's declarations on a login node, without a
device. `eval/acceptance.py` is the one registry of what the human defines:
accuracy requirements, framework conventions and each Target's budget. The
official-baseline tier is per model: `eval/pi05/reference.py`,
`eval/pi0/reference.py`.

Decisions, their alternatives and their evidence live in
[Agent Notes](.agents/notes/README.md).
