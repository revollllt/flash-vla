# Architecture

This repository builds fixed-workload VLA inference targets for a specific GPU,
model revision, shape profile, and precision policy. Peak end-to-end action
latency on that fixed workload takes priority over a universal operator
abstraction.

```text
Target = hardware x model revision x shape profile x precision policy
```

A Target is a subclass of the VLA template (`src/flash_vla/runtime/vla.py`)
that declares its model contract and writes its computation graph as data
against the framework's op vocabulary (`runtime/ops.py`, `runtime/graph.py`):
three stages, `vision_encoder -> llm_backbone -> action_expert`, every op with
explicit inputs, outputs and weights, one shipped plan and one reference plan.
One runner (`runtime/runner.py`, `ModelRunner`) allocates the buffers the
graph declares, binds the plan through the Target's backend registry, executes
and captures each stage, attributes profiles and derives costs -- for every
Target, without model knowledge. An agent-driven
optimization plane produces and improves Targets from measured evidence; the
human defines accuracy requirements, the metric and framework conventions and
a budget, and the latency objective is derived from a floor model over
measured hardware constants.

The architecture is documented as a tree under
[`docs/architecture/`](docs/architecture/README.md):

| Document | Owns |
|---|---|
| [`README.md`](docs/architecture/README.md) | overview, vocabulary, status of each component |
| [`00-target.md`](docs/architecture/00-target.md) | the Target, identity, acceptance module and comparability |
| [`10-runtime.md`](docs/architecture/10-runtime.md) | the Static Inference Runtime and the engine protocol |
| [`20-target-layout.md`](docs/architecture/20-target-layout.md) | target composition, dependency direction, specialization rules |
| [`30-acceptance-criteria.md`](docs/architecture/30-acceptance-criteria.md) | what the human defines, the acceptance registry, stop condition |
| [`31-latency-evaluation.md`](docs/architecture/31-latency-evaluation.md) | end-to-end latency metrics and measurement discipline |
| [`32-correctness-evaluation.md`](docs/architecture/32-correctness-evaluation.md) | numerical correctness: oracles and comparison structure |
| [`33-latency-floor-model.md`](docs/architecture/33-latency-floor-model.md) | the derived latency objective and its validation |
| [`40-optimization-plane.md`](docs/architecture/40-optimization-plane.md) | the agent flywheel, skills, notes and promotion gate |

The documents under `docs/architecture/` predate the explicit graph and the
runner; where they describe engines, buffer plans or cost declarations, the
source above is authoritative (they are collapsed into this page in the next
change). Decisions and their alternatives live in
[Agent Notes](.agents/notes/README.md).
