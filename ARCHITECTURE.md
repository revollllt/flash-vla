# Architecture

This repository builds fixed-workload VLA inference targets for a specific GPU,
model revision, shape profile, and precision policy. Peak end-to-end action
latency on that fixed workload takes priority over a universal operator
abstraction.

```text
Target = hardware x model revision x shape profile x precision policy
```

A Target owns its pipeline, buffer plan, call-site bindings, fusion boundaries,
kernels, tuning results, plans, parity scripts, benchmark cases and cost
declarations. A small runtime owns only what is invariant across Targets:
static addresses, scratch, graph segments and their lifecycle. An agent-driven
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

Read `20-target-layout.md` before changing a pipeline, backend or buffer plan;
it carries the dependency direction and the specialization rules. Decisions
and their alternatives live in [Agent Notes](.agents/notes/README.md).
