# Architecture

Flash-VLA specializes inference for a fixed model workload and GPU.

```text
Target   = hardware × inference-compatible architecture × shape profile
Workload = Target × execution policy
```

A Target owns its graph, shapes, buffers and kernel routing. Execution policy
covers precision/quantization and cache behavior. Checkpoint values and input
fixtures are measurement context; changing them does not redefine architecture.
Plan and source revision identify the implementation being compared.

## Components and dependencies

```text
models/                       model semantics and checkpoint loading
runtime/                      graph, buffers, execution and plan binding
hardware/<vendor>/<device>/   Targets and reusable component kernels
eval/, benchmarks/           accuracy and latency consumers
tools/, tests/                diagnostics and engineering checks
lab/                          experiments and optional historical tooling
```

- Models have no hardware dependencies. Runtime imports no model, backend or Target.
- A Target combines models, runtime and device component packages. It owns shapes,
  plans and routing; components own reusable kernel implementations.
- A Target never imports another Target's kernels. Shared expert builders live
  in `gemma_expert`; each Target keeps its own JIT registry and compiler settings.
- CUDA tile primitives and TileLang JIT conventions are vendor-level utilities.
- Production source imports no `benchmarks`, `eval`, `tools`, `tests` or `lab` modules. Experiments
  may import production code; deployment never imports experiments.

## Forward execution

```text
images → vision_encoder → [host work] → llm_backbone → action_expert → actions
```

The explicit graph names inputs, outputs, weights and call sites. `ModelRunner`
materializes that graph and binds each call site to the selected plan. Model
shape and control-flow differences belong to the Target rather than branches
inside the runner. Shared kernels accept the arguments and layout of their
call-site interface.

A host slot can overlap with the preceding GPU segment when dependencies allow.
Input copies, host work, graph replay and the final synchronization are part of
deployed inference. Model loading and capture are setup work.

## Execution invariants

- A CUDA graph captures and replays on one dedicated stream; caller-stream input
  and output dependencies are preserved. Measurement protocol is owned by the
  [optimization workflow](docs/optimization.md).
- Resolve plans and route constraints before capture. The static arena owns
  buffer addresses, padding and aliases; no allocation follows its freeze.
- Operators write to their declared output buffers. The graph, not return-value
  inference, describes the data dependencies.
- Check checkpoint shapes and semantic compatibility before allocation. Targets
  own architecture metadata; checkpoint producers supply weight provenance.
- Compare performance only under matching workload and measurement conditions.
  A different checkpoint, fixture, GPU or execution policy is a different
  comparison context, not an optimization gain.

## Targets, plans and assets

A Target supplies model configuration and checkpoint mapping, a graph-building
pipeline, a backend registry and two plans: `shipped` and `reference`. Candidate
plans stay under `lab/plans/`. Factories in `src/flash_vla/inference.py` handle model
construction. OpenPI loading/conversion belongs to `models/pi0/openpi.py` and
`models/pi05/openpi.py`; evaluation adds official forward adapters. Accuracy,
latency and profiling tools all consume this same inference entrypoint.

Machine configuration maps logical asset IDs to files. `FLASH_VLA_ASSETS` is a
JSON mapping for file-backed construction; relative asset paths resolve beside
that JSON. Explicit path overrides retain their separate checkpoint/fixture IDs.
The runner receives its resolved assets without mutating another runner's
process environment.

Official references use `OPENPI_PYTHON` or `LINGBOT_PYTHON`. Pi0 official weights
also use `OPENPI_PI0_CHECKPOINT` and its explicit immutable ID. Missing reference
configuration is reported as unavailable rather than replaced with another
checkpoint.

## Optimization and evidence

[docs/optimization.md](docs/optimization.md) owns the two optimization loops and
links the relevant skills. Hardware rooflines and measured primitive limits
guide hypotheses; deployed end-to-end measurements decide whether a change helps.

Current reports use Identity v3. Legacy schemas remain readable for existing
results; migration and Campaign qualification belong to the optional legacy
tools. Ordinary benchmarks do not require those workflows.

[Agent Notes](.agents/notes/README.md) explain durable decisions. Historical notes
and plans are context, not additional requirements for the current workflow.
