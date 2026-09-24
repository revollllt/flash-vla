# Architecture

Flash-VLA specializes inference for a fixed model workload and GPU.

```text
Target   = hardware × inference-compatible architecture × shape profile
Workload = Target × execution policy
```

A model definition owns the graph, shapes, buffers and host work; a Target
composes it with one device's layout choices, backends and kernel routing. Execution policy
covers precision/quantization and cache behavior. Checkpoint values and input
fixtures are measurement context; changing them does not redefine architecture.
Plan and source revision identify the implementation being compared.

Quantization is a build-time choice among the Target's approved recipes
(`Target.quantization`, `quantization=<recipe>`). A recipe fixes the math and the
call sites it quantizes, routes those call sites to its kernels in the shipped
plan and to its fake-quant reference in the reference plan, and is recorded in
the identity's execution variant, so each recipe is its own workload. The
runner rejects a plan that runs a recipe's call sites on other backends or a
recipe's backends anywhere else.

## Components and dependencies

```text
models/<model>/               model semantics: graph, call sites, shapes, host
                              work, checkpoint loading, weight and fixture sources
runtime/                      graph API, Target composition, buffers, execution,
                              plan binding
hardware/<vendor>/<device>/   Targets and reusable component kernels
inference.py                  Target names and runner construction by name
provenance.py, assets.py      provenance values; logical asset IDs to local paths
measurement/                  timing loops and statistics, kernel timing,
                              attribution, environment and measurement context,
                              source-checkout qualification, result rendering
eval/, benchmarks/            accuracy and latency consumers
tools/, tests/                diagnostics and engineering checks
lab/                          experiments
```

- Models have no hardware dependencies. Runtime imports no model, backend or Target.
- `inference.py` names every Target and imports none until one is asked for;
  `provenance.py`, `assets.py` and `hardware/roofline.py` import nothing of
  the package.
- A Target combines one model, runtime and device component packages. It owns the
  layout the device's kernels see, plans and routing; components own reusable
  kernel implementations and may read a model's spec constants and reference math.
- A Target never imports another Target, and one device never imports another
  device's code: the same model on two devices is two Targets over one model
  definition. Shared expert builders live in `gemma_expert`; each Target keeps
  its own JIT registry and compiler settings.
- `tests/test_layering.py` checks these rules on the source.
- CUDA tile primitives and TileLang JIT conventions are vendor-level utilities, and so
  are components that serve a whole architecture family rather than one device: the
  Blackwell (sm_100+) block-scaled quantize ops live in `hardware/nvidia/quant_ops`.
- Production source imports no `measurement`, `benchmarks`, `eval`, `tools`,
  `tests` or `lab` modules. Experiments may import production code; deployment
  never imports experiments.
- `measurement` is the one layer every harness measures through: it imports
  production code and no harness, and `benchmarks` and `tools` never import each
  other. What the floor model reads of a device is that device's
  `spec.ROOFLINE` (`hardware/roofline.py`), found from the identity's hardware
  axis (`hardware.nvidia.HARDWARE_SPECS`).

## Forward execution

```text
images → vision_encoder → [host work] → llm_backbone → action_expert → actions
```

The explicit graph names inputs, outputs, weights and call sites. `ModelRunner`
materializes that graph and binds each call site to the selected plan. Model
shape and control-flow differences belong to the model definition rather than
branches inside the runner; a device's layout needs (row padding, which call
sites take the prefix mask) are parameters the Target passes to the model. Shared kernels accept the arguments and layout of their
call-site interface.

A host slot can overlap with the preceding GPU segment when dependencies allow.
Input copies, host work, graph replay and the final synchronization are part of
deployed inference. Model loading and capture are setup work.

## Execution invariants

- A CUDA graph captures and replays on one dedicated stream; caller-stream input
  and output dependencies are preserved. Measurement protocol is owned by the
  [optimization workflow](.agents/skills/model-optimization/SKILL.md).
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

A model is a `runtime.vla.ModelDefinition` subclass in
`models/<model>/definition.py`: identity, configuration, shape numbers, weight
schema, forward inputs and stage outputs, the extension ops its graph uses,
host slots, and `build`, which writes the graph (`models/<model>/graph.py`)
against the op vocabulary. A Target is a `runtime.vla.Target` value in
`hardware/<vendor>/<device>/<model>/target.py`: the model object with its
layout, a backend registry, two plans (`shipped` and `reference`), quantization
recipes, measured ceilings and the logical IDs of its assets. Candidate
plans stay under `lab/plans/`. OpenPI loading/conversion belongs to
`models/pi0/openpi.py` and `models/pi05/openpi.py`; evaluation adds official
forward adapters.

A runner is built from a Target and a `runtime.runner.RunnerSource`: the
checkpoint, its `WeightsProvenance`, the input fixture's `FixtureProvenance`,
local assets and the model configuration. `models/<model>/sources.py` resolves
one from construction options (`runner_source`), with no hardware of its own,
so every Target of a model shares it. `src/flash_vla/inference.py` maps Target
names to the Target and its model's sources and is the one place a named
Target's runner is constructed (`build`, `declare`, `build_runner`,
`get_target`); accuracy, latency and profiling tools all consume it. The
runner receives its provenance at construction and never changes it;
`implementation_source` names the checkout its backends came from when that is
not this one (`measurement.source_checkout`, which qualifies a LingBot backend
revision).

A backend is a `runtime.registry.Backend` value declared beside its
implementation: the call sites it implements, the factory that builds their
wrappers from the runner's workspace allocator (`runtime.workspace.Scratch`),
the route constraints its buffer contracts need, and the kernel-name contract
of its captured program given the call sites routed to it. A Target's registry
maps its own backend names to these values; a variant -- the same wrappers with
a launch attribute armed, one rung of LingBot's replacement ladder -- is a
`dataclasses.replace` of another backend, so no backend knows the name it is
registered under. Hand-written CUDA libraries are `hardware/nvidia/native`
`NativeLibrary` declarations: one compiler lookup (`FLASH_VLA_NVCC`, then
`CUDA_HOME`, then `PATH`) and one cache under `.cache/cuda_ext/`, keyed on the
compiler, the command and every source and declared header.

Machine configuration maps logical asset IDs to files (`flash_vla.assets`).
`FLASH_VLA_ASSETS` is a JSON mapping for file-backed construction; relative
asset paths resolve beside that JSON. Explicit path overrides retain their
separate checkpoint/fixture IDs. The runner receives its resolved assets
without mutating another runner's process environment.

Official references use `OPENPI_PYTHON` or `LINGBOT_PYTHON`. Pi0 official weights
also use `OPENPI_PI0_CHECKPOINT` and its explicit immutable ID. Missing reference
configuration is reported as unavailable rather than replaced with another
checkpoint. Where the upstream stack cannot be imported beside `flash_vla`, the
comparison runs in two interpreters instead of one: `eval.pi05.parity` captures
the official tensors with the fixture that produced them, `OPENPI_PI05_MODULE`
names the module the official forward came from, and the oracle records it.

## Optimization and evidence

[.agents/skills/model-optimization/SKILL.md](.agents/skills/model-optimization/SKILL.md) owns the two optimization loops and
links the relevant skills. Hardware rooflines and measured primitive limits
guide hypotheses; deployed end-to-end measurements decide whether a change helps.

Current reports use Identity v3. Existing reports and traces remain readable.
[Result tools](measurement/results/README.md) render saved traces independently of inference;
the retired Campaign controller is no longer part of the project.

[Agent Notes](.agents/notes/README.md) explain durable decisions. Historical notes
and plans are context, not additional requirements for the current workflow.
