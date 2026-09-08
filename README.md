# flash-vla

Extreme-latency-optimized vision-language-action (VLA) inference, built as a
hardware/model co-design. Two targets ship on H100 SXM5. Pi0's single captured
graph became three — `vision_encoder`, `llm_backbone`, `action_expert`, one
CUDA graph each — so a stage can be timed, compared and injected into on its
own. Pi0.5 runs the same three stages with a host slot between the first two,
where the robot state is discretized and tokenized into the prompt while the
vision graph is still running. `forward()` copies three inputs into static
buffers, walks that program and returns the action chunk.

## Design philosophy

This is not a general-purpose inference framework. The atomic production unit
is a *target*:

> Target = hardware × model revision × shape profile × precision policy

A target owns its computation graph, its buffers, its per-call-site kernel
bindings, its fusion boundaries and its tuning results. Everything is
specialized to a fixed workload and measured inside the CUDA graph, at the real
shape, with cold weights — the only regime these numbers mean anything in.

**The graph is explicit data.** A target writes its forward pass as an ordered
list of ops, each naming its inputs, outputs and weights against a shared
vocabulary; the runner allocates, routes, executes, captures, attributes and
costs it without knowing the model. Bringing up a model is writing that graph.

**TileLang is the current main line, not the destination.** It is a
performance/efficiency trade-off for agile development. To reach the remaining
performance on a given device the project also uses hand-written CUDA. The
`backends/` layout is designed for exactly that: each backend exposes the same
call-site wrappers, and one graph can mix backends per call site through a
plan (`--plan`, or `plan=` on the runner).

## Install

```
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install -e .
```

`requirements.txt` pins the environment reports are measured in;
`pip install -e .` alone leaves torch unpinned so the CUDA build can match your
driver. TileLang is pinned to 0.1.11, which the kernels depend on for specific
lowering behaviour (see *Constraints that bite* in each TileLang backend's
`wrappers.py`).

Measured on an H100 SXM5, driver 610.43.02, python 3.12, torch 2.13.0.

## Use

```python
from flash_vla import ModelRunner
from flash_vla.hardware.nvidia.h100.pi0 import TARGET
from flash_vla.models.pi0 import random_checkpoint, random_checkpoint_revision

runner = ModelRunner(TARGET, random_checkpoint(),
                     model_revision=random_checkpoint_revision(0),
                     num_views=3, chunk_size=50)
actions = runner.forward(images=images, state=state, noise=noise)
```

`random_checkpoint()` fabricates weights so the pipeline can be run and timed
without a trained model. For real weights, pass a dict matching
`flash_vla.models.pi0.spec.weight_shapes()` and its immutable checkpoint ID as
`model_revision`. Benchmark factories derive a project-defined revision from
their deterministic random-fixture version and seed.

## Benchmarks and checks

```
python -m benchmarks latency  --target h100/pi05 --plan reference --plan shipped --plan reference
python -m benchmarks profile  --target h100/pi05 --plan shipped
python -m benchmarks kernels  --target h100/pi05
python -m benchmarks floor    --target h100/pi05
python -m eval.correctness --target h100/pi05 --steps 1 --layers 1
python -m eval.gate        --target h100/pi05 --candidate shipped --reference reference
python -m eval.smoke
```

The model is an input to every runner. `--plan` takes `shipped` (the one
deployed plan), `reference` (the correctness oracle route), or a path such as
`lab/plans/pi05-attn-ffn-cuda.json`. Everything except `eval.smoke` needs a GPU;
submit them on a Slurm cluster rather than running on a login node, since
TileLang compiles against the local device and re-reads the source at compile
time.

## Layout

| path | responsibility |
|---|---|
| `src/flash_vla/models/pi0/`, `models/pi05/` | hardware-independent checkpoint schema, weight fold, tokenizer |
| `src/flash_vla/runtime/` | op vocabulary, explicit graph, VLA template, `ModelRunner`, registry, identity |
| `src/flash_vla/runtime/cuda/` | static arena, graph capture and in-graph timing |
| `src/flash_vla/tuning/` | backend-agnostic config sweeps |
| `src/flash_vla/hardware/nvidia/cuda/tile/sm90/` | header-only SM90 tile primitives shared by every hand-written kernel |
| `src/flash_vla/hardware/nvidia/h100/pi0/` | H100/Pi0 Target: contract, graph, plans |
| `.../pi0/backends/` | its TileLang backend and the fused overlay the shipped plan routes to |
| `src/flash_vla/hardware/nvidia/h100/pi05/` | H100/Pi0.5 Target: contract, graph, host slot, plans |
| `.../pi05/backends/` | its TileLang and CUDA backends, the latter also with the PDL chain armed |
| `benchmarks/` | latency, profile, per-kernel and floor runners, plus the Target registry |
| `eval/` | acceptance registry, correctness, promotion gate, declaration smoke, baselines |
| `lab/` | the optimization workspace: candidate plans, kernel trials, their sbatch wrappers |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the boundary, dependency rules and
invariants.

## Measured

There are no latency numbers in this file. Every one lives in a report produced
by `python -m benchmarks latency`, stamped with the identity it was measured
under — hardware, immutable model revision, complete shape profile, precision
policy, resolved plan and engine revision. Workload comparison ignores the
last two and rejects any mismatch in the Target axes. The checkpoint producer
supplies `model_revision`; deterministic benchmark fixtures derive it from
their fixture version and seed. `engine_revision` is the full commit of a clean
checkout and is unresolved for dirty source. `eval/acceptance.py` is
the registry of what gates: the correctness checks and their tolerances, the
latency statistics and repetition policy, each Target's budget. `python -m
eval.gate` turns a run into one verdict.

## Requirements

H100 (Hopper WGMMA and TMA), TileLang 0.1.11, PyTorch with CUDA. Vision
attention stays in torch SDPA: full bidirectional attention over a long
sequence, where cuDNN is already at the roofline. The backbone's multi-query
attention has a hand-written CUDA kernel that the Pi0.5 shipped plan selects,
with the torch chain kept as its reference route; the action expert's
multi-query attention over the KV cache has both a TileLang and a CUDA
implementation.

`num_views == 2` is not supported (upstream's two-part vision branch).

## Provenance

Extracted from a larger H100 megakernel research repository, where this
pipeline was developed as a port of the Triton kernels in the realtime-vla Pi0
implementation. Every kernel was validated op-by-op against it before the
dependency was dropped; what remains in-tree is the shipped-vs-reference gate
(`python -m eval.correctness --target h100/pi0 --steps 1 --layers 1`).

Source commit of the parent repository: `a53bcf9`.

## License

MIT — see [LICENSE](LICENSE).
