# flash-vla

VLA inference specialized for a model, workload and GPU. Models describe the
forward graph; Targets compose a model with one GPU's kernels and choose among
them through a plan; shared components provide reusable CUDA/TileLang
implementations.

## Install

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install -e .
```

The pinned environment is for reproducing existing measurements. On another
CUDA stack, `pip install -e ".[pi05]"` leaves the PyTorch build selectable;
keep the environment fixed within a performance comparison. Machine access,
assets and local environment setup belong in user-directory skills.

## Run

Commands below run from the project root in the configured GPU environment.

| Target | Benchmark inputs |
|---|---|
| `h100/pi0` | Seeded synthetic weights and inputs |
| `h100/pi05` | Synthetic weights, or an explicit OpenPI checkpoint/config |
| `h100/lingbot_vla` | Real checkpoint and fixture, resolved through `FLASH_VLA_ASSETS`; or `synthetic=true`, seeded random weights and inputs needing no asset (its `reference` plan runs on any CUDA device) |
| `rtx5090/pi0` | Seeded synthetic weights and inputs |
| `rtx5090/pi05` | Synthetic weights, or a checkpoint OpenPI's converter wrote |
| `thor/pi05` | BF16 shared Torch route; synthetic weights or a converted OpenPI checkpoint; `robodojo` and `libero` workloads |
| `thor/pi0` | BF16 shared Torch route; seeded synthetic weights or converted OpenPI checkpoint; `robodojo` workload |
| `thor/lingbot_vla` | BF16 model reference; real assets or `synthetic=true`; `robodojo` workload |
| `thor/groot_n17` | BF16 model reference; real checkpoint and prepared observation fixture; `libero` and `robodojo` workloads |
| `rtx5090/groot_n17` | [GR00T N1.7 LIBERO PyTorch reference](src/flash_vla/models/groot_n17/README.md), real checkpoint and observations |

```bash
python -m benchmarks latency --target h100/pi0 --plan shipped --out artifacts/current.json
python -m eval.correctness --target h100/pi0 --plan shipped --steps 1 --layers 1
python -m tools.profiling.model --target h100/pi0 --plan shipped --overview --trace-dir artifacts/profile/overview
```

`shipped` selects the deployed plan; `reference` selects the numerical reference
route. A candidate is a plan JSON under `lab/plans/`. Use the target's real asset
options for checkpoint experiments; synthetic-weight results establish only
that stated workload. Each command exposes its options through `--help`.

The initial Thor targets use BF16 Torch operators and CUDA Graphs;
`shipped` and `reference` currently select the same route. Create the Python
environment with `uv venv` and install Python packages into that environment.
Use an ARM64 CUDA PyTorch build that supports the device, a C++ compiler and
Python development headers for compiled Pi0/Pi0.5 attention, and the Pi0.5
tokenizer (`PALIGEMMA_TOKENIZER`). For Pi0.5's declared workloads:

```bash
python -m eval.model_reference --target thor/pi05 --workload robodojo --steps 1 --layers 1
python -m benchmarks latency --target thor/pi05 --workload robodojo --out artifacts/thor-robodojo.json
```

Thor power-mode provenance comes from `nvpmodel -q`; unsupported NVML clock and
power-limit observations remain unavailable. Hardware floor constants and
device-specific kernels have not been established, so `--workloads` transfer
matrices and floor reports are not yet supported for Thor.

LingBot's synthetic route needs `--option synthetic=true` on both evaluation
and latency commands. GR00T needs the existing logical checkpoint and fixture
IDs resolved through `FLASH_VLA_ASSETS`; its depth is fixed, so evaluate it with
`--steps 0 --layers 0`. See its [asset setup](src/flash_vla/models/groot_n17/README.md).
Official RoboDojo [seed-0 checkpoints](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/35efbc7dedfdbeeb6e95fb749bd885d73d483e41/ckpt/RoboDojo)
are available for Pi0, Pi0.5 and GR00T N1.7. Convert the Pi Orbax checkpoints
with [OpenPI's pinned converter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/convert_jax_model_to_pytorch.py)
before passing `converted_checkpoint`; GR00T loads the dataset's safetensors
directly. LingBot-VLA's official [pretrained](https://huggingface.co/robbyant/lingbot-vla-4b)
and [Robotwin post-trained](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin)
weights load through `checkpoint`. The RoboDojo dataset does not publish a
LingBot-VLA fine-tune; its `Lingbot_VA` directory is a different model.

## Optimize

The workflow lives in the
**[model-optimization](.agents/skills/model-optimization/SKILL.md)** skill, which
defines the kernel and model loops, measurement conditions and the skill to use
at each step. A normal iteration needs a relevant correctness check, local kernel
timing and end-to-end timing after deployment, and can reuse applicable
measurements.

Launch a run with the brief in **[prompts/](prompts/)**, filling in the model,
GPU, workload and budget. The brief carries only those values; the launching
agent binds the skill so its text is injected in full, and the run record keeps
both the injected text and the filled-in values. Comparing agents needs matching
project instructions, tools and budgets — a shared brief standardizes the task,
not the execution.

## Find the right place

| Location | Responsibility |
|---|---|
| `src/flash_vla/models/` | Model semantics, checkpoint formats and loading |
| `src/flash_vla/hardware/` | GPU-specific Targets, shared components and kernels |
| `src/flash_vla/runtime/` | Graph execution, buffers, plan binding and capture |
| `eval/` | Model output accuracy and official numerical references |
| `benchmarks/` | Deployed model and kernel/fusion-chain latency |
| `tools/` | Profiling, trace analysis and hardware-limit estimates |
| `tests/` | Runtime, loading and tool correctness checks |
| `lab/` | Candidate plans and experiments |

[ARCHITECTURE.md](ARCHITECTURE.md) owns the dependency and execution boundaries.
Use [eval](eval/README.md) for accuracy and [tests](tests/README.md) for scoped engineering checks. Published measurements
live in [results](results/README.md). [Result tools](measurement/results/README.md) rebuild
those views from saved traces; [historical plans](docs/history/README.md) remain for reference.

## License

MIT — see [LICENSE](LICENSE). The initial pipeline was extracted from research
commit `a53bcf9`, following the realtime-vla Pi0 implementation; current references
and validation are maintained in `eval/` and beside the relevant kernels.

The adapted GR00T N1.7 reference retains its Apache-2.0 license; see its
[source notice](src/flash_vla/models/groot_n17/README.md#implementation-details-that-preserve-the-official-forward).
