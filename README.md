# flash-vla

VLA inference specialized for a model, workload and GPU. Targets describe the
forward graph and choose kernels through a plan; shared components provide
reusable CUDA/TileLang implementations.

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
| `h100/lingbot_vla` | Real checkpoint and fixture, resolved through `FLASH_VLA_ASSETS` |

```bash
python -m benchmarks latency --target h100/pi0 --plan shipped --out artifacts/current.json
python -m eval.correctness --target h100/pi0 --plan shipped --steps 1 --layers 1
python -m tools.profiling.model --target h100/pi0 --plan shipped --overview --trace-dir artifacts/profile/overview
```

`shipped` selects the deployed plan; `reference` selects the numerical reference
route. A candidate is a plan JSON under `lab/plans/`. Use the target's real asset
options for checkpoint experiments; synthetic-weight results establish only
that stated workload. Each command exposes its options through `--help`.

## Optimize

Start with **[the optimization workflow](docs/optimization.md)**. It defines the
kernel and model loops, measurement conditions and the skill to use at each step.
A normal iteration needs a relevant correctness check, local kernel timing and
end-to-end timing after deployment. It can reuse applicable measurements.

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
live in [results](results/README.md). [Result tools](lab/results/README.md) rebuild
those views from saved traces; [historical plans](docs/history/README.md) remain for reference.

## License

MIT — see [LICENSE](LICENSE). The initial pipeline was extracted from research
commit `a53bcf9`, following the realtime-vla Pi0 implementation; current references
and validation are maintained in `eval/` and beside the relevant kernels.
