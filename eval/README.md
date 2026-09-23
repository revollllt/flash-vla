# Accuracy evaluation

Compare model outputs against the existing numerical reference. Keep the same
checkpoint, inputs, shape and execution policy on both sides.

```bash
python -m eval.correctness --target h100/pi0 --plan shipped --steps 1 --layers 1
python -m eval.pi0.reference --help
python -m eval.pi05.reference --help
python -m eval.pi05.parity --help
python -m eval.lingbot.parity --help
```

`correctness.py` compares a plan with its in-engine reference, including stage
outputs. The model directories contain official-reference execution and parity
adapters. `metrics.py` owns error calculations; `tolerances.py` owns the existing
numerical thresholds. Use the configured upstream environment and real asset
options for official comparisons; missing assets are reported as unavailable.
Synthetic checks do not establish policy quality.

`pi05/reference.py` runs OpenPI and the Target in one process, which needs the
whole upstream stack importable beside `flash_vla`. `pi05/parity.py` is the same
comparison split across two interpreters, for a machine whose environment is the
pinned flash-vla one: `capture` writes the official tensors and the fixture it
used, `compare` replays that fixture through the Target. `OPENPI_PI05_MODULE`
names the module the official forward comes from, and the oracle records which
one ran.

## Pi0.5 LIBERO task success

`python -m eval.libero` runs an OpenPI PyTorch reference or the Flash-VLA
shipped plan in LIBERO. Use the official `pi05_libero` checkpoint converted to
PyTorch, including its `assets/physical-intelligence/libero/norm_stats.json`.
The converted `config.json` must carry OpenPI's model settings: `pi05=true`,
`action_horizon=10`, `discrete_state_input=false`, `max_token_len=200`.
Belt-cup weights are a different policy and cannot establish LIBERO quality.

The evaluation environment needs the normal Flash-VLA runtime, the official
LIBERO checkout (including assets and initial states), robosuite 1.4.1,
MuJoCo, Pillow, and an OpenPI PyTorch implementation through
`OPENPI_PI05_MODULE`. Configure `LIBERO_CONFIG_PATH`, `MUJOCO_GL` and the
matching `PYOPENGL_PLATFORM` for that machine. EGL and OSMesa are supported by
the simulator; software rendering affects run time and must be recorded.
Machine-specific environment setup belongs in ignored artifacts.

```bash
python -m eval.libero --engine official --checkpoint "$LIBERO_CHECKPOINT" \
  --tokenizer "$PALIGEMMA_TOKENIZER" --suite libero_spatial --trials 50 \
  --out artifacts/libero/spatial-official.json
python -m eval.libero --engine flashvla --checkpoint "$LIBERO_CHECKPOINT" \
  --tokenizer "$PALIGEMMA_TOKENIZER" --suite libero_spatial --trials 50 \
  --out artifacts/libero/spatial-flashvla.json
```

Start with `--task-ids 0 --trials 1` for a smoke episode. The first observation
and noise are saved next to each report as `.fixture.npz`; use
`--fixture <saved.npz>` with either engine to compare the same observation
without running the simulator. This mode saves normalized and unnormalized
actions in `.npz`. It does not measure success rate.

The episode protocol follows
[OpenPI's LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py):
256-pixel rendering, 180-degree image rotation and bilinear resize to 224,
eight state components, ten settling steps, five executed actions per replan,
and 220 policy steps for Spatial. The model uses task-only tokenization,
quantile normalization, chunk 10 and ten denoising steps. OpenPI retains a
masked third image; Flash-VLA omits that masked view and executes the two real
views. Same-observation comparison is needed before interpreting task results.

The two engines use the same 50 supplied initial states per task, reset both
environment and NumPy seeds to `seed + task_id * 1000 + trial`, and draw paired
noise with an independent NumPy generator per episode. Noise is rounded to
BF16 before either model consumes it. This deliberately controls randomness
across implementations; it differs from running an unmodified upstream script
with its own random-number stream. The reference executes eagerly and retains
upstream RoPE buffers by default. `--exact-rope` restores FP32 frequencies for numerical diagnosis
only and is not the default task-success reference.

A report records each episode, completed count, successes, seeds, renderer and
source state after every trial. Runtime exceptions abort visibly and are not
counted as policy failures. Only a report with `status=complete` and the intended
number of episodes supports a full-suite result. Adding this runner does not
itself establish that the optimized policy preserves task success.

Engineering checks live in [tests](../tests/README.md). Model and kernel latency
live in [benchmarks](../benchmarks/README.md); diagnostics live in
[tools](../tools/README.md).
