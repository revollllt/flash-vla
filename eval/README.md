# Accuracy evaluation

Compare model outputs against the existing numerical reference. Keep the same
checkpoint, inputs, shape and execution policy on both sides.

```bash
python -m eval.correctness --target h100/pi0 --plan shipped --steps 1 --layers 1
python -m eval.pi0.reference --help
python -m eval.pi05.reference --help
python -m eval.lingbot.parity --help
```

`correctness.py` compares a plan with its in-engine reference, including stage
outputs. The model directories contain official-reference execution and parity
adapters. `metrics.py` owns error calculations; `tolerances.py` owns the existing
numerical thresholds. Use the configured upstream environment and real asset
options for official comparisons; missing assets are reported as unavailable.
Synthetic checks do not establish policy quality.

LIBERO and other task-success evaluations will be added when integrated. There
is no placeholder suite or claim of task-quality coverage today.

Engineering checks live in [tests](../tests/README.md). Model and kernel latency
live in [benchmarks](../benchmarks/README.md); diagnostics live in
[tools](../tools/README.md).
