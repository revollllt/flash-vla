# Evaluation

For an optimization, run the affected correctness path from
[docs/optimization.md](../docs/optimization.md). Numerical thresholds remain in
`eval/acceptance.py`; `eval/metrics.py` computes the shared error metrics.

## Direct checks

```bash
# Default CPU suite: runtime, workload construction, timing and trace behavior.
python -m pytest -q
# A timing-only change needs only its affected tests.
python -m pytest -q eval/tests/test_latency.py eval/tests/test_environment.py
# GPU example: the changed model path, at the relevant shape/depth.
python -m eval.correctness --target h100/pi0 --plan shipped --steps 1 --layers 1
```

Use a model's existing official-reference adapter when checkpoint conversion or
model semantics change. Kernel refactors need the affected numerical checks;
performance claims also need uninstrumented measurements under matched conditions.
Documentation-only changes need link/example checks, not model or GPU reruns.

## Legacy consumers

Campaign, onboarding receipts, migration and complete qualification tests live
in `eval/legacy_tests/`. Run them when changing those tools or a shared interface
they consume:

```bash
python -m pytest -q eval/legacy_tests
```

`python -m eval.gate` remains an explicit legacy qualification command. The
results workflow validates publication when its source or dependencies change;
it is not part of a routine kernel experiment. Production source imports no eval.
