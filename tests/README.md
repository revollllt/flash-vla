# Engineering tests

Run tests for the component being changed. The default CPU suite covers runtime,
model loading, benchmark behavior and trace analysis.

```bash
python -m pytest -q
python -m pytest -q tests/test_latency.py tests/test_environment.py
python -m tests.targets --target h100/pi05
```

Focused implementation checks have separate commands:

```bash
python -m tests.pi05.fold
python -m tests.pi05.tokenize --help
python -m tests.tile_sm90
```

The tile primitive checks need a GPU. Model output accuracy and official-reference
comparisons belong to [eval](../eval/README.md).

`legacy/` tests historical Campaign/publication consumers and is excluded from
default discovery. Invoke it only when changing those consumers or their shared
interfaces:

```bash
python -m pytest -q tests/legacy
```

Documentation changes need relevant link/example checks. A timing-only change
needs its affected tests; it does not require every model or historical suite.
