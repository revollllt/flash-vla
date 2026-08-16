# Evaluation

Evaluation is intentionally separate from performance benchmarking.

- `baselines/` contains adapters for official model implementations.
- `correctness/` compares target outputs and named intermediate stages against
  those baselines.
- `tasks/` measures policy quality in environments such as LIBERO.

Production code under `src/` must not import this package.
