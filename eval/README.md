For daily optimization use [the direct workflow](../docs/optimization.md).
Run correctness checks relevant to the change. `eval.gate` is an explicit
complete qualification tool, not the default iteration entry point.

# Evaluation

Evaluation is intentionally separate from performance benchmarking, and the
deployment checks here are separate from the trials under `lab/`.

- `acceptance.py`: the one registry of what the human defines (accuracy
  requirements, framework conventions, each Target's budget).
- `correctness.py`: any Target's candidate plan against its reference plan,
  stage by stage, on the captured runner.
- `gate.py`: the promotion gate; turns the registry's checks and an A/B/A
  latency run into one verdict and an evidence record.
- `smoke.py`: every Target's declarations checked without a GPU.
- `metrics.py`: the error metrics every check imports.
- `pi05/`, `pi0/`: the official-baseline tier per model (`reference.py`
  against OpenPI, plus the model-contract checks `fold.py` and `tokenize.py`
  for Pi0.5); `baselines/` holds the adapters they use.
- `tile_sm90/`: the parity suite of the shared SM90 tile primitives.
- `tasks/`: policy quality in environments such as LIBERO (out of scope now).

Production code under `src/` must not import this package.
