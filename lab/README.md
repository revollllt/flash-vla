# Experiments

Use [the optimization workflow](../docs/optimization.md). Candidate plans live
under `plans/`; model/component directories hold focused experiments and parity
harnesses. They may import production code; production never imports `lab/`.
A candidate enters deployment only through its Target's plan and backend.

`stage_dump.py` compares declared stage outputs. `examples/` contains diagnostic
examples. Experiment outputs belong under ignored `artifacts/` directories.

[Result tools](results/README.md) render saved traces and dashboards.
Campaign creation, onboarding ledgers, qualification and automatic publication
have been retired; their code remains available in Git history.
