# Experiments

Use [the optimization workflow](../docs/optimization.md). Candidate plans live
under `plans/`; model/component directories hold focused experiments and parity
harnesses. They may import production code; production never imports `lab/`.
A candidate enters deployment only through its Target's plan and backend.

`stage_dump.py` compares declared stage outputs. `examples/` contains diagnostic
examples. Experiment outputs belong under ignored `artifacts/` directories.

## Optional legacy tools

`optimize/`, `onboarding.py` and `results/` retain their existing command interfaces
for historical Campaigns and publication consumers. Their
[procedure](optimize/README.md) and `eval/legacy_tests/` apply only to that work.
New kernel experiments do not need a Campaign, receipt or publication step.
