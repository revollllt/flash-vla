# Bounded optimization experiments

This controller is for the existing H100 Pi0/Pi0.5 tools. Acceptance remains
owned by `eval/acceptance.py`; `eval.gate` remains the qualification verdict.
The owner requested no new hashes, frozen contracts, baselines or gates.

Use `python -m lab.optimize preflight SPEC.json` before reserving a GPU, then
`start SPEC.json --out RUN`. Run declared stages with `run RUN --until check`
or `--until measure` inside the existing Slurm allocation. Experiment fields
are defined in `schema.py`; the exercised component specification is retained
at `artifacts/optimization/component-spec.json`.

A task's trials must share `task_id` and an index directory for candidate,
non-improving and job accounting. Distinct run IDs are not new task budgets.
Classification must distinguish invalid measurements from valid no-benefit
results. `related RUN` reports prior applicable experiments and explicit
reopening reasons; deliberate replication uses `replication_of`.

An interrupted stage never silently repeats. `reconcile RUN` requires positive
owner-termination evidence. For a worker killed before it could account cost,
provide `--recovered-seconds` from accounting. This is an explicit recovery
value, not an estimated timer reading. Logs and previous attempts remain.
Allocated job time, including setup failures, is separately retained in Slurm
accounting; controller stage time alone is not the campaign's total cost.

Source copies cover declared inputs only. Dependency selection belongs to the
experiment author. Unlisted external/runtime dependencies remain outside the
copy's identity coverage; this cannot prove a no-op. Actual loaded artifacts,
workload conditions and sampling protocol belong in the probe result.

Supported coexistence is deliberately narrow: the existing Gemma CUDA prefix
attention component has isolated libraries with separate graphs and two deployed
shapes. QKV diagnostics support the installed TileLang TVM-FFI adapter. Other
backend/loading protocols are unsupported until separately exercised.

Qualification currently supports route variants within one checked source tree.
It requires explicit incumbent/candidate source evidence, affected Targets and
live checkout paths. Source-version qualification through `eval.gate` is
unsupported: two independent diagnostic libraries do not by themselves make
that evaluator load both versions. A passing gate still requires current-source
applicability review; the controller never edits shipped or merges candidates.
Conflicting candidates need a combined experiment before review.

Attention fixtures are correctness snapshots, not production performance
fixtures. The QKV pilot rotates 100 MiB of weights across ten calls per graph.
Isolated latency and profiled software ranges cannot substitute for E2E
qualification. See `docs/optimization-results.md` for evidence and limitations.
