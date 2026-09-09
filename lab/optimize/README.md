# Bounded optimization experiments

This controller is for the existing H100 Pi0/Pi0.5 tools. Acceptance remains
owned by `eval/acceptance.py`; `eval.gate` remains the qualification verdict.
The owner requested no new hashes, frozen contracts, baselines or gates.

A long-running campaign adds lineage around these same experiments. Create one
from an existing baseline report with `campaign-create CAMPAIGN
--baseline-evidence REPORT --objective NAME --protocol NAME --fixture NAME`,
then allocate candidates with `start SPEC --campaign CAMPAIGN`. The spec must
declare the baseline's Identity v2, protocol and fixture. `campaign-status` and
`campaign-validate` rebuild `state.json`; `campaign-resume` continues a clean
stage or requires explicit reconciliation for an interrupted one. Record a
terminal result with `campaign-finalize CAMPAIGN --iteration N --result
RESULT.json`; verdict, correctness, measurement and qualification stay in that
iteration's evidence.

`campaign.json`, optional ranked `hypotheses.json`, and
`runs/iter-*/evidence.json` are campaign facts. `state.json` is a disposable
view. The campaign identity is created once and has no update operation.
Accepted evidence advances the incumbent only after
correctness, valid measurement, qualification and gate results all pass;
other verdicts stay in the lineage without changing it.

`campaign-render CAMPAIGN` normalizes ledger evidence once into
`optimization_trace.json`, writes `progress.md`, and sends only that normalized
trace to the repository's Matplotlib renderer for canonical `progress.svg`.
Use `--png` and `--html` for optional previews; the static HTML formats the
same plot points and does not calculate a second attribution result.

The registered Pi0/Pi0.5 history can be imported with
`campaign-migrate-legacy CAMPAIGN --root REPOSITORY`; the output directory name
selects `pi0` or `pi05`. Imported v1 evidence remains verbatim under
`raw_measurement`, is labeled `legacy_import`, and pauses the campaign. Supply
a fresh normalized incumbent measurement with `campaign-reanchor CAMPAIGN
--result REPORT` before allocating another candidate. A re-anchor updates the
measured incumbent but is never labeled as a code promotion.

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
