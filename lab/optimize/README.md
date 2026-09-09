# Bounded optimization experiments

This controller is for the existing H100 Pi0/Pi0.5 tools. Acceptance remains
owned by `eval/acceptance.py`; `eval.gate` remains the qualification verdict.
The owner requested no new hashes, frozen contracts, baselines or gates.

V3 campaigns are located by CampaignKey through CampaignRegistry. Create one
from validated baseline evidence with `campaign-create --baseline-evidence REPORT
--objective NAME --protocol NAME --fixture NAME`; the returned directory is
canonical under artifacts/optimization/campaigns. V3 creation takes no manual
Campaign directory. The legacy v2 command still requires one. With a canonical
key JSON, `campaign-find KEY.json`, `campaign-open KEY.json` and
`campaign-open-or-create KEY.json` discover or restore the default lineage.
The last command needs --baseline-evidence only when no local Campaign exists.
Use --root to select the repository.

Opening a known key with new checkpoint evidence restores the existing lineage;
it does not activate the new checkpoint. Complete the context transition before
comparing its measurements. A missing local Campaign and a corrupt one are
different outcomes: corruption never creates a replacement. Published snapshot
seeding is not implemented yet.

Use `campaign-fork KEY.json --reason TEXT` for an explicit alternative lineage,
then `campaign-open KEY.json --fork-id ID` to reopen that fork. Forks retain
terminal history, unresolved hypotheses and declared source snapshots. The portable
source must match its committed engine revision; each fork gets an independent
execution checkout. They require no active candidate or transition and do not copy
raw logs or rerun jobs. Failed checkout preparation leaves an unpublished fork
that cannot be opened as a ready Campaign.

After resolving the directory, allocate candidates with `start SPEC --campaign CAMPAIGN`. The spec must
declare the baseline's workload identity and protocol. V3 candidates also bind the
active measurement_context and measurement_segment; their fixture identifies that
context rather than a permanent Campaign identity. `campaign-status` and
`campaign-validate` rebuild `state.json`; `campaign-resume` continues a clean
stage or requires explicit reconciliation for an interrupted one. Record a
terminal result with `campaign-finalize CAMPAIGN --iteration N --result
RESULT.json`; verdict, correctness, measurement and qualification stay in that
iteration's evidence.

`campaign.json`, optional ranked `hypotheses.json`, iteration evidence,
transition evidence, and source materialization receipts are campaign facts. `state.json` is a disposable
view. The campaign identity is created once and has no update operation.
Accepted evidence advances the incumbent only after
correctness, valid measurement, qualification and gate results all pass;
other verdicts stay in the lineage without changing it. V3 performance candidates
must declare applicability as invariant, rebuild, retune, or checkpoint_specific.
Rebuild and retune require executable artifact_recipe and retune_recipe commands,
respectively. An accepted checkpoint-specific result remains context-only; it
cannot replace the portable incumbent or its plotted latency.

Before editing the next candidate after a checkpoint-specific experiment, use
`campaign-materialize CAMPAIGN --root CHECKOUT`. It restores the recorded portable
source inputs and returns its plan; the next spec declares that plan as
`parent_plan`. Unrecorded checkout edits are preserved by refusing restoration.
Baseline source coverage is explicit through repeatable `--source-input PATH`
arguments to campaign-create. Missing source coverage must be resolved before
materialization; declared-input copies are not a full repository snapshot.

V3 context activation uses `campaign-transition CAMPAIGN --root CHECKOUT
--result REQUEST.json`. The request supplies identity, protocol, objective,
measurement_context and compatibility/check/measure commands. Commands emit a
single normalized JSON report to stdout. Compatibility binds the workload ABI
and assets; correctness additionally binds the exact implementation and stable
environment. Latency carries objective name/value/unit, protocol, validity and an
explicit instrumented=false marker. Missing provenance prevents activation.

Transitions retain their own command evidence and costs outside optimization
iterations. The portable engine commit runs in an isolated clean checkout; its
declared source snapshot must match that commit. Every new execution directory
runs the inherited rebuild/retune recipes, then correctness and measurement.
Only successful activation changes the current context and segment. The state
returns execution_repository for continued work. Candidate measurements include
A/B/A leg evidence; parent and candidate scalars must derive from those validated
same-context legs under the fixed protocol.

`campaign-resume` resumes completed transition stages without repeating them.
Failed or interrupted stages require explicit reconciliation and recovered cost,
as for ordinary runs. `campaign-transition-abort` preserves failed evidence and
requires a new re-anchor before candidate allocation. The legacy re-anchor
command remains available only for v2 ledgers.

The normalized trace stores segment anchors separately from optimization
iterations. Rendering breaks latency lines at segment boundaries, annotates
checkpoint/fixture provenance, and produces deterministic SVG from the same
trace.

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

Transition commands must use the validated repository as their execution source;
the runner puts that checkout first in PYTHONPATH. Use repository modules for
its measurement/evaluation code and explicit paths for external reference tools.
The normalized command boundary is not a substitute for a registered Target's
actual compatibility, correctness or benchmark evidence.

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

Production environment reports distinguish the requested clock protocol from
observed policy. GPU UUID, driver and requested/enforced power limits refer to
the runner's device. Application clock readings alone do not certify locked
clocks: until the actual policy is established, clock_policy remains null and
the complete-context check rejects formal acceptance. Environment probes do not
substitute for model correctness, re-anchor or A/B/A evidence.

The existing evaluator can run its registered correctness ladder separately via
`python -m eval.gate --target TARGET --baseline --correctness-only`. Required
failures still stop subsequent work. Success exits zero with the distinct
correctness_pass verdict, which cannot authorize promotion without full
qualification. Formal gate timing disables attribution
sampling; diagnostic timing remains available through the benchmark CLI.

`python -m lab.optimize.reports correctness GATE.json --out CHECK.json` converts
a completed ladder report. `anchor LATENCY.json --correctness CHECK.json` converts
an incumbent A/A/A run; `comparison LATENCY.json --segment N` converts candidate
A/B/A evidence. Both accept --objective and --out. A full gate report can also
supply its embedded latency, and its correctness can be used directly for an
anchor. These commands read existing reports and emit one normalized JSON
object; they never rerun measurement or infer missing provenance.

Conversion requires the actual latency-v2 repetition, warmup and soak settings,
valid control spread, complete uninstrumented statistics, and bound correctness
evidence. A faster numerical failure cannot become acceptable through this
format conversion. Compatibility command production, actual checkpoint loading
and verifiable clock policy remain necessary for the real transfer drill.


Terminal v3 Campaign views can be published with
`python -m lab.results publish CAMPAIGN --root REPOSITORY`. The generated
results/index.json provides discovery and results/README.md presents the active
context's performance. Each Campaign has a compact trace, per-context summaries
and the canonical progress.svg; checkpoint changes retain the same Target
directory. A failed publication can be retried without rerunning the experiment.
Publication includes a compact resume snapshot. A fresh checkout can run
`python -m lab.optimize campaign-open-or-seed KEY.json --root REPOSITORY`.
It prefers a local Campaign, otherwise discovers the canonical published
lineage; only an absent lineage may use `--baseline-evidence` to create one.
The incumbent Git commit and declared source inputs must exist locally.
Imported history retains iteration IDs, failed/open hypotheses and portable
recipes. A fresh context transition is mandatory before candidate allocation
or publication, even when the checkpoint and old environment metadata match.
Automatic publication after finalize is still pending.
`python -m lab.results validate --root REPOSITORY`
checks published facts and JSON views.
`python -m lab.results rebuild --check --root REPOSITORY` checks generated files
without modifying results; omit
`--check` to repair derived views from valid traces. Rebuild needs Matplotlib
(the current SVG evidence uses 3.10.8) but no old Campaign artifacts or GPU
measurement. Invalid trace facts and unowned files require explicit
reconciliation; rebuild does not invent missing measurement evidence.
A missing trace or stale derived snapshot state can be rebuilt from retained
snapshot facts. Missing normalized receipts cannot be inferred from a trace:
republish from the original ledger if resume.json is absent. Hosted CI wiring
remains pending.
