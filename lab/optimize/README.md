# Legacy Campaign tools

For daily optimization, use [the direct workflow](../../docs/optimization.md).
This page documents the optional historical controller. Its activation, A/B/A,
qualification and publication requirements apply only when explicitly using
that controller; they are not prerequisites for new experiments.


This controller is for the existing H100 Pi0/Pi0.5 tools. Acceptance remains
owned by `eval/acceptance.py`; `eval.gate` remains the qualification verdict.
The owner requested no new hashes, frozen contracts, baselines or gates.

Real-checkpoint Pi0.5 declarations resolve the upstream OpenPI configuration and
require the configured `OPENPI_PYTHON`, including its local `openpi-client` package;
the CPU tooling interpreter is insufficient. In the existing lab-H100 worktree,
source `artifacts/optimization/reference-runtimes-lab-h100.env` to select the
reference runtimes. Set `PYTHONPATH=$PWD/src:$PWD` in the intended source checkout
so its code precedes any editable installation. On a shared login node, bound
OpenPI/JAX CPU affinity to the assigned CPU set (four available CPUs for a
declaration probe) and set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before
importing. These are machine-local execution settings, not Target identity.

V3 campaigns are located by CampaignKey through CampaignRegistry. Create one
from validated baseline evidence with `campaign-create --baseline-evidence REPORT
--objective NAME --protocol NAME --fixture NAME`; the returned directory is
canonical under artifacts/optimization/campaigns. V3 creation takes no manual
Campaign directory. The legacy v2 command still requires one. With a canonical
key JSON, `campaign-find KEY.json`, `campaign-open KEY.json` and
`campaign-open-or-create KEY.json` discover or restore the default lineage.
The last command needs --baseline-evidence only when no local Campaign exists.
Use --root to select the repository. CLI baseline creation requires declared
committed inputs through --source-input; it refuses missing or dirty source
before reserving the Campaign.

Opening a known key with new checkpoint evidence restores the existing lineage;
it does not activate the new checkpoint. Complete the context transition before
comparing its measurements. A missing local Campaign and a corrupt one are
different outcomes: corruption never creates a replacement. Use
`campaign-open-or-seed` to recover a published lineage when no local ledger exists.

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

Qualification takes Target construction options from the experiment's
spec.options mapping, shared with preflight, and seed from conditions.seed. Real-checkpoint runs
must record checkpoint location, immutable ID/digest and required reference
configuration there, together with tokenizer or fixture overrides. Option values
use the existing gate CLI's string, integer and boolean types; Target and official
adapter support still applies. These options survive resume and participate
in duplicate-experiment comparison. Records using the obsolete conditions.options
field must move it to spec.options before preflight or resume; it is not silently ignored.

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

Production environment reports distinguish the benchmark's inherited-device
execution policy and explicit Slurm frequency request from application-clock
observations. GPU UUID, driver and requested/enforced power limits refer to the
runner's device. Effective administrative locked-clock bounds remain unobserved;
the execution policy is not a certificate of globally unlocked hardware.
Policy and observed environment changes invalidate the existing segment.
Environment probes do not substitute for model correctness, re-anchor or A/B/A
evidence.

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
Publication includes a compact resume snapshot retaining each experiment's explicit
construction options. These are historical inputs: preserve semantic settings such
as `openpi_config`, but resolve checkpoint/fixture paths on the current machine
and match the recorded context before executing. Historical options do not
authorize reusing old paths or activating a different checkpoint. A fresh checkout can run
`python -m lab.optimize campaign-open-or-seed KEY.json --root REPOSITORY`.
It prefers a local Campaign, otherwise discovers the canonical published
lineage; only an absent lineage may use `--baseline-evidence` to create one.
The incumbent Git commit and declared source inputs must exist locally.
Imported history retains iteration IDs, failed/open hypotheses and portable
recipes. A fresh context transition is mandatory before candidate allocation
or publication, even when the checkpoint and old environment metadata match.
V3 CLI creation, open-or-create, open-or-seed and fork configure publication
to the selected repository. A terminal baseline, finalized iteration or activated
context is published automatically. Imported history still requires its fresh
re-anchor first. A publication failure preserves the terminal verdict and exposes
current_stage=publication; use `campaign-resume CAMPAIGN` to retry publication
without rerunning experiment stages. New candidates wait until publication succeeds.
The local publication destination is excluded from identity and resume snapshots.
Once configured, later execution checkouts do not change it. A portable accepted
candidate must have declared source matching its measured commit before its
verdict becomes immutable. If this check fails, retain the unusable trial as
invalid, commit the candidate and requalify in a new iteration.
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
republish from the original ledger if resume.json is absent. The published-results CI workflow runs the CPU integrity suites and these
validation commands; it does not establish GPU or real-checkpoint acceptance.
