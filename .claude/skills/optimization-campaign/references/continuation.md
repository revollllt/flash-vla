# Legacy Optimization Campaign

For ordinary optimization use [the direct workflow](../../../../docs/optimization.md).
The procedures below apply only to an explicitly requested legacy Campaign;
never impose them on independent benchmark or kernel experiments.


Keep optimization lineage separate from checkpoint measurements. Resolve the
requested workload and recover its incumbent before proposing another experiment.
The command and evidence contracts live in [lab/optimize](../../../../lab/optimize/README.md);
read the relevant section when preparing a command or receipt, not all historical logs.

## Discover and resume

1. Resolve the request against `benchmarks.targets`: use the registered factory's
   weight-free `declare` with the requested shape and asset options. Use the
   Target's reference runtime as documented in lab/optimize/README.md; weight-free
   does not mean dependency-free. Check the
   returned hardware, architecture revision, inference signature and
   ExecutionVariant against the request. A declaration is not checkpoint ABI or
   numerical evidence. Do not derive architecture revision from a checkpoint,
   upstream commit, engine revision or filesystem location.
2. Read `results/index.json` when present and select the canonical lineage by
   Target, ExecutionVariant, objective and benchmark protocol. Read its linked
   `summary.json`, then compact `trace.json`, and the continuation fields of
   `resume.json`: portable incumbent/source, next iteration, contexts, failed and
   open hypotheses. The representative context is not necessarily the requested
   checkpoint. Do not select a different protocol or an explicit fork silently.
3. Construct the canonical key with `lab.optimize.campaign.campaign_key` from
   the declared identity, requested objective and recorded benchmark protocol.
   Save that small key JSON as an execution input. Use
   `python -m lab.optimize campaign-open-or-seed KEY.json --root REPOSITORY`.
   This prefers an existing local ledger, then a published snapshot. The returned
   directory is the discovery result; do not ask the human for an old Campaign path.
4. Read the returned state: current stage, active context/segment, incumbent,
   remaining budget, unresolved hypotheses, next action and execution repository.
   Follow that execution repository when present. For missing construction inputs,
   inspect the context-matching history record's `spec.options` in `resume.json`
   before loading raw logs. Resolve asset paths locally; a historical
   `source_checkout` does not override the returned execution repository.
   A fresh clone can recover
   history but must validate and re-anchor before allocating another candidate.
   Missing required source commits or corrupt publication remain unresolved;
   they do not justify a replacement lineage or invented history.

If neither local nor published history exists, obtain the registered Target's
real correctness and baseline evidence. Use the documented Registry creation or
onboarding handoff, with declared committed source inputs. Existing history takes
precedence over a newly supplied baseline. Read the
[onboarding skill](../../target-onboarding/SKILL.md) only for an actual onboarding need.

Resume from summary/state and the open hypothesis queue. Load one relevant
iteration receipt or source input when a decision requires it. Raw NCU/NSYS,
old stdout and full artifact trees are lazy diagnostic inputs, never the default
recovery context.

## Checkpoint or context change

For a new checkpoint, keep the same Target and Campaign only after its inference
signature is proven compatible. A mismatch requires resolving the appropriate
architecture revision/Target; do not relabel weights to satisfy the old identity.

Inspect the portable incumbent's declared weight dependency:

- `invariant`: reuse its implementation.
- `rebuild`: execute its recorded artifact recipe with the new weights.
- `retune`: execute its recorded tuning recipe under the fixed quality contract.
- `checkpoint_specific`: keep that result in its original context; inherit the
  portable incumbent instead.

Prepare the documented compatibility, correctness and measurement commands for
`python -m lab.optimize campaign-transition CAMPAIGN --root REPOSITORY --result REQUEST.json`.
The transition owns the inherited source checkout and required recipes. It must
finish correctness and measure the incumbent under the new context before
activation. Observe the returned state rather than assuming the requested
checkpoint became active. Fixture or environment changes also require a new
segment; a protocol change is a distinct CampaignKey.

Re-anchor does not consume an optimization iteration and is not speedup or
regression. Never carry old checkpoint latency into a new context or calculate
cross-context speedup. Resume iteration IDs from the ledger, not from filenames
or the number of successful candidates.

## One experiment

Use the existing acceptance policy and record mechanism, alternative explanation,
falsifier, cheapest probe, affected call sites, expected recoverable latency and
weight dependency before implementation. Rank work by recoverable latency times
confidence divided by validation cost. Start with the cheapest discriminating
probe; a failed probe changes the hypothesis rather than justifying a larger rewrite.

Build the experiment spec for the active workload/context/segment and incumbent.
Target construction options belong in `spec.options`, shared by preflight and
qualification; seed belongs in `conditions.seed`. Real weights need their explicit
path, immutable ID/digest and any required reference configuration. Record fixture
and tokenizer overrides there too. Check and measurement commands must use those
same selections. Do not leave defaults that change the experiment's assets.

Use `preflight SPEC.json`, then `start SPEC.json --campaign CAMPAIGN` through
`python -m lab.optimize`; run the returned run directory through its declared
stages in the appropriate allocation. Preserve fixed ExecutionVariant, checkpoint,
fixture and benchmark protocol throughout uninstrumented A/B/A. Required
correctness failure vetoes promotion. Exclude environment drift and control noise
before attributing a gain. Instrumented timings are diagnostic only.

Qualification supports route variants in one checked source tree and LingBot
backend source variants through explicit candidate `source_checkout` and incumbent
construction options. These comparisons load and record each actual committed
source revision; shared inference code must remain identical. Other Targets do
not yet support distinct-source qualification. Use the relevant
[benchmark skill](../../benchmark-kernel/SKILL.md),
[profiler skill](../../gpu-profiler-analysis/SKILL.md), or
[kernel skill](../../kernel-design/SKILL.md) only when that work is reached.

Finalize with `campaign-finalize CAMPAIGN --iteration N --result RESULT.json`.
Retain accepted, no-benefit, correctness-failed and invalid trials with their
actual evidence. The CLI publishes the context summary, common trace, resume
snapshot, plot and discovery views. A publication failure does not undo a verdict
or authorize rerunning its experiment. Rebuild/check existing results with
`python -m lab.results validate` and `python -m lab.results rebuild --check` at
the appropriate milestone, not after every probe.

## Interrupted or failed work

Inspect the recorded worker/job and its actual current state before resuming.
An observation timeout is not termination. Use `campaign-resume CAMPAIGN` for
completed-stage continuation or publication recovery; interrupted stages require
documented reconciliation and accounting after the owner is confirmed stopped.
Do not silently restart an expensive run or reuse an uncertain GPU allocation.

Retain failure evidence and unblock conditions. After the user's retry limit,
defer that item and continue independent work without claiming it complete.
Report verified progress, the unresolved items and one next action. Keep the
task's existing recovery checklist; do not create parallel progress documents.
Commit validated milestones locally under the user's review policy; this skill
does not authorize pushing, changing acceptance thresholds or expanding scope.
