# Agent Note: persistent campaign ledger and legacy migration

Status: implemented

## Problem

The experiment controller retained one run at a time, while optimization spans
many sessions, source worktrees, accepted incumbents, rejected hypotheses, and
measurement environments. Historical Pi0/Pi0.5 results also predate model
revision identity. Treating their prose summaries as current measurements would
mix workloads and attribute environment drift to code.

## Decision

Legacy campaigns bind TargetKey, objective, benchmark protocol and fixture.
V3 Campaign identity binds the workload (TargetKey and ExecutionVariant),
objective and benchmark protocol. Checkpoint and fixture provenance identify a
measurement context; the stable measurement environment additionally identifies
its segment. Context changes preserve Campaign and optimization iteration IDs.

The local Registry maps the semantic CampaignKey to one canonical directory.
Its digest uses the existing small-metadata serializer; it never reads source
or tensor bytes. A Registry lock serializes discovery and creation, and the
Campaign lock protects validation/reconstruction against concurrent finalization.
Missing data may permit a new lineage; partial or corrupt authoritative data
does not. Derived state can always be reconstructed from the validated ledger.

An alternative lineage requires an explicit fork reason and retains parent
Campaign and iteration provenance. Forking is limited to terminal ledgers and
preserves iteration genealogy, failed hypotheses and independent declared source
snapshots. Raw logs, profiler output and execution checkouts are not duplicated.
Fork execution uses an independent checkout of the committed portable engine;
historical parent paths remain provenance, never the child's execution entrypoint.
The child becomes discoverable only after checkout preparation succeeds.
The default key continues to discover the original Campaign. New checkpoint
discovery never silently changes the active context.

Campaign metadata, iteration evidence, transition evidence and materialization
receipts are authoritative; state, normalized trace and plots are rebuildable
views. Context activation requires compatible assets, the inherited portable
implementation, its necessary artifact/calibration recipes, correctness and an
uninstrumented incumbent measurement. Re-anchor is a segment boundary, never an
optimization iteration or a cross-context speedup.

A transition executes the committed portable engine in a clean isolated
checkout. Declared inputs must match that commit, and actual runner provenance
must match the execution revision. Every new execution directory rebuilds
required artifacts; equal asset identities do not prove artifacts exist there.
Completed stages survive interruption. Failed stages require explicit
reconciliation; aborted transitions require a fresh anchor before comparison.

Correctness binds the full stable context, while structural compatibility is
independent of candidate kernels. Candidate and parent latencies derive from
validated same-context A/B/A legs, including the fixed sampling protocol and
control spread policy. The canonical Matplotlib plot contains separate latency
lines per segment and independent anchor annotations; reproducible SVG IDs
ensure deterministic output.

Measurement environment queries select the actual CUDA device by UUID, including
visibility remapping. Requested and enforced power limits are distinct measured
values. Query failures remain errors. Environment snapshots bracket every latency
leg, including the final control, so a boundary change invalidates the run.
Application clocks are observations; they do not establish the GPU locked-clock
policy. The protocol request remains separate, and an unverified clock policy is
null and cannot satisfy the existing complete-context requirement. A policy change
that happens and reverses entirely between snapshots is not detected by this
boundary check; continuous diagnostic sampling is a separate source of evidence.

The registered Pi histories import the original job 600566 latency JSON and
candidate JSONL rows verbatim. They receive `legacy_import` evidence level,
with unresolved classification retained instead of inferred. The seed-0 model
revision is recoverable because the recorded config names seed 0 and the
checkpoint generator did not change when revision naming was added. A migrated
campaign is paused until a fresh Identity-v2 incumbent measurement is recorded;
that control record consumes no candidate budget and is not a promotion.

V3 performance candidates declare weight dependence explicitly. Portable lineage
contains only accepted invariant, rebuild and retune optimizations; the latter
two carry executable recovery recipes. Checkpoint-specific evidence can be
accepted for its context without promoting the portable implementation. Legacy
records do not implicitly acquire portable applicability.

After a checkpoint-specific trial, subsequent source preparation restores the
portable snapshot and its plan before a new candidate is edited. Restoration
covers declared inputs only and preserves unrecorded checkout changes by refusing
to overwrite them. A materialization receipt records this source preparation;
labeling a candidate's parent alone does not establish implementation inheritance.

## Alternatives considered

- Reconstructing missing A/B/A values from notes was rejected because summaries
  do not retain all samples or identity axes.
- Accepting a portable parent label without restoring its implementation was
  rejected because checkpoint-specific code could remain in the next candidate.
- A separate mutable Registry index was rejected because the canonical location
  already provides discovery without another source of truth.
- Silently replacing an incomplete directory was rejected because an interrupted
  creator may have retained evidence that needs explicit recovery.
- Dropping rejected history was rejected because it would erase falsifiers and
  make repeated work likely.
- Treating a re-anchor as an optimization gain was rejected because source and
  environment changes are confounded across the boundary.

## Consequences

Deleting a candidate checkout or any derived view does not lose campaign state.
Historical numbers remain auditable but cannot enter a current comparison until
the explicit re-anchor. Optional HTML is a view of the same normalized points,
not another attribution implementation.

## Verification

Campaign tests cover lineage, budgets, recovery, source deletion, immutable
Target identity, semantic trace reconstruction, target isolation, legacy value
preservation, and mandatory re-anchor. Renderer tests exercise SVG plus optional
PNG/HTML using Matplotlib 3.10.8 in the original environment. Applicability tests
cover dependency validation, portable ancestor recipe retention, actual temporary
file restoration, context-only trace points and preservation of unrelated edits.
Transition tests execute CPU subprocesses against small on-disk weights,
materialize packed/calibration outputs, reject incompatible input and stale
measurement evidence, recover failed commands, and retain iteration numbering.
Separate Git commits verify the actual isolated execution revision. Rendering
tests verify disconnected segment lines and byte-identical repeated SVG output.
Registry tests use separate Python processes contending on the same key, real
temporary Git source snapshots, corrupt authoritative files and stale derived
state. CLI tests resolve Campaign paths without human directory naming.
These fixtures do not establish real Pi0.5/LingBot checkpoint transfer or GPU
performance.

Related: [the original optimization campaign](../performance/2026-09-06-optimization-campaign-plan.md).
