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
V3 workload identity also separates ExecutionVariant; the pending
[checkpoint-independent transition proposal](../../proposed/architecture/2026-09-09-checkpoint-independent-optimization.md)
moves fixture measurement provenance outside campaign identity. `campaign.json`, iteration evidence and source materialization receipts are facts;
status, normalized trace, Markdown, and plots are rebuildable views. Incumbent
movement is derived only from atomic terminal evidence. The canonical plot is
rendered by Matplotlib from normalized plot points and marks environment
re-anchors separately from code promotions.

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
These CPU tests do not establish real checkpoint transfer or recipe execution.

Related: [the original optimization campaign](../performance/2026-09-06-optimization-campaign-plan.md).
