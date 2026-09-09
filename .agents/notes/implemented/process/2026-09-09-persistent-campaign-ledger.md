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

The derived context registry retains each activated checkpoint/fixture pair,
its first anchor observation timestamp and latest segment ID. Environment changes
create new segments under the same context. Returning to a prior context updates
its latest segment only after a fresh validated anchor; historical latency is
never reinstated as a current measurement. Pending, failed and aborted transitions
do not register or activate contexts. The registry is rebuilt from anchor evidence,
not maintained as a second authoritative ledger.

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

Campaign report conversion consumes the existing evaluator and latency outputs.
It preserves original A/B/A legs, validates the producer's fixed sampling policy
and refuses diagnostic instrumentation, incomplete statistics or unknown context.
A re-anchor uses repeated measurements of the same implementation; its middle
leg is the anchor, and all three control minima must agree within the existing
acceptance policy. Conversion itself never reruns an experiment.

Correctness conversion uses the existing acceptance ladder, including its
gate/report distinction, numerical tolerances and official adapter provenance.
In-engine checks at different ladder depths must describe the same checkpoint,
fixture and stable environment, with candidate implementation and prescribed
depths bound to the report. Finite and replay-identical outputs remain required.
A diagnostic deep comparison does not acquire a numerical gate by conversion.
Unknown quality contracts cannot authorize nondefault ExecutionVariants.
The reference route and engine revision must match the producer's declared
numerical oracle; required official scripts must exactly cover the registered
adapters. Leg instrumentation cannot contradict an uninstrumented label.
Correctness-only success is explicitly scoped as correctness_pass and cannot
substitute for the full qualification pass required by promotion.

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

Published results use the existing canonical CampaignKey digest as their Target
directory, so checkpoint changes do not split lineage. Explicit forks have
separate child directories. The representative performance is the active
context's latest segment; each other context retains its own latest segment
summary. No cross-context minimum or speedup is defined.

Publication accepts terminal, validated v3 ledgers, including failed experiments,
and refuses pending transitions or missing re-anchors. Compact trace retains
iteration/segment/context, plans, hypotheses, applicability and verdicts; raw
diagnostics, command receipts and profiling files remain local. Context summaries,
canonical Matplotlib SVG and Markdown are staged before replacing the published
view. The trace is replaced first as the atomic ownership/history boundary;
SVG/Markdown precede context summaries; Target summary and global discovery
are written last. Segment and iteration changes therefore expose an interrupted
replacement even when headline latency is unchanged. Local publication is
serialized. Multi-file replacement is not a filesystem-wide transaction: an
interruption requires the same lineage to republish before the view is used.
Missing summary does not release the trace's ownership. Index rebuilding refuses
summary/context views that disagree with trace facts, and unresolved files
without a trace owner cannot be overwritten. Published terminal history cannot
be rewritten or truncated by a later publish.
Automatic CLI publication is bound to a local repository outside Campaign
identity and exported snapshots. Execution checkouts do not relocate that
destination. The existing committed portable-source requirement is checked before
enabling publication and before a portable accepted verdict becomes immutable.
A dirty-source candidate can retain an invalid verdict before a committed,
requalified candidate is allocated; historical evidence is never rewritten.
CLI baseline creation checks declared committed inputs before reserving a lineage.
Terminal verdict persistence precedes publication;
a rendering, replacement or index failure cannot erase or reopen that verdict.
A completion receipt names the last fully published iteration and segment.
Until it matches the ledger, terminal state remains paused for publication and
cannot allocate another candidate. Resume retries publication without repeating
experiment stages or costs. Context validation and fresh-clone re-anchor remain
prerequisites. Successful context activation publishes its new segment without
consuming an optimization iteration.

Published facts must be internally consistent before summary generation or
offline reconstruction: Target and variant, contiguous iterations/segments,
anchor implementation, complete context provenance, portable parent lineage,
terminal verdicts and numerical deltas are checked independently of generated
summaries. This verifies retained facts, not the unavailable raw correctness
or A/B/A receipts. Rebuild must never manufacture those receipts.

Offline validation checks JSON facts and views. Rebuild checks all generated
bytes using the same canonical renderer as publication; check mode does not
modify results. Repair stages every view before replacement, preserves trace
bytes, and shares publication's explicit view-before-summary-before-index
replacement order. Published forks must bind their parent to the canonical
CampaignKey directory and use the Registry's canonical UUID child identity. Unowned files are reported rather than deleted. The renderer
environment is part of reproducibility: current SVG evidence uses Matplotlib
3.10.8. The published-results workflow runs the relevant CPU identity and
continuation tests, validates retained results, checks deterministic rebuild,
and requires an unchanged results diff. It covers changes in the harness,
Targets, runtime, models and result tooling. The job has read-only repository
permissions and does not run GPU measurements or publish performance. A checkout
without tracked results has no publication facts to verify; an empty inventory
cannot satisfy the real-model publication requirement. Hosted runner setup and
dependency installation require their own actual CI execution evidence.

A resume snapshot retains terminal normalized measurement/correctness receipts,
the accepted portable recipes, compact legacy knowledge, open hypotheses and
the Campaign metadata. Raw logs, profiler output, command execution stages and
source bytes remain outside the snapshot. Existing A/B/A leg observations are
retained; missing legs are never inferred from a headline latency. Snapshot
state and its matching compact trace must be reconstructable through the
existing ledger validators. This is validated continuation evidence, not the
original raw experiment archive.

The local ledger is preferred over published state. Seeding is allowed only
from the canonical index entry and a mutually consistent summary, trace and
snapshot. Missing index/trace with an existing snapshot requires offline repair;
it cannot fall back to a new baseline. An interrupted local seed remains an
explicit incomplete directory. The Campaign metadata becomes discoverable only
after its history and portable source have been written.

Portable source is recovered from an exact available Git commit, using its
declared input set. Export compares that input snapshot with the recorded
commit. Historical candidate snapshots are not needed or copied. Source commit
availability and portable rebuild/retune recipes are required for continuation;
a shallow clone missing the incumbent commit fails explicitly. Recipes should
use committed relative code paths and the existing interpreter/output tokens;
old machine paths are not guessed or silently rewritten.

An imported ledger records the last published segment boundary. All historical
latencies, including prior re-anchors, remain history until a newer transition
validates compatibility, rebuild/retune, correctness and a fresh incumbent
anchor. Import consumes no iteration and never resumes historical commands.
Checkpoint-specific records remain in history but cannot supply portable
source. A new transition supplies the execution checkout and materialization
receipt needed by subsequent candidates.

A missing trace can be rebuilt from retained snapshot receipts. Stale derived
snapshot fields can also be regenerated; conflicting trace/snapshot facts
require reconciliation. A missing snapshot cannot be invented from a trace
that lacks normalized receipts and source provenance: republish from the local
ledger. Published history may extend, but its existing trace prefix cannot be
rewritten by continuation.

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
Report-conversion tests execute the real harness and gate logic with CPU toy
engines, then feed their produced schemas through the offline CLI and Campaign
creation. Failure cases include missing gates, wrong depth or implementation,
checkpoint/context drift, weak official evidence and changed latency sampling.
They do not certify real model numerical parity or GPU measurement provenance.
Registry tests use separate Python processes contending on the same key, real
temporary Git source snapshots, corrupt authoritative files and stale derived
state. CLI tests resolve Campaign paths without human directory naming.
Publication tests exercise actual ledger-to-file output, context-local summaries,
deterministic repeated generation, failure trace retention, lineage ownership,
fork-first publication and render/replacement failure recovery. Automatic
publication tests cover all terminal verdicts, actual context activation and CLI
finalization. Injected renderer, index and completion-receipt failures retain
the verdict and resume publication with experiment execution forbidden.
Actual dirty source files exercise pre-promotion rejection and recovery through
a new committed candidate. Missing source and execution-checkout CLI tests verify
that configuration failures cannot trap an immutable ledger or relocate results. A same-number
environment transition cannot index an old SVG after replacement is interrupted.
Offline rebuild tests inject stale SVG, summaries and global views, invalid
identity/context/arithmetic, and a trace edit without plot regeneration.
They also remove the original ledger and rebuild published views in another
filesystem layout. Check mode preserves damaged files for inspection; repair
produces the expected bytes without running an experiment.
Fresh-clone tests use actual Git repositories and delete the original checkout
before importing the snapshot. CPU toy checkpoint transitions execute inherited
recipes and checks, then continue and publish on the same lineage. Tests cover
pre-existing segments, a second clone, checkpoint-specific exclusion, missing
commits, corrupted receipts, missing discovery files and interrupted metadata
publication. Legacy rejected and open hypotheses survive the import.
These fixtures do not establish real Pi0.5/LingBot checkpoint transfer or GPU
performance.

Related: [the original optimization campaign](../performance/2026-09-06-optimization-campaign-plan.md).
