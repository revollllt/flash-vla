# Agent Note: evidence-first Target onboarding

Status: implemented

## Problem

An onboarding record must distinguish a compatible model architecture from the
initial fine-tuned checkpoint and reference implementation. A textual Campaign
path and claimed baseline state do not establish a recoverable optimization
handoff, especially after publication failure or a checkpoint change.

## Decision

V2 onboarding separates the canonical Target axes, ExecutionVariant, initial
weight provenance and reference repository/commit. Model and observed checkpoint
contracts use the existing machine-checkable inference signature. Weight values
and filesystem locations do not identify the architecture. Incompatible
checkpoint structure or inference semantics require another compatible Target.

Stage dependencies are requirements, reference, model contract, weights
compatibility, official oracle, computation inventory, Target bring-up,
correctness ladder, four baseline tiers, floor/profile, Campaign and publication.
Failed attempts remain visible. The compatibility inventory expresses explicit
engineering judgment; it does not infer runtime ownership from source syntax.
Runtime changes still require an invariant a legal Target cannot otherwise
express and the existing model-independence checks.

The canonical Registry owns Campaign discovery and uniqueness. Local state,
then published continuation, take precedence over a supplied new baseline.
The initial checkpoint, reference and objective/protocol must match a validated
activated anchor. The requested optimization budget must match the registered
Target policy; handoff does not silently replace it. A compatible existing
Campaign retains its portable incumbent and iterations. Changed checkpoint,
fixture, environment or rebaselined reference requires a context transition.
Fresh-clone imports require a newer validated segment even for equal assets.

Readiness requires actual Campaign and published facts, including a matching
trace segment, resume snapshot, context summaries and global index entry.
Campaign creation may survive a failed publication, but that failure cannot
produce READY_FOR_OPTIMIZATION. Handoff retries publication without replaying
experiments. Interrupted transitions retain the Campaign's explicit reconciliation
requirements. A local handoff lock serializes duplicate callers.

New handoff receipts require the currently validated segment, with no active
experiment or unresolved re-anchor. A new publication receipt also requires
completed publication of the current terminal ledger, including same-segment
iterations after its anchor. Historical reconstruction of completed
onboarding retains its original activated segment after later Campaign work.
Missing or inconsistent authoritative/published facts invalidate a new readiness
check until repaired through existing Campaign/results tools.

Legacy v1 records remain readable under their original semantics. Completed
v1 evidence is LEGACY_REVALIDATION_REQUIRED; it cannot be silently upgraded to
architecture identity or used to authorize v2 optimization. New initialization
requires an explicit v2 spec.

## Alternatives considered

- A claimed directory and BASELINED label cannot prove the underlying ledger,
  checkpoint validation or published continuation exists.
- Creating a new lineage on checkpoint change would lose compatible optimization
  history; attaching without revalidation would misattribute performance.
- Treating stage evidence as another deployment gate would duplicate acceptance
  policy. Onboarding instead binds existing evidence to ordered handoff facts.
- A generic model generator or source classifier would invent graph and
  ownership decisions; the Target and reviewed computation inventory own them.

## Consequences

The tool maintains an ignored evidence workspace, without model/weight hashes
or raw profiler copies. Published results remain owned by the existing publisher
and canonical renderer. The skill provides the v2 schema and evidence entrypoints
without moving Target-specific behavior into runtime.

## Verification

Targeted tests cover v2 field separation, machine-checkable model/checkpoint
contracts, old-record reading, false textual handoff rejection, actual Registry
creation, context activation and publication, duplicate CLI callers, interrupted
publication and replay-free recovery. Fresh Git-clone tests preserve snapshot
history, require a new anchor and reject historical-anchor bypass.
The fixtures use small CPU contracts and synthetic latency receipts. They do not
establish real Pi0.5/LingBot checkpoint compatibility, numerical parity, GPU
measurements or autonomous model onboarding.
