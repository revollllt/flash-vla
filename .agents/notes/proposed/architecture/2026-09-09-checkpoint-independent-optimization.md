# Agent Note: checkpoint-independent optimization identity

Status: proposed

## Problem

The legacy Identity v2 treated checkpoint revisions as model revisions, and
legacy Campaigns fixed fixture identity. Compatible fine-tuned weights therefore lose
optimization lineage, while removing those comparisons without replacing them
would permit false cross-checkpoint speedups. This proposal replaces the
checkpoint identity decision in the
[existing runtime boundary note](../../implemented/architecture/2026-09-06-runtime-target-boundary-and-acceptance-first.md).

## Proposal

Target identity describes hardware, inference-compatible architecture and shape.
A machine-checkable inference signature covers semantic topology, parameter ABI,
I/O and control flow, independently of weight values, execution plans and paths.
Targets own the architecture revision. ExecutionVariant separates high-level
quality/performance strategies from kernel candidates.

Campaign identity combines that workload with objective and measurement protocol.
Weights and fixture identify a context; comparable environment facts additionally
identify a measurement segment. Node, observation time and reference commit are
retained provenance, not automatic new Target identities. Context transitions
must validate compatibility and incumbent applicability, then revalidate
correctness and re-anchor latency before accepting comparisons.

Published continuation preserves optimization lineage and portable implementation
without claiming prior checkpoint measurements apply to a new checkpoint.
Legacy v2 identity remains visibly legacy until an explicit known-Target mapping
and compatible signature are available.

## Alternatives considered

- Keep checkpoint in Target: safe comparisons but incompatible with portable lineage.
- Drop checkpoint/fixture checks: loses the measurement comparability boundary.
- Derive signature from candidate kernels: turns ordinary optimization into a new Target.
- Hash all files or weight values repeatedly: unnecessary cost; existing immutable
  provenance identifies weights, while the signature represents only the inference ABI.

## Acceptance criteria

The v3 identity matrix, explicit migration, runner producers and context changes
must be tested independently of GPU latency. Identity semantics and the
[transition ledger boundary](../../implemented/process/2026-09-09-persistent-campaign-ledger.md)
are implemented. Automatic Campaign discovery, publication, fresh-clone recovery
and real checkpoint drills remain required in their planned phases. Random-weight tests do not establish real-checkpoint transfer.

## Risks

The folded Pi0.5 runtime weights are not the original checkpoint ABI. Compatibility
must distinguish source weight schema from rebuild-derived tables. Layer bisection,
chunk and denoise count are shape configuration and must not silently rewrite the
architecture revision. Measurement-only provenance cannot be dropped when the
identity is migrated.
