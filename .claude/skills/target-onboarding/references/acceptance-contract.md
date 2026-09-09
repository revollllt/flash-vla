# Acceptance contract

READY_FOR_OPTIMIZATION requires all ordered v2 stage evidence and a verified
Campaign/publication handoff. Existing acceptance policies own deployability;
this workflow does not introduce another deployment gate.

- Target architecture revision, inference signature and complete shape are explicit.
- Initial weights carry checkpoint ID and immutable provenance outside Target identity.
- ExecutionVariant and reference repository/commit are independently recorded.
- The model contract reproduces the Target signature.
- The observed checkpoint contract matches that signature before official-reference work.
- Official oracle outputs and the computation compatibility inventory exist.
- Target, loaders, registration and the full correctness ladder pass.
- All four baseline tiers and call-site floor/profile evidence are retained.
- The canonical Registry supplies the actual Campaign and registered budget.
- An activated segment binds the initial weights, reference, correctness and uninstrumented anchor.
- Published trace, snapshot, summaries, SVG and index retain that segment and lineage.

New checkpoint, fixture, environment or rebaselined reference changes retain a
compatible Campaign and require context validation before handoff. A re-anchor
consumes no optimization iteration and supplies no cross-context speedup.
A fresh clone seeds published history and must establish a newer validated anchor.

Legacy v1 evidence cannot establish these v2 conditions; its completed state is
LEGACY_REVALIDATION_REQUIRED. Do not infer architecture identity from its checkpoint.
