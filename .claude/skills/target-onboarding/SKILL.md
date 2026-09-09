---
name: target-onboarding
description: Integrate a model that does not yet exist in flash-vla as a Target, from immutable upstream and oracle evidence through compatibility, bring-up, correctness, baselines, floor/profile, and Campaign creation. Use when asked to add, port, onboard, or bring up a new VLA model or a new model/hardware Target. Do not use for optimizing an already registered Target or for writing one isolated kernel.
---

# Target onboarding

Turn a human deployment request into one model-specific Target while preserving
the runtime/Target boundary. The human supplies intent; the agent owns the
integration. Do not write an optimized implementation or a CUDA kernel until
the official oracle and compatibility report exist.

## Start

1. Read `ARCHITECTURE.md`, the upstream implementation,
   [the acceptance contract](references/acceptance-contract.md), and
   [the evidence contract](references/evidence-contract.md).
2. Resolve the upstream repository to an immutable commit and fill a JSON spec
   conforming to [the schema](assets/onboarding-spec.schema.json). Put every
   deployment value inferred from official configuration in `assumptions`;
   never silently guess a missing value.
3. Create the durable workspace:

   ```bash
   python -m lab.onboarding init artifacts/onboarding/<target> <spec.json>
   ```

4. Follow [the workflow](references/workflow.md) in its fixed order. Record a
   failed or blocked attempt before correcting it; the next attempt is appended.

The v2 spec separates the Target's architecture revision/signature and shape
from initial checkpoint provenance, ExecutionVariant and reference repository/commit.
A compatible fine-tuned checkpoint reuses the existing Target and Campaign.
Its new correctness and latency anchor belong to a new measurement segment.
Legacy v1 records remain readable but require explicit v2 revalidation.

## Boundary rules

- Model schema, weights, tokenization/processing, graph composition, fixed
  shapes, buffers, backends, costs, and plans belong to the Target.
- Modify `src/flash_vla/runtime/` only when a correct legal Target cannot
  express an invariant through the existing graph/runtime vocabulary. “More
  generic” is not a reason. A runtime change must be model-agnostic and retain
  Pi0/Pi0.5 smoke, dependency direction, and benchmark API behavior.
- During compatibility scan, classify every upstream computation. Do not write
  new CUDA. Use the report to choose existing runtime ops/components first,
  Target-local composition or ops second, and a runtime primitive only when the
  invariant above requires it.
- A correctness-ladder failure blocks performance attribution. An unavailable
  official optimized upstream route is recorded with a reason; it is not
  silently omitted.
- Measure baselines on the same frozen fixture and protocol. Report all four
  tiers; never claim speedup from only an upstream-eager comparison.
- Build floors from actual call-site geometry. Each new geometry cites its own
  measured ceiling; H100 alone does not make a Pi/Gemma ceiling applicable.

## Finish

The workflow is ready for autonomous optimization only when
`python -m lab.onboarding validate <workspace>` reports
`READY_FOR_OPTIMIZATION`. Readiness requires a real canonical Campaign, a validated anchor for the initial
checkpoint, and matching published trace/snapshot/context summary and discovery.
Use the handoff command in the workflow; a textual Campaign path is insufficient.
Publication failures are recoverable without rerunning experiments. At readiness,
the Campaign owns candidate selection and optimization evidence. Continue with
[optimization-campaign](../optimization-campaign/SKILL.md) for discovery, checkpoint
transitions and subsequent experiments.
