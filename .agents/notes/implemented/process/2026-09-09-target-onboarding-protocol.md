# Agent Note: evidence-first Target onboarding

Status: implemented

## Problem

The repository documented what a completed Target contains, but not how an
agent should get an unfamiliar upstream model there. Starting with kernels
would lose the official oracle and obscure whether missing behavior belonged
in an existing component, the Target, or the runtime. Deployment values not
given by the human also had no durable place to expose their provenance.

## Decision

Add the `target-onboarding` skill and a small `lab.onboarding` evidence tool.
The workflow fixes only the dependency order: requirement and upstream freeze,
official reference, compatibility scan, Target bring-up, correctness ladder,
four-tier baseline, geometry-specific floor/profile, then Campaign creation.
Unsuccessful stage attempts are appended and remain visible; derived state
blocks later stages until the current one passes.

The machine-readable spec records the upstream repository/commit, checkpoint,
model revision, hardware/deployment configuration, precision, complete shape
profile, objective, correctness requirements, benchmark protocol, deployment
bound, optimization budget, and explicit assumptions. It adds no hashes.

Compatibility is a reviewed computation inventory, not an inferred source
parser. The tool renders the JSON and Markdown summaries from that one
inventory and answers the required coverage, reuse, Target-local operation,
runtime-primitive, dynamic-shape, allocation, synchronization, capture, and
control-flow questions. This keeps classification judgment explicit while
preventing the two report formats from drifting.

Runtime modification remains exceptional: the evidence must name an invariant
that a correct legal Target cannot express and show model independence, Pi0
and Pi0.5 smoke, dependency direction, and benchmark API compatibility.

## Alternatives considered

- Generate a generic model package from the spec: rejected. Model graphs,
  weights, tokenization, and shapes are model semantics and belong in the
  Target; a generator would invent an abstraction before a second use.
- Parse upstream Python to classify operations automatically: rejected. Static
  syntax cannot reliably decide capture behavior or runtime ownership, and a
  confident wrong classification is worse than an explicit reviewed inventory.
- Add the sequence to the deployment gate: rejected. It is onboarding evidence
  and stage ordering; the acceptance registry already decides deployability.
- Create a real third model in this change: deferred to the following LingBot
  reference bring-up so the protocol and the integration remain separate,
  reviewable changes.

## Consequences

`python -m lab.onboarding` creates and validates an ignored onboarding
workspace, renders compatibility reports, retains failed attempts, and hands
off only to a recorded Campaign. The skill tells the agent where model-specific
files and registrations belong without modifying runtime by default.

The protocol test uses a fictitious previously unseen model to exercise the
complete state transition and contract. That proves the orchestration tool,
not real-model compatibility; the LingBot bring-up must supply the first real
upstream, oracle, and GPU evidence.

## Verification

- `python -m unittest eval.tests.test_target_onboarding -v`: nine focused tests
  cover spec fields, identity matching, oracle completeness, nine compatibility
  answers, conditional runtime evidence, retained correctness failure, fixed
  correctness/baseline order, geometry-specific ceilings, and Campaign handoff.
- `quick_validate.py .claude/skills/target-onboarding`: skill structure and
  frontmatter valid.
- `python -m unittest discover -s eval/tests -p "test_*.py"`: 100 tests pass,
  one existing optional-renderer test skipped.
