# Workflow

This reference applies only to explicitly requested legacy `lab.onboarding`
records/handoffs. Ordinary Target bring-up follows [SKILL.md](../SKILL.md).

The authoritative workspace is onboarding.json plus append-only stage receipts.
state.json is derived. New work uses the v2 spec; preserve old v1 evidence without
interpreting its checkpoint-based revision as architecture identity.

## Requirement, reference and compatibility

Initialize with python -m lab.onboarding init WORKSPACE SPEC.json. The spec
records the Target, initial_weights, execution_variant, reference, deployment,
objective, correctness requirements, protocol, deployment bound, registered
optimization budget and evidence-backed assumptions.

The initial stages are requirement_freeze, reference_freeze, model_contract and
weights_compatibility. The first is recorded by init; record the others with
python -m lab.onboarding record WORKSPACE --stage STAGE EVIDENCE.json.
The model contract and observed checkpoint contract must reproduce the Target's
inference signature. A mismatch stops that Target's onboarding.

## Official reference and computation inventory

Before optimized code exists, establish upstream loading, tokenizer, processor,
fixtures, seed/noise, reference outputs and environment. Run upstream eager and
its official optimized route when available; retain a concrete unavailability
reason otherwise. The required checks are in evidence-contract.md.

At compatibility_scan, inventory upstream computations and run
python -m lab.onboarding compatibility-report WORKSPACE INVENTORY.json.
Classify existing runtime ops/components, Target-local composition/ops, new
reusable components, genuine runtime primitives, host work, capture blockers
and unsupported work. Record dynamic shape, allocation, synchronization,
capture and control-flow risks. Review the nine derived coverage/reuse answers.

## Bring-up and correctness

target_bring_up covers model schema, loaders/processors, Target, pipeline,
plans, backend and registrations. Runtime changes require an invariant a legal
Target cannot express and the five boundary checks in evidence-contract.md.

correctness_ladder retains this order: declaration smoke, op/component parity,
stage parity, shallow model, full layers, single denoise step, full denoise
loop, official end-to-end parity. A failure prevents performance attribution.

## Baselines and profile

baseline_ladder records upstream eager, upstream official optimized/compile
(or explicit unavailability), Flash-VLA reference plan and initial shipped plan.
Use the same fixture and protocol and consistently exclude setup/compilation.

floor_profile records actual call-site geometry, minimal bytes/FLOPs, measured
ceilings for those geometries, recoverable latency and the diagnostic profile.
These observations guide later optimization; profiler timing is not a latency claim.

## Campaign handoff

Run python -m lab.onboarding handoff WORKSPACE BASELINE.json --root REPOSITORY
--source-input PATH. Repeat --source-input for the committed portable source set.
BASELINE.json is normalized validated anchor evidence, with correctness and full
MeasurementContext including the declared reference provenance.

The command restores local or published Campaign history before considering a
new baseline. New contexts use --transition-request REQUEST.json and the
Campaign's compatibility, rebuild/retune, correctness and re-anchor operations.
Fresh-clone imports always require a new anchor; lineage and iteration IDs persist.

Campaign creation and publication are distinct durable stages. A failed publish
leaves readiness pending; retry handoff to publish without repeating experiments.
An interrupted transition requires campaign-resume/reconciliation first.
python -m lab.onboarding validate WORKSPACE reports READY_FOR_OPTIMIZATION only
after the actual handoff facts and published discovery agree. Continue candidate
work through lab.optimize and its recorded execution checkout.
