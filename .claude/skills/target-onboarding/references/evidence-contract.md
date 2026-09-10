# Evidence contract

This reference applies only to explicitly requested legacy `lab.onboarding`
records/handoffs. Ordinary Target bring-up follows [SKILL.md](../SKILL.md).

All paths below are evidence pointers, not copied artifacts. Do not add hashes.
Use `python -m lab.onboarding record <workspace> --stage <stage> <evidence.json>`.

Every evidence object has `status: passed`. To retain an unsuccessful attempt,
use `failed` or `blocked` plus `reason`; correct it in a later appended attempt.

## Identity and checkpoint compatibility

reference_freeze records repository and commit matching spec.reference.
model_contract records a contract with architecture, parameter_shapes,
weight_layout, io_contract and control_flow; the existing inference_signature
function must reproduce spec.target.inference_signature.

weights_compatibility records the observed checkpoint contract in the same
format, plus weights.checkpoint_id and weights.checkpoint_digest matching
initial_weights. Derive this from the checkpoint's actual schema and inference
configuration, including alias/layout/control-flow semantics. Copying the
expected contract is not compatibility evidence. Weight values and source
locations never enter the architecture signature.

These metadata receipts do not replace the subsequent numerical correctness
ladder. Retain source/probe evidence for the observed contract; incompatible
weights require a compatible Target/model revision.

## Official reference checks

`checks` contains all names below. A passed check is
`{"status":"passed","evidence":"<path or command/result>"}`.

```text
upstream_load
checkpoint
tokenizer
processor
correctness_fixture
performance_fixture
seed_noise
upstream_eager
upstream_official_optimized
reference_outputs
upstream_environment
```

Only `upstream_official_optimized` may instead be
`{"status":"unavailable","reason":"<why>"}`.

## Target bring-up checks

Use the same passed-check object for:

```text
model_schema, weight_loader, tokenizer_processor_contract,
target, pipeline, reference_plan, shipped_plan, registry,
backend_implementation, benchmark_registration, acceptance_registration,
official_baseline_adapter, smoke
```

Also record:

```json
{"runtime_modification":{"required":false,"reason":"existing vocabulary covers the Target"}}
```

If `required` is true, add `unexpressible_invariant` and passed checks for
`model_agnostic`, `pi0_smoke`, `pi05_smoke`, `dependency_direction`, and
`benchmark_api`.

## Correctness and baseline ladders

Each `ladder` item has `name`, `status`, and `evidence`, in the exact order in
`workflow.md`. Only the upstream official optimized/compile baseline may be
`unavailable` with a reason.

## Floor/profile

`call_sites` is non-empty. Every item records:

```json
{
  "name": "call site",
  "geometry": "fixed geometry ID",
  "minimal_bytes": 0,
  "flops": 0,
  "measured_ceiling": {
    "geometry": "same fixed geometry ID",
    "metric": "throughput metric",
    "value": 1,
    "unit": "metric unit",
    "evidence": "measurement artifact"
  },
  "recoverable_latency_ms": 0
}
```

The stage also has a non-empty `profile` evidence pointer.

## Campaign creation and publication

Use lab.onboarding handoff with a normalized baseline produced by the existing
report converter, the results repository and declared committed source inputs.
The command records repository, canonical directory, campaign_id and segment
from the actual ledger. Manual passed receipts must satisfy the same checks.

The anchor must bind the initial checkpoint, fixed objective/protocol and the
declared reference provenance. Publication evidence additionally requires
consistent Target/context summaries, trace and resume snapshot, the canonical
SVG, and the matching global discovery entry.

An existing local or published Campaign is reused. If its active checkpoint,
fixture, environment or reference differs, supply a Campaign transition request
for the requested context; inherited source and recipes remain Campaign-owned.
Fresh-clone imports always require a new validated segment.
An interrupted or failed transition uses explicit Campaign reconciliation,
not a silent onboarding rerun. A publication-only failure can retry handoff.

New handoff receipts require the currently validated segment with no active
experiment or unresolved re-anchor. A new publication receipt requires all
current terminal history to be published before readiness. Historical validation of completed onboarding
retains its original handoff segment after later Campaign work. Missing or
inconsistent publication requires existing results repair/publish commands.
