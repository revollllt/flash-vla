# Evidence contract

All paths below are evidence pointers, not copied artifacts. Do not add hashes.
Use `python -m lab.onboarding record <workspace> --stage <stage> <evidence.json>`.

Every evidence object has `status: passed`. To retain an unsuccessful attempt,
use `failed` or `blocked` plus `reason`; correct it in a later appended attempt.

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

## Campaign creation

Record non-empty `directory`, `objective`, `protocol`, `fixture`, and `state`.
The Campaign ledger remains authoritative for later optimization.
