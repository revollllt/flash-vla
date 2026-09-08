# Workflow

Use one evidence JSON per command. The authoritative workspace is append-only
stage evidence plus `onboarding.json`; `state.json` is derived and may be
rebuilt with `python -m lab.onboarding validate <workspace>`.

## 1. Requirement freeze

`init` validates and records the machine-readable spec. Resolve user input into:

- upstream repository and immutable commit;
- checkpoint ID/source and model revision;
- hardware and deployment configuration;
- precision and complete fixed shape profile;
- performance objective, correctness requirements, benchmark protocol, and
  deployment bound;
- optimization budget;
- explicit assumptions for values derived from official deployment config.

Do not proceed while a deployment-affecting value is unknown. Ask the human
only when official source and repository evidence cannot resolve it.

## 2. Upstream freeze

Record the repository, commit, checkpoint ID, and model revision. They must
equal the spec. Preserve the upstream environment at the official-reference
stage rather than copying it into flash-vla.

## 3. Official reference

Before optimized code exists, load the upstream and checkpoint; freeze the
tokenizer, processor, correctness fixture, performance fixture, seed/noise,
reference outputs, and environment. Run upstream eager. Run the official
optimized/compile route when available; otherwise record `unavailable` and the
concrete reason. Without saved reference outputs, stop.

## 4. Compatibility scan

Inventory every upstream computation using only these classifications:

```text
existing runtime op
existing shared component
Target-local composition
Target-local op
new reusable component
genuine runtime primitive
host work
capture blocker
unsupported
```

Each item has `name`, `category`, source `evidence`, proposed `action`, and any
observed risks from: `dynamic shape`, `post-freeze allocation`, `host/device
sync`, `graph capture blocker`, `data-dependent control flow`.

```bash
python -m lab.onboarding compatibility-report artifacts/onboarding/<target> inventory.json
```

This writes both `compatibility-report.json` and `compatibility-report.md` and
records the stage. Review the nine derived answers before implementation.

## 5. Target bring-up

Add only the model-specific files needed under:

```text
src/flash_vla/models/<model>/
src/flash_vla/hardware/nvidia/<hardware>/<model>/
```

Implement the model schema, weight loader, tokenizer/processor contract,
`target.py`, `pipeline.py`, reference and shipped plans, registry entry, and
backend. Register the benchmark target, acceptance policy, official baseline
adapter, and smoke. Keep the initial shipped plan conservative; aggressive
optimization belongs to the Campaign.

If runtime changes were required, identify the existing invariant that could
not express the legal Target and record all five runtime checks from the
evidence contract.

## 6. Correctness ladder

Run and record exactly in this order:

```text
declaration smoke
op/component parity
stage parity
shallow model
full layers
single denoise step
full denoise loop
official end-to-end parity
```

Stop at the first failure. Do not profile or interpret latency while numerical
behavior is unresolved.

## 7. Baseline ladder

On the same frozen performance fixture and protocol record:

1. upstream eager;
2. upstream official optimized/compile, or explicit unavailability;
3. flash-vla reference plan;
4. flash-vla initial shipped plan.

Exclude setup and compilation according to the frozen protocol consistently.

## 8. Floor/profile

For each real call site record geometry, minimal bytes, FLOPs, a ceiling
measured for that geometry with evidence, and recoverable latency. Record the
profile artifact used to attribute the gap. A new geometry requires a new
ceiling measurement.

## 9. Campaign creation

Create the persistent Campaign with the spec's objective, protocol, fixture,
and registered Target budget. Record its directory and initial state. Validate
the onboarding workspace, then hand candidate work to `lab.optimize`.
