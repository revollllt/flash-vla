# Latency Floor Model

Status: partial. The model form exists (`python -m benchmarks floor`) with
both tiers, validation against measurement per segment and per call site
(through the profile's call-site attribution), versioning by the constants
file, and cost declarations for both Targets. The structural tier carries
the launch term only; the under-one-wave derating is counted and reported,
not yet modelled, so the structural floor is optimistic where such kernels
dominate.

## Purpose

The floor model produces the Target's latency objective. It is an analytic
cost model over measured hardware constants, not a human number and not a
cycle-level simulator: no usable cycle simulator exists for this hardware,
and building one would be a project of its own. The model is a versioned
artifact with its own validation; a floor that has not been checked against
measurement is a guess, not an objective.

## Two tiers

| Tier | Inputs | Depends on the plan | Answers |
|---|---|---|---|
| roofline | per call site: bytes moved and FLOPs, divided by the measured streaming rate and the measured tensor-core rate; the larger of the two | no | the physical bound of this model on this hardware at this shape and precision |
| structural | roofline plus: launch count times measured grid ramp (modelled); the measured minimum latency of a kernel that occupies less than one wave, the cold-burst ramp of each weight stream, declared host slots, serial dependencies, minus the overlap a dependent-launch chain recovers (planned terms) | yes | the bound of the current execution plan's form |

Every term divides by a measured, tagged constant from the
`hardware-unit-test` skill's constants table, cited by tag. A datasheet peak
never appears.

## The gap decomposition

```text
measured - structural   what kernel-level work can still recover
structural - roofline   what changing the plan's form can recover
                        (fusion, dependent launch, persistence)
```

Both differences are reported per segment and per call site. The
decomposition is the input to the Analyze step of the flywheel: it says which
lever applies before any experiment is run. The structural tier falls as the
plan's form improves; that is intended, because it measures what the current
form has left.

## Inputs from the Target

Each call site declares its cost as a function of the shape profile and the
precision policy: bytes read once, bytes written once, FLOPs, and how many
times the segment invokes it (`runtime/cost.py` types; the Target's
`costs.py` declares them and the engine exposes them as `costs`). The
declaration is a property of the call site's contract, not of a backend,
which is what keeps the roofline tier plan-independent. Launch count, grid
size and per-call-site in-graph time are backend properties and come from the
profile's attribution of one replay rather than from declarations. The floor model
contains no model or call-site names; segments and host slots come from the
engine protocol.

## Validation

The model is validated like any other artifact:

- per segment, the predicted roofline and structural values are reported
  beside the measured replay minimum, and per call site the roofline beside
  the in-graph time the profile attributes to it; a structural floor above a
  measured segment time, or a call-site roofline above its attributed time,
  is a model error, marks the report invalid and blocks the model's use as an
  objective. Under a dependent-launch chain the attributed time includes
  waiting time, so the per-call-site check is conservative there and the
  per-segment check against the replay minimum is the strict one;
- when a constant is re-measured, the model version changes and every floor
  in every report carries the version;
- the promotion gate refuses a report whose floor model version does not
  match the plan it was computed for.

## Ownership

The model form is framework code under `benchmarks/floor.py`; the cost
declarations belong to each Target (`costs.py`); the constants belong to the
`hardware-unit-test` skill's per-architecture table, located by the identity's
hardware axis. A new device is a new constants table and the same model
form.
