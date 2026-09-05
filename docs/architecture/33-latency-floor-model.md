# Latency Floor Model

Status: planned. A per-call-site roofline exists for one Target's decoder in
the benchmarks, hard-coded to its call sites and dividing by datasheet peaks;
it is replaced by the model described here.

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
| structural | roofline plus: launch count times measured grid ramp; the measured minimum latency of a kernel that occupies less than one wave; the cold-burst ramp of each weight stream; declared host slots; serial dependencies; minus the overlap a dependent-launch chain recovers | yes | the bound of the current execution plan's form |

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
precision policy: bytes read, bytes written, FLOPs, launch count, and the
wave count of its grid. The declaration lives with the call site in the
Target (it is the tensor table a kernel task contract already carries, made
machine-readable); the floor model contains no model or call-site names.
Segments and host slots come from the engine protocol.

## Validation

The model is validated like any other artifact:

- per call site, the predicted roofline and structural values are reported
  beside the measured in-graph time; a predicted structural floor above a
  measured time is a model error and blocks the model's use as an objective;
- when a constant is re-measured, the model version changes and every floor
  in every report carries the version;
- the promotion gate refuses a report whose floor model version does not
  match the plan it was computed for.

## Ownership

The model form is framework code under `benchmarks/` (planned); the cost
declarations belong to each Target's call sites; the constants belong to the
`hardware-unit-test` skill's per-architecture table. A new device is a new
constants table and the same model form.
