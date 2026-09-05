# Acceptance Criteria

Status: partial. The registry exists (`eval/acceptance.py`) with the
framework defaults, the bf16 tolerances every shipped gate reads, and one
entry per Target. The derived latency objective, the generic runners and the
promotion gate do not exist yet.

## What the human defines

The human defines three things and no performance number:

1. **Accuracy requirements**: the precision policy a Target may use, which
   oracle tiers are required, and any override of the framework's default
   tolerances.
2. **Metric and framework conventions**: which metrics and statistics are
   reported, the repetition counts, the comparison discipline, and which
   checks gate. These are ratified once and apply to every Target.
3. **Budget**: how many candidates, how many jobs or how much wall clock a
   Target's optimization may consume before it stops.

The latency objective is not a human input. It is derived from the Target's
floor model ([`33-latency-floor-model.md`](33-latency-floor-model.md)) and
the optimization closes the gap to it. A human-chosen millisecond bound would
be arbitrary and unfalsifiable; a floor derived from measured constants can be
checked against measurement and revised when a constant is.

## One registry, not one file per Target

Acceptance is data in one device-free module, `eval/acceptance.py`, on the
precedent of the plan registry: importable offline, no torch, no device.

- **Framework defaults** hold everything that is the same for every Target:
  the metric set, the statistics, repetition counts, the mandatory gates, the
  comparison structure, the noise policy, and the tolerance defaults keyed by
  precision policy (the tolerance for a `bf16` Target is a property of `bf16`,
  not of the model).
- **Per-Target entries**, keyed by Target name, hold only what differs:
  budget, and rare overrides (a missing baseline adapter disables that gate;
  a precision policy other than the default selects its tolerance set).

A Target without an entry runs every correctness gate and every latency
metric; only its budget is undefined, so it cannot enter the promotion gate.
Adding a Target to the framework is one dictionary entry. Numbers that the
model spec or the engine already hold are never restated here.

## Schema

### defaults.latency

- **metrics and statistics**: the five metrics of
  [`31-latency-evaluation.md`](31-latency-evaluation.md), each with `min`,
  `median` and `p99`, all reported.
- **conditions**: clocks unlocked, repetition and warmup counts, the minimum
  repetition count under which `p99` is reported at all.
- **noise policy**: the calibration run that establishes the minimum
  detectable effect per statistic; a delta below it is indistinguishable.
- **objective**: the structural floor from the Target's floor model; the
  reported objective quantity is the gap between measured and structural per
  segment and in total.
- **candidate rule**: a candidate is promotable on latency when, in a
  same-process A/B/A, its chunk latency `min` improves by more than the
  minimum detectable effect and neither `median` nor `p99` regresses by more
  than theirs.

### defaults.correctness

An ordered list of checks. Each names the script, the oracle tier, the
configuration (steps, depth, oracle injection point, weights: random or
checkpoint), the metric thresholds by precision policy, and the mode.
Mandatory gates for every Target:

- replay determinism: two replays of the candidate are bit-identical;
- finiteness of every output and of every padded region a kernel touches;
- the shallow structural comparison against the in-engine reference route
  (single step, single layer or layer 0);
- the layer-0 comparison against the official baseline where an adapter
  exists.

Deep and multi-step comparisons are `report`; see the comparison rules in
[`32-correctness-evaluation.md`](32-correctness-evaluation.md).

### defaults.policy_quality

A placeholder in this phase: the suite name and threshold slots exist and are
empty. A precision policy other than `bf16` cannot be promoted while this
section is empty.

### targets.<name>

- **budget**: candidate cap, consecutive-non-improvement cap, job or
  wall-clock budget.
- **overrides**: any deviation from the defaults, each with a one-line
  reason.

## Stop condition

A Target's optimization stops at the first of:

1. the remaining gap between measured and structural floor is attributed:
   every residual term names a measured constant and carries evidence, so
   nothing left is a tuning question;
2. the remaining blockers are explicit and outside the Target (a constant not
   yet measured, a dependency elsewhere);
3. the budget is exhausted.

"Reaching the floor" is not a stop condition, because a floor is a bound and
not a destination. Auto mode returns only at a stop condition, with the
evidence pack.

## Gate semantics

- A `gate` failure blocks promotion. A `report` result is recorded in the
  evidence and in the Agent Note.
- Latency never overrides a correctness gate.
- A gate whose measurement is not admissible (identity mismatch, missing
  calibration, missing plan stamp, floor model out of date for the plan)
  counts as failed, not as skipped.
- The promotion gate ([`40-optimization-plane.md`](40-optimization-plane.md))
  is the only consumer that turns this registry into a pass/fail; harnesses
  only produce reports.
- Changing a default is a framework decision and is its own PR with its own
  Agent Note; changing a Target entry is a Target decision.
