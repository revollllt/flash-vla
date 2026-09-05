# Autonomous Optimization Plane

Status: partial. The skills, the Agent Notes lifecycle, the PR rules and the
promotion gate script are implemented. The target-onboarding skill is
planned. Continuous integration is deferred and under discussion; nothing
here assumes it.

## Human role

The human defines accuracy requirements, ratifies the metric and framework
conventions, and sets the budget
([`30-acceptance-criteria.md`](30-acceptance-criteria.md)); chooses auto or
human mode per kernel task; and reviews promotion PRs. The human sets no
latency number: the objective is the gap to the structural floor derived by
the floor model ([`33-latency-floor-model.md`](33-latency-floor-model.md)).
No manual kernel tuning is required. In human mode the human may redirect between candidates;
directives are folded into the task's plan so the plan, not the transcript,
stays the source of truth.

## The flywheel

```text
Profile -> Analyze -> Design -> Implement -> Validate -> deploy -> Profile
```

Analyze starts from the gap decomposition: measured minus structural names
what kernel work can recover, structural minus roofline names what a change of
plan form can recover, per call site. Each round then verifies the smallest
evidence that can falsify the current hypothesis, in this order of cost:
analysis against measured constants first, then a controlled same-process
A/B, then a profile if the A/B needs localizing. Three sources of evidence stay independent: correctness reports,
latency reports on the real captured workload, and profiler diagnostics.
Profilers localize; the latency harness decides; correctness cannot be
overridden by either.

Before attributing a change, other causes are excluded: node and clock drift
(the A/B/A control leg), measurement regime (eager versus in-graph), and
identity mismatch (a different plan or shape). A conclusion drawn without
that exclusion is not recorded as a conclusion.

## Skills

| Skill | Owns |
|---|---|
| `kernel-design` | the entry point for any kernel work: contract, torch reference, parity, the candidate loop, promotion; the symptom-indexed sm90 wiki and compilable templates |
| `benchmark-kernel` | per-kernel timing method and the amortized in-graph regime |
| `hardware-unit-test` | measured machine constants with tags; the denominator under every floor |
| `gpu-profiler-analysis` | capture of Torch, Nsight Systems and Nsight Compute evidence |
| `ncu-report` | reading a Nsight Compute report into a named bottleneck and a next move |
| `target-onboarding` (planned) | bringing a new model or device up as a Target: model contract and reference, pipeline against the op table, buffer plan and segment list, cost declarations, reference backend, factory entry, acceptance entry |

Skills carry distilled, portable experience only. Evidence (job ids,
measurements, experiment history) lives project-side, in Agent Notes and in
per-task workspaces that are never committed.

Measured constants are a design dictionary, not a datasheet: every tile,
stage, CTA-count and fusion decision cites the constant's applicability
conditions, and a target set from an unreachable peak is a wrong target.

## Agent Notes

Notes are the cross-round memory. Accepted directions record the decision,
its applicability, consequences and verification. Rejected directions record
the candidate and the reason (occupancy collapse, register pressure, TMA
latency, lost PDL overlap, a floor shown unreachable). A rejection is an asset
of equal rank: it stops a later agent from repeating an expensive, invalid
experiment. Lifecycle and format follow
[`.agents/notes/README.md`](../../.agents/notes/README.md).

## Promotion gate

A script, not a service (`python -m eval.promotion_gate`). It reads the
Target's entry of the acceptance registry, runs the in-engine correctness
checks and the same-process A/B/A latency run through the generic harnesses,
computes the candidate's floor model as context, applies the gate semantics,
and writes one evidence record with a verdict: `pass`, `fail`, or `blocked`
when a gate could not run (a baseline adapter not installed). The
baseline-tier scripts run only when asked for, as subprocesses; not running
them blocks the verdict rather than passing it. A promotion PR attaches the
record; the PR rules in
[`.claude/rules/agent-notes-and-pr-workflow.md`](../../.claude/rules/agent-notes-and-pr-workflow.md)
govern the rest. Only a candidate that passes every gate enters the Target's
shipped plans; a failed candidate keeps its reason in a note and never touches
the production route. Whether and how this gate runs automatically is the
deferred CI discussion.

## Kernel task versus Target

The `kernel-design` skill works one kernel task at a time, each with its own
one-screen contract, baselines measured before any candidate, a floor from
measured tags, a promotion rule and a budget. The acceptance registry is
the outer criterion: a kernel task's promotion is necessary, and the Target's
gates decide whether the resulting plan ships.

## Hardware portability

A new device is a new Target. What transfers is the method, the runtime,
the floor model's form and this plane; what is rebuilt is the constant table,
the precision policy, the kernels and the execution plan. Megakernel,
persistent and dependent-launch forms are Target execution choices, adopted
or rejected on recorded evidence for that Target, never defaults.
