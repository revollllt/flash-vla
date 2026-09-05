# End-to-End Latency Evaluation

Status: partial. The generic runner (`python -m benchmarks latency`) measures
any Target through the engine protocol, emits the report schema below, reads
A/B/A deltas against the control spread and runs the noise calibration. The
older per-model harnesses remain for their profile checks; the Pi0.5 one
still carries datasheet-based stage floors, which the floor model replaces.
Floors in the generic report are empty until the floor model exists.

## Scope

One request, batch 1, the Target's fixed shape profile, closed-loop latency.
This is the whole scope. Throughput, tokens per second, batch or request-rate
sweeps and time-to-first-token are serving metrics for variable workloads and
do not apply: a VLA Target has no request distribution to summarize.

## Metrics

| Metric | Definition | Role |
|---|---|---|
| chunk latency | wall clock from inputs available to the action chunk available: input staging, host slots, every segment launch and replay | the headline; the candidate rule and the gap are read on it |
| device latency | CUDA-event time around the same forward | uncontended device floor; its spread exposes host stalls |
| host time | wall clock of each declared host slot | localizes host intrusion |
| segment latency | each segment replayed alone, back to back, no host gap | the diagnostic against per-segment floors |
| overhead | chunk latency minus the sum of segment latencies | what staging, host slots and launches cost |

All metrics are reported per run. No metric carries a human-chosen bound;
the objective is the gap to the structural floor of
[`33-latency-floor-model.md`](33-latency-floor-model.md), and promotion of a
candidate follows the candidate rule in
[`30-acceptance-criteria.md`](30-acceptance-criteria.md).

## Statistics

Every metric reports `min`, `median` and `p99`.

- `min` is the uncontended floor and the regression detector: it is the
  statistic that reproduces across jobs on shared, unlocked-clock nodes.
- `median` is the expected latency under the run's node conditions.
- `p99` is the tail. In closed loop a late chunk is a missed control tick, so
  the tail is a deployment property and is reported under deployment
  conditions: clocks unlocked, no isolation beyond what the scheduler gives.
  Reporting `p99` requires a repetition count at which the percentile is not
  simply the maximum; the acceptance registry states the count.

Clocks are never locked for any measurement, because deployment does not lock
them. The consequence is accepted: `median` and `p99` carry node and clock
drift, and only same-process comparisons cancel it.

## Comparison discipline

- **Deltas come from same-process A/B/A.** Both plans are built and measured
  in one process, the control leg is measured twice, and the control leg's
  two measurements must agree; a delta is the difference between the
  candidate and the control legs of that run. Cross-job comparisons are
  informational.
- **Noise calibration.** A calibration run measures the same plan as every leg
  of an A/B/A and reports, per statistic, the spread between legs. That spread
  is the minimum detectable effect for this Target on this node class. A
  claimed improvement smaller than it is "indistinguishable" and is recorded
  as such; it never promotes a candidate.
- **Identity gate.** A comparison between reports whose identity blocks differ
  in target, shape profile, precision policy or plan is refused.
- **Profiles do not decide.** Profiler runs carry their own overhead and are
  reported separately with that overhead stated; they localize, the harness
  decides.
- **The floor is context for the gap, not a gate.** A candidate is judged by
  the A/B/A delta; the floor says how much is left and where.
- **Kernel-level numbers** belong to the `benchmark-kernel` skill and are read
  in the amortized in-graph regime; an eager per-kernel time is not evidence.

## Report schema

One JSON document per run:

- `identity`: target, model revision, shape profile with its numbers,
  precision policy, plan, git revision, device name, driver, framework and
  toolchain versions, node, job id.
- `config`: repetitions, warmup, workload seed, prompt, whether this run is a
  calibration or an A/B/A and which leg.
- `metrics`: for each metric of the table above, `min`, `median`, `p99`, the
  repetition count and, for segments, the segment name.
- `floors`: per segment and per call site, the roofline and structural
  values, the tags each divides by, the floor model version, and the two gap
  terms of the decomposition.
- `calibration`: when present, the per-statistic minimum detectable effect.
- `verdict`: absent from a harness report. Verdicts are produced only by the
  promotion gate against the acceptance registry.

## Generic runner

The runner constructs the engine through the engine protocol
([`10-runtime.md`](10-runtime.md)) via the Target factory registry
(`benchmarks/targets.py`, the one place Targets are named), takes the segment
list and host slots from the engine, generates inputs from the Target's
seeded sampler, and emits the schema above. Legs run in the order given; a
leg whose plan repeats the first leg's is a control leg, the spread between
control legs is the minimum detectable effect, and every delta is marked
distinguishable or not against it. Per-Target code is the factory, the input
sampler and the acceptance entry; the runner contains no model or stage
names.
