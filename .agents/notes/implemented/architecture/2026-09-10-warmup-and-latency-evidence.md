# Agent Note: Warmup and latency drift evidence

Status: implemented

## Problem

A fixed ten-second load period did not prevent the same LingBot parent in job
610937 from measuring 90.052 and 79.529 ms. Summary-only uninstrumented reports
discarded temporal samples and recorded application clocks rather than actual
operating clocks, so the historical cause cannot be established from those
reports. Re-anchoring isolates comparisons but does not diagnose the drift.

## Decision

Warmup remains required before every timed loop. Fixed-time soak is optional and
defaults to zero; explicit soak duration remains in each report. This partially
supersedes the mandatory-soak decision in
[acceptance is deployability](2026-09-06-acceptance-is-deployability.md).
The ten-second historical reports remain valid inputs. The other sampling,
correctness, A/B/A and noise requirements are unchanged.

Uninstrumented reports retain the already-collected latency samples in temporal
order, with device-state observations at each leg's boundaries. These observations
reuse the existing environment queries outside timed loops. They do not enter
stable context identity or introduce frequency locks, extra samplers or gates.
The existing optional attribution collector supplies continuous telemetry when a
diagnostic run is needed.


Each CUDA graph is now permanently paired with its own capture stream. Replay
orders the caller's producer and consumer work around that stream using reusable
events; changing the caller does not move the graph. The shared graph mechanism
covers inference, runtime timing, and the existing kernel-timing entry points.
A/B/A recaptures fresh graph/stream pairs before each leg, retaining model weights
and static buffers. No stream-affinity gate or content hash is introduced.

## Alternatives considered

Increasing the soak duration would add cost without explaining the recorded
state change. Globally locking GPU clocks would change the deployment regime.
Assigning a node-change cause from the hostname alone would overstate the evidence.

## Consequences

Warmup-only measurements avoid the extra load period. Boundary observations can
miss a transient inside a leg; raw samples preserve its latency signature, and
continuous attribution is still needed to identify such a transient's cause.
No historical speedup or published Campaign data is recalculated.

## Verification

Job 611453 compared the same qualified RoPE-table engine, real checkpoint and
fixture on one H100, with soak 0/10 seconds and warmup 5/50. All six chunk minima
fell within 0.022 ms; GPU samples observed 1980 MHz throughout and no active clock
event reasons. Each ten-second soak consumed approximately 11.2 wall seconds.
This supports the default change for the observed regime, not a claim that the
older 570-driver events have been explained.

Evidence: `artifacts/latency-drift/611453/` and the same-source paired follow-up
under `artifacts/latency-drift/611471/`. CPU checks cover raw-sample ordering,
runtime observations staying outside context identity, optional soak propagation
and historical report compatibility. The causal diagnosis remains open. Job 611471 reproduced about 11 ms of
same-engine drift on driver 570.86.10 while sampled SM/HBM clocks stayed
constant. Job 611488 localized the observed action-expert difference mainly
to intervals between graph kernels, with profiler perturbation still a caveat.

Job 611586 compared the original A/B/A flow and an independent process that
additionally warmed both engines before timing, on the same allocated H100.
Neither reproduced the drift: A1/A2 median changes were -0.019895 ms and
+0.003944 ms respectively. This does not establish prewarming as a fix. All
1,418 GPU observations recorded SM 1980 MHz, HBM 2619 MHz and zero active clock
event reasons, without another compute PID on the selected GPU. Evidence and
limitations: `artifacts/latency-drift/611586/analysis.md`. No graph or kernel
implementation was changed by these diagnostic probes.


Further same-engine diagnostics found a repeatable trigger, not yet a repair.
Job 611664 stayed near 87.08 ms for 600 samples with no B constructed.
Jobs 611690 and 611708 changed from about 87 ms to 98 ms after the first replay
of the same graph on a second stream, retaining the slower state after returning
to the original stream. Job 611708 used continuous telemetry and an unchanged-
stream control; the pre-switch gap was only 0.291 ms. Job 611718 reproduced
the transition on the same physical GPU with CUDA_DEVICE_MAX_CONNECTIONS=1,
so that setting is not a remedy. The relation to the original reverse 98-to-87
A/B/A transition is still unproved. Full evidence and limitations are in
`artifacts/latency-drift/drift-investigation-2026-09-10.md`. The pending graph-recapture
control (611731) was cancelled after the user selected fixed-stream ownership.
This avoidance policy does not establish a complete driver-level causal model.

The real-checkpoint validation (611837, ACD1-21, driver 610.43.02) passed three
GPU ownership/ordering tests and exact eager/graph output comparisons for both
A and B, including callers on another stream. A/B outputs and all three
post-measure outputs matched exactly. The real latency harness captured nine
new graph/stream pairs across A/B/A; medians were 91.799087, 86.641164, and
91.773142 ms, with control minimum spread 0.016951 ms. CPU checks passed
68 tests and three subtests. Evidence: `artifacts/latency-drift/611837/`.

Recapture exposed a pre-existing LingBot module-global RoPE binding conflict:
a second policy replaced the function while the first policy still cleared its
own tables. Each policy now binds its own RoPE function during eager/capture
execution and restores the previous binding afterwards. This is required for
recapture while retaining both models; it does not change RoPE arithmetic.
