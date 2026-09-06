# Agent Note: runtime/Target boundary, and acceptance criteria before optimization

Status: implemented

## Problem

The repository's stated architecture (a small model-agnostic runtime under a
Target that owns every specialization, driven by an agent optimization plane)
was only partly represented by the code, and the missing parts were the ones
that decide whether a new VLA model or device can be brought up quickly:

- The runtime was ScratchPool, graph capture and in-graph timing. Buffer
  allocation, the warmup/freeze/capture lifecycle, input staging and host
  slots were written separately in the Pi0 engine (one graph) and the Pi0.5
  engine (three graphs), and the two duplicated each other.
- Latency evaluation was two divergent per-model harnesses with different
  report shapes, and the Pi0.5 stage floors divided by datasheet peaks,
  contrary to the change-gates rule that a floor divides by a measured tag.
- Correctness evaluation had five metrics duplicated across scripts and no
  shared module.
- There was no acceptance criterion per Target and no derived latency
  objective; the flywheel had no stop condition beyond per-kernel task
  contracts, and no harness could turn a report into a verdict.
- A draft design proposed that the runner allocate per-op buffers by reading
  the op table, that VLA latency be evaluated with LLM-serving metrics, and
  that a human choose a millisecond bound per Target.

## Decision

1. **Boundary rule.** What is invariant across Targets lives under
   `src/flash_vla/runtime/`; all else in the Target. *Amended by the
   [explicit graph and ModelRunner note](2026-09-06-explicit-graph-and-model-runner.md):*
   a Target now declares its computation graph (buffers, nodes, host slots)
   with the graph API and subclasses the `VLA` template; `ModelRunner` is the
   one implementation of the `Engine` protocol, owning materialization, plan
   binding, workspace allocation, the warmup -> freeze -> capture -> replay
   lifecycle and the derived costs; `Identity` carries the four axes, the
   shape numbers, the resolved plan and the git revision (table options were
   retired with the fused overlay becoming a backend route).
2. **Acceptance as one registry.** `eval/acceptance.py` holds framework
   defaults (metrics, statistics, repetition policy, unlocked clocks,
   same-process A/B/A deltas, the candidate rule, the ordered correctness
   checks with gate/report modes, tolerances keyed by precision policy) and
   one entry per Target holding only budget, baseline-tier scripts and
   overrides. Parity scripts read their thresholds from it. Precision policy
   is an identity axis with `bf16` its only value; a policy other than
   `bf16` cannot be promoted while the policy-quality slot is empty.
3. **Latency for VLA.** `benchmarks/latency.py` reports chunk, device, host,
   segment and overhead latency with `min`, `median` and `p99`, clocks
   unlocked because deployment does not lock them; deltas come only from
   same-process legs, the control-leg spread is the minimum detectable
   effect, and a delta below it is indistinguishable. No metric carries a
   human-chosen bound.
4. **Derived latency objective.** `benchmarks/floor.py` computes, from the
   Target's cost declarations (`runtime/cost.py`, per-Target `costs.py`) and
   the hardware-unit-test constants (cited by tag, versioned by the
   constants file), a plan-independent roofline tier and a plan-dependent
   structural tier (launch count from a profiler trace times the measured
   ramp), validates the structural floor against the measured segment, and
   decomposes the gap into kernel-level and form-level parts. The stop
   condition of a Target's optimization is attribution of the residual gap or
   budget exhaustion, never "reach the floor".
5. **Correctness structure.** One metrics module (`eval/correctness/
   metrics.py`); `eval/correctness/in_engine.py` compares any candidate plan
   or table option against the Target's reference in lockstep over the
   program, on the declared stage outputs, with per-layer profiles over the
   active depth, padded-allocation finiteness, oracle injection and replay
   determinism; shallow runs gate, deep runs report. The official-baseline
   tier stays per model and is registered per Target.
6. **Promotion gate.** `eval/promotion_gate.py` runs the registry's checks
   and the A/B/A through the generic harnesses, computes the floor as
   context, and writes one evidence record with a verdict of pass, fail or
   blocked.
7. **Documentation as a tree.** `docs/architecture/` owns the normative
   architecture with a status line per document; `ARCHITECTURE.md` is the
   index. *Superseded in part:* the tree predates the explicit graph and is
   being collapsed into one `ARCHITECTURE.md` page; until then the source
   named there is authoritative where the two disagree.

## Alternatives considered

- Runner infers per-op buffers from the op table: rejected. Padding,
  masking, aliasing and the KV-cache layout are Target decisions that fusion
  changes; inferring them makes the runtime a compiler and removes the
  agent's control over layout.
- Copy TTFT/TPOT/throughput from LLM serving: rejected. A batch-1
  fixed-shape closed-loop workload has no request distribution; the
  analogues are chunk latency and its tail.
- Lock clocks to make `median` and `p99` reproducible: rejected by the
  owner. Deployment runs unlocked; the harness compensates with A/B/A and a
  calibrated minimum detectable effect.
- A human-chosen latency bound per Target: rejected by the owner. It is
  arbitrary and unfalsifiable; a floor derived from measured constants can be
  checked against measurement and revised with the constants.
- A cycle-level hardware simulator as the objective's source: rejected. None
  is usable for this hardware and building one is a project of its own; an
  analytic model validated against measurement is the reachable form.
- A per-Target manifest file, or one acceptance module per Target: rejected
  by the owner. Nearly everything is the same for every Target and depends
  on the precision policy, not the model; shapes are not restated outside
  the model spec and the engine.
- Write the generic latency harness before the engine protocol: rejected.
  It would have hard-coded the stage names and been rewritten after the
  runtime lift.
- Adopt LIBERO in this phase: deferred by the owner; the registry keeps the
  slot.

## Consequences

- Adding a Target is: model contract and reference, a pipeline against the
  op table, a declared buffer plan and segment list, cost declarations, a
  reference backend, a factory entry in `benchmarks/targets.py`, and an
  acceptance entry. Every harness and the gate then work unchanged. A
  `target-onboarding` skill to sequence this is still to be written.
- The structural tier models the launch term only; under-one-wave kernels
  are counted and reported, so the structural floor is optimistic where they
  dominate (the Pi0.5 decoder: 560 of 1100 launches below the CTA knee on
  the pdl plan). Adding that term bumps the model form version.
- The floor is validated per segment; per-call-site validation needs the
  launch-order attribution of the profiler analysis.
- The budgets in the Target entries are the kernel-design loop's defaults,
  not the owner's numbers.
- On shared, unlocked nodes `median` and `p99` carry node noise; the
  calibration reports it and the candidate rule reads `min` for improvement.
- No per-model harness remains under `benchmarks/`: latency, profile,
  kernels and floor take the Target as an input. The production-graph
  structure checks the Pi0.5 harness carried became a graph contract the
  CUDA backend declares and the profile runner checks; per-call-site
  attribution comes from an instrumented eager run matched positionally
  against the replay, so no per-plan kernel-sequence table is maintained.

## Verification

All on H100 SXM5 (`acd_u`, clocks unlocked), against the same seeded
random weights and inputs before and after each step.

- Runtime lift (jobs 596742 pre-refactor worktree, 596769 after): the action
  chunk, a second replay and the prefix K/V are bit-identical between the
  two revisions for Pi0.5 on the all-TileLang and the
  attn-ffn-cuda-fused-producer-pdl plans, and for Pi0 fused and unfused;
  plan_parity 1/1 and 1/18 and the Pi0 fused-vs-unfused check pass; chunk-latency
  `min` moves by at most 0.13 ms. Binding reproduces the previous
  hard-coded routing rules on all 243 route combinations (CPU).
- Registry and metrics module (job 596804): plan_parity reports the same
  numbers as before the change; kernel_parity and enc_attn_parity pass;
  reports carry the identity block.
- Generic runners (job 596812): in_engine on the pdl plan passes at 1/1
  (min cosine 0.9999994, replay identical, allocations finite) and reports
  at 1/18 cumulative and isolated; Pi0 fused vs unfused passes at 1/1;
  latency A/B/A at 100 reps resolves the pdl plan's -1.16 ms on `min`
  against a 0.010 ms control spread, and marks `p99` indistinguishable on
  the shared node.
- Floor model (job 596820, version 1+ae75dea09e35): valid on every segment
  of both Targets; Pi0.5 pdl plan roofline 7.70 ms, structural 9.46 ms,
  measured 16.32 ms.
- Promotion gate (job 596906): the Pi0.5 pdl candidate against the
  TileLang reference passes every in-engine gate (shallow 0.9999994,
  deep 0.99943 and multistep 0.96954 reported), improves chunk `min` by
  1.12 ms against a 0.038 ms control spread with no regression, and comes
  out `blocked` because the official-baseline scripts cannot import their
  adapter in this environment; the Pi0 unfused candidate against fused comes
  out `fail` on the candidate rule (+1.32 ms), the intended negative test.
  The older Pi0.5 harness runs without its datasheet floors.

## Related notes

- [explicit graph and ModelRunner](2026-09-06-explicit-graph-and-model-runner.md):
  the form the runtime/Target boundary took after this note; amends Decision
  §1 and §7 above.
- [kernel-design workflow](../process/2026-09-01-kernel-design-workflow.md):
  the kernel-task loop this note's Target-level acceptance wraps around.
