# Agent Note: runtime/Target boundary, and acceptance criteria before optimization

Status: proposed

## Problem

The repository's stated architecture (a small model-agnostic runtime under a
Target that owns every specialization, driven by an agent optimization plane)
is only partly represented by the code, and the parts that are missing are the
ones that decide whether a new VLA model or device can be brought up quickly:

- The runtime is ScratchPool, graph capture and in-graph timing. Buffer
  allocation, the warmup/freeze/capture lifecycle, input staging and host
  slots are written separately in the Pi0 engine (one graph) and the Pi0.5
  engine (three graphs), and the two duplicate each other.
- Latency evaluation is two divergent per-model harnesses with different
  report shapes; the Pi0.5 stage floors divide by datasheet peaks, contrary to
  the change-gates rule that a floor divides by a measured tag.
- Correctness evaluation has five metrics duplicated across scripts and no
  shared module, although the parity reference already requires one.
- There is no acceptance criterion per Target and no derived latency
  objective. The flywheel has no stop condition beyond per-kernel task
  contracts, and no harness can turn a report into a verdict. The one
  per-call-site roofline is hard-coded to one Target's decoder and divides by
  datasheet peaks.
- A draft design proposed that the runner allocate per-op buffers by reading
  the op table, and that VLA latency be evaluated with LLM-serving metrics.
- The architecture was described in one file plus an external draft, which
  cannot carry per-component status or be edited by one owner per topic.

## Proposal

1. **Boundary rule.** What is invariant across Targets goes into the runtime;
   all else goes into the Target. The Target declares its buffer plan and its
   segment list as data; the runtime materializes the arena, owns the shared
   lifecycle, validates backend-declared route constraints at bind time, and
   exposes an engine protocol that generic harnesses consume. The runtime never
   infers lifetimes, aliasing or padding. Both existing engines move onto it.
2. **Acceptance criteria first, as one registry.** A device-free
   `eval/acceptance.py` holds framework defaults (metrics, statistics,
   mandatory gates, comparison structure, noise policy, tolerances keyed by
   precision policy) and per-Target entries holding only budget and
   overrides. The human defines accuracy requirements, ratifies the
   conventions and sets the budget; no latency number is human-chosen. The
   engine exposes an `identity` property that harnesses stamp into every
   report; a promotion-gate script turns reports into verdicts. No manifest
   file: shapes are not restated outside the model spec and the engine.
   Precision policy is defined now as an identity axis with `bf16` its only
   value.
2a. **Derived latency objective.** A floor model over measured
   hardware-unit-test constants, with a plan-independent roofline tier and a
   plan-dependent structural tier (launch ramp, under-one-wave minimum
   latency, cold-burst ramp, host slots, dependency chains, dependent-launch
   overlap). Call sites declare bytes, FLOPs, launch and wave counts; the
   model is validated per call site against measured in-graph time and is
   versioned. The objective is the gap to the structural floor; the stop
   condition is attribution of the residual gap or budget exhaustion, never
   "reach the floor".
3. **Latency metrics for VLA, not serving metrics.** Chunk latency is the
   headline; device, host, per-segment and overhead are reported; `min`,
   `median` and `p99` are all reported with clocks unlocked, because
   deployment does not lock them; deltas come only from same-process A/B/A;
   a noise calibration establishes the minimum detectable effect per
   statistic, below which a delta is indistinguishable.
4. **Correctness structure, not only metrics.** Five shared metrics; two
   tensor-level oracle tiers (in-engine reference route, official baseline);
   shallow gates and deep reports; oracle injection at stage boundaries;
   per-layer smoothness; padding finiteness; replay determinism as a gate.
   Policy quality keeps an empty slot and is out of scope this phase.
5. **Documentation as a tree.** `docs/architecture/` owns the normative
   architecture, one topic per file with a status line; `ARCHITECTURE.md`
   becomes the index. The previous single-file content is carried into
   `20-target-layout.md` and `10-runtime.md` without loss.

Order of work: acceptance schema and the Pi0.5 instance; shared metrics
module and report schemas; engine protocol with both engines refactored onto
the runtime; generic latency and correctness runners; noise calibration;
promotion gate. The first two need no GPU.

## Alternatives considered

- Runner infers per-op buffers from the op table: rejected. Padding, masking,
  aliasing and the KV-cache layout are Target decisions that fusion changes;
  inferring them makes the runtime a compiler and removes the agent's control
  over layout.
- Copy TTFT/TPOT/throughput from LLM serving: rejected. A batch-1 fixed-shape
  closed-loop workload has no request distribution; the analogues are chunk
  latency and its tail.
- Lock clocks to make `median` and `p99` reproducible: rejected by the owner.
  Deployment runs unlocked; the harness compensates with A/B/A and a
  calibrated minimum detectable effect.
- Write the generic latency harness before the engine protocol: rejected. It
  would hard-code the current stage names and be rewritten after the runtime
  lift.
- Adopt LIBERO in this phase: deferred by the owner. The acceptance schema
  keeps the slot so a precision-policy change cannot be promoted without it.
- A human-chosen latency bound per Target: rejected by the owner. It is
  arbitrary and unfalsifiable; a floor derived from measured constants can be
  checked against measurement and revised with the constants.
- A cycle-level hardware simulator as the objective's source: rejected. None
  is usable for this hardware and building one is a project of its own; an
  analytic model validated per call site is the reachable form.
- One acceptance module per Target: rejected. Nearly everything in it is the
  same for every Target and depends on the precision policy, not the model;
  one registry with defaults and per-Target overrides keeps it in one place.
- Keep `ARCHITECTURE.md` in full beside the tree: rejected. Two copies of the
  dependency rules would drift.
- A per-Target manifest file (`target.toml`) carrying identity, plans and
  acceptance: rejected by the owner as needless machinery. It would restate
  shapes the spec and engine already hold, and it would add a parser and a
  schema for what a Python dictionary expresses with import-time checking.

## Acceptance criteria

- `docs/architecture/` exists with one file per topic and a status line each;
  `ARCHITECTURE.md` is an index; every rule from the previous
  `ARCHITECTURE.md` is present in the tree.
- `eval/acceptance.py` exists with framework defaults and an H100 x Pi0.5
  entry, imports nothing that needs a device, and the promotion gate reads
  it; the engine exposes `identity`; a latency report and a correctness
  report each carry the identity block and refuse comparison across
  identities.
- Every call site of the H100 x Pi0.5 Target carries a cost declaration; the
  floor model reports roofline and structural per call site and per segment
  beside measured in-graph time, with no predicted structural floor above a
  measured time and every term citing a tag.
- One metrics module under `eval/correctness/` is imported by every parity
  script; no script re-declares a metric.
- Both engines construct through the runtime's lifecycle and satisfy the
  engine protocol; the generic latency runner emits the report schema for both
  Targets with no model names in the runner.
- After the refactor, a same-process A/B/A of the pre-refactor and
  post-refactor Pi0.5 engines on the shipped plan is within the calibrated
  minimum detectable effect on `min`, and every existing parity gate passes.
- Floors in every report divide by hardware-unit-test tags.

## Risks

- The runtime lift touches capture; a regression would show as a latency
  change or a lost gate. Mitigation: the A/B/A above is the gate for the
  refactor PR itself.
- `p99` may not be resolvable on contended nodes; the calibration will say so
  and the bound downgrades to `report` rather than gating on noise.
- Scope creep from "generic" toward a framework; the boundary rule and the
  engine protocol are the limit, and any per-model name in the runtime or a
  runner is a defect.

## Related notes

- [kernel-design workflow](../../implemented/process/2026-09-01-kernel-design-workflow.md):
  the kernel-task loop this note's Target-level acceptance wraps around.
