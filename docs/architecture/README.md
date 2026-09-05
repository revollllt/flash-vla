# Flash-VLA Architecture

Flash-VLA builds fixed-workload inference targets for vision-language-action
(VLA) models. The atomic production unit is a **Target**: one device, one model
revision, one shape profile, one precision policy. Peak end-to-end action
latency on that fixed workload takes priority over any general-purpose
abstraction.

The system has two planes (figure: [`../arch.png`](../arch.png)):

- **Static VLA Inference Plane.** A small runtime that owns static addresses,
  scratch, graph segments and their capture/replay, and knows nothing about the
  model, the kernels or the device. Every specialization lives in the Target.
- **Autonomous Optimization Plane.** An agent workflow that produces and
  improves Targets from measured evidence, guided by skills and Agent Notes.
  The human defines accuracy requirements, the metric and framework
  conventions, and a budget. The latency objective is derived from a floor
  model over measured hardware constants, never chosen by hand.

"Support for any VLA model" means the optimization plane can bring a new model
or a new device up as a Target quickly, not that the runtime carries a
universal operator library.

## Documents

Read in order. Each document carries a `Status` line: `implemented` means the
repository state matches it, `partial` means some of it does, `planned` means
it is the agreed contract for work not yet shipped. Planned items are also
marked inline.

| Document | Owns |
|---|---|
| [`00-target.md`](00-target.md) | the Target definition, identity, acceptance module and comparability |
| [`10-runtime.md`](10-runtime.md) | the Static Inference Runtime: arena, scratch, segments, lifecycle, binding, the engine protocol |
| [`20-target-layout.md`](20-target-layout.md) | what a Target contains, dependency direction, specialization rules |
| [`30-acceptance-criteria.md`](30-acceptance-criteria.md) | what the human defines, the acceptance registry, stop condition, gate semantics |
| [`31-latency-evaluation.md`](31-latency-evaluation.md) | end-to-end latency metrics, statistics, measurement discipline, report schema |
| [`32-correctness-evaluation.md`](32-correctness-evaluation.md) | numerical correctness: metrics, oracle tiers, comparison structure, report schema |
| [`33-latency-floor-model.md`](33-latency-floor-model.md) | the derived latency objective: two floor tiers, gap decomposition, validation |
| [`40-optimization-plane.md`](40-optimization-plane.md) | the agent flywheel, skills, Agent Notes, promotion gate, human role |

## Vocabulary

- **Target**: hardware x model revision x shape profile x precision policy.
- **Engine**: the constructed instance of a Target: weights loaded, buffers
  materialized, segments captured. Its online path is what the figure calls
  the Minimal Runner: input staging, host slots, replay.
- **Segment**: one captured CUDA graph. A Target declares an ordered list.
- **Call site**: one named operation the pipeline invokes through the op table.
- **Backend**: one implementation strategy providing call-site wrappers
  (TileLang, hand-written CUDA, library).
- **Plan**: the per-call-site backend route of one engine; the provenance
  record of every measurement.
- **Oracle**: the implementation a correctness comparison reads truth from.
- **Gate / report**: a check whose failure blocks promotion / a check whose
  result is recorded only.
- **Floor model**: the analytic cost model over measured constants that
  derives a Target's latency objective; its two tiers are the roofline (plan
  independent) and the structural floor (of the current plan's form).
- **Gap**: measured latency minus the structural floor; what the flywheel
  closes. Structural minus roofline is what a change of plan form could
  recover.
- **Kernel task**: the unit of the `kernel-design` skill, one contract inside a
  Target. Distinct from the Target, whose criterion is the acceptance registry and the gap to the structural floor.

Earlier drafts used "Task" for the co-designed unit; that word is retired.
"Target" is the only name.

## Status summary

| Component | Status |
|---|---|
| ScratchPool, graph capture, in-graph timing (`runtime/cuda`) | implemented |
| Target-declared buffer plan materialized by the runtime (`StaticArena`) | implemented; both Targets declare their plans |
| Segment list with host slots, shared lifecycle (warmup, freeze, capture, replay) (`Program`) | implemented; both engines construct through it |
| Op-table binding before capture, per-call-site plans, backend-declared route constraints | implemented (`runtime/binding`) |
| Engine protocol (`runtime/engine.Engine`) and identity property | implemented; both engines satisfy it, engine-level reports carry the identity, the generic runners consume it |
| Acceptance registry with framework defaults and per-Target entries | implemented (`eval/acceptance.py`); no consumer turns it into a verdict yet |
| Call-site cost declarations and the floor model | planned; one decoder roofline exists, hard-coded and datasheet-based |
| Shared correctness metrics module | implemented (`eval/correctness/metrics.py`); every parity script imports it |
| Generic e2e latency runner with a fixed report schema, A/B/A deltas and noise calibration | implemented (`benchmarks/latency.py`); floors empty until the floor model exists |
| Generic in-engine correctness runner (stage outputs, oracle injection, per-layer profiles) | implemented (`eval/correctness/in_engine.py`); the baseline tier stays per model |
| Skills: kernel-design, benchmark-kernel, hardware-unit-test, gpu-profiler-analysis, ncu-report | implemented |
| Target-onboarding skill | planned |
| Agent Notes lifecycle | implemented |
| Promotion gate script | planned; CI deferred and under discussion |
| Policy-quality suite (LIBERO) | out of scope for the current phase |

## Relation to other documents

These documents are normative: interfaces, invariants, ownership, measurement
conditions and validation requirements. Source is authoritative for
implementation. Decisions and their alternatives live in
[Agent Notes](../../.agents/notes/README.md); the collaboration rules live in
[`.claude/rules/`](../../.claude/rules/).
