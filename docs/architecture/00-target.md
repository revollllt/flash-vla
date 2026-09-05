# The Target

Status: partial. The four axes, target ownership and the engine identity
property are the repository state; the acceptance registry entry, the
call-site cost declarations and identity stamping in reports are planned.

## Definition

```text
Target = hardware x model revision x shape profile x precision policy
```

Every Target owns its own pipeline, buffer plan, call-site bindings, fusion
boundaries, backend kernels, tuning results, plans, parity scripts, benchmark
cases and acceptance criteria. Nothing of that is shared through a global
architecture-only directory; two Targets share code only after the common part
has been proven identical in production and extracted afterwards, never in
anticipation.

## The four axes

**Hardware** is a device profile at SKU granularity (H100 SXM5 80 GB, not
"Hopper"). It carries a static spec (published capacities and capabilities)
and the measured constants produced by the `hardware-unit-test` skill. Floors
and roofline targets divide by measured, tagged constants; a datasheet peak is
never a floor.

**Model revision** is the hardware-independent model contract under
`src/flash_vla/models/<model>/`: the constant set, the checkpoint schema, the
weight folding rules, the readable torch reference, and the tokenizer where the
model needs one. The contract is defined up to lossless, schedule-independent
relayout; anything that depends on the inference schedule (a folded per-step
constant, for instance) belongs to the Target's runtime shapes, not to the
model contract.

**Shape profile** is the fixed set of shapes the engine is captured at: views,
prompt padding, action chunk, denoising steps, depth. A profile is named. A
different profile that changes graph topology or fusion boundaries is a
different execution plan; one that only changes a compile-time constant is the
same plan re-tuned.

**Precision policy** names the dtype of each tensor class: weights,
activations, KV cache, accumulation, and any quantization format with its
scaling scheme. The only policy defined today is `bf16` (all classes bf16,
fp32 accumulation). A precision policy change changes correctness tolerances,
so acceptance criteria are per Target, never global. A policy other than
`bf16` cannot be promoted without a policy-quality gate; until that suite
exists the policy is fixed at `bf16`.

## Identity, acceptance entry and cost declarations

No manifest file. A Target is described by three things it already has or
nearly has:

- **identity** is a property of the engine (`runtime.Identity`): the named
  axes, the shape profile's numbers, the resolved plan, any target-local
  table option, and the git revision. Every harness stamps it into its report
  (planned). Nothing restates a shape that the model spec or the engine
  already knows.
- **acceptance** is one entry in the framework's acceptance registry
  ([`30-acceptance-criteria.md`](30-acceptance-criteria.md)): the budget and
  any override of the framework defaults. Metrics, statistics, gates and
  tolerances are framework defaults keyed by precision policy, not per-Target
  text. The latency objective is not written anywhere; it is derived from the
  floor model.
- **cost declarations** on each call site: bytes, FLOPs, launch count and
  wave count as functions of the shape profile and precision policy. The
  floor model ([`33-latency-floor-model.md`](33-latency-floor-model.md))
  consumes them to derive the Target's latency objective.

## Identity and comparability

Every report a harness emits carries the identity block: target, shape profile,
precision policy, plan, git revision, device name, driver and framework
versions, node and job id where a scheduler is used. Two measurements are
comparable only when target, shape profile, precision policy and plan match;
a tool given two reports with differing identity refuses to compare them.
This is the rule that already governs plans (a report that does not say which
implementation ran cannot be compared) extended to the whole identity.

## Plans

A plan maps each call site to the backend that implements it. It is resolved
once, before capture; the replay path never dispatches. Backends declare route
constraints (call sites that must move together because they share a buffer
contract); binding rejects a plan that violates them. A plan is the provenance
record of a run and is stamped into every report.

## Lifecycle

1. **Onboarding**: the model contract, the pipeline written against the op
   table, the buffer plan, the first (reference) backend, the parity scripts,
   the e2e benchmark case, the call-site cost declarations and the acceptance
   registry entry. A `target-onboarding` skill
   (planned) sequences this; today it is done by hand.
2. **Optimization**: the agent flywheel of
   [`40-optimization-plane.md`](40-optimization-plane.md), closing the gap to
   the structural floor one kernel task at a time, each promoted through the
   acceptance criteria, until the gap is attributed or the budget is spent.
3. **Promotion**: a candidate that passes every gate enters the Target's
   shipped plan set; a failed candidate keeps its reason in an Agent Note and
   never touches the production route.

## Multi-target rules

- One Target must not import another Target's private kernels.
- Shared primitives (the sm90 tile library) are extracted from proven common
  code and carry their own parity suite.
- A new device is a new Target. The method, the runtime and the agent workflow
  transfer; hardware constants are re-measured, precision and kernels are
  re-selected, and the execution plan is rebuilt. RTX 5090 / Blackwell is the
  next hardware Target in this sense and has no validated numbers.
