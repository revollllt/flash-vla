# Agent Note: the explicit computation graph, the VLA template and one ModelRunner

Status: implemented

## Problem

Two Targets (Pi0, Pi0.5) each carried a hand-written engine class, a
hand-written buffer plan, a hand-written cost table and a module-global
scratch pool in their TileLang backend, and used different names for the same
three stages and the same operations. Bringing up a third model would have
meant copying all of that again. Nothing in the runtime knew the model's
computation graph, so buffers, costs and profile attribution were maintained
by hand beside the pipeline instead of derived from it. Deployment
configuration was spread over nine named plans and a `fused` table option
that never reached the identity.

## Decision

1. **The op vocabulary** (`src/flash_vla/runtime/ops.py`). Every call site has
   one `OpSpec`: the wrapper's positional parameter names (every backend for
   that call site must take them in that order), which parameters are written,
   which are weights, which are auxiliary outputs no cost counts, and the FLOP
   formula over argument shapes. The standard vocabulary uses full stage
   names, `vision_encoder_*`, `llm_backbone_*`, `action_expert_*`, and is
   shared by both Targets; a Target's backends may add extension ops (Pi0's
   state projection and action MLP). Every op writes its outputs in place; the
   runner ignores return values. Pi0's parameter lists were aligned to the
   shared specs by passing `None` for the AdaRMS terms it does not have.
2. **The explicit graph** (`runtime/graph.py`). A Target writes its forward
   pass as data with a builder API: `g.buf` declares buffers (padded
   allocation, exposed view, alias views for stage contracts), `g.w` names
   weights, `g.op` appends a node with arguments by parameter name, `g.copy`
   appends a copy node, `g.stage` / `g.host` lay out the program. References
   are immutable descriptions that support leading-axis indexing and `view`;
   they resolve to tensors once, after allocation. Python loops unroll layers
   and steps into as many nodes as the captured graph has launches. The graph
   yields the buffer plan, the per-stage node sequence, each node's reads and
   writes, the derived per-call-site costs, and the call-site set a plan must
   route. Nothing in construction touches a device.
3. **The VLA template** (`runtime/vla.py`). `VLA` fixes the three stages,
   the input spec, the output buffer, the canonical stage outputs
   (`vision_encoder_x`; `prefix_k`/`prefix_v`; `actions`, `suffix_k`/`suffix_v`),
   and defaults for sampling, staging and plan selection. A Target subclasses
   it with its configuration, shape numbers, weight schema and loader, backend
   registry, shipped and reference plans, host slot, and `build`. Onboarding a
   model is rewriting its original forward pass in this form and passing the
   precision gate; the result need not be written the way the original was.
4. **One runner** (`runtime/runner.py`). `ModelRunner` builds the graph,
   binds the plan through the Target's `Registry` (`runtime/registry.py`),
   builds the op table with the runner's workspace allocator injected into
   every backend factory, allocates the declared buffers, resolves every node
   once, warms up, freezes the allocator, and captures each stage. It is the
   engine protocol's only implementation. Construction without a checkpoint
   and without capture stops after the graph, which is what `python -m
   eval.smoke` checks on the login node.
5. **Workspaces come from the runner.** Backend-internal memory (the
   FlashDecoding partials, Pi0's staged QKV projection and score matrices,
   the persistent FFN's K-major input and counters) is requested through the
   injected `scratch(role, shape, dtype, device)` during warmup and frozen
   before capture. The module-global pool and its `use_pool` are gone; a
   wrapper called as a free function (the lab scripts) gets a plain
   allocator by default.
6. **One deployment configuration per Target.** A Target declares `plan`
   (shipped) and `reference_plan` (the correctness oracle route). Every
   harness takes `--plan shipped|reference|<json>|<path>`; the seven other
   Pi0.5 plans moved to `lab/plans/`. Pi0's fused overlay is a backend,
   `tilelang-fused`, that its shipped plan routes three call sites to;
   `Identity.options` no longer exists, the plan is the only implementation
   axis of an identity.
7. **Names.** Stages are `vision_encoder`, `llm_backbone`, `action_expert`
   for both Targets (Pi0's single segment is split into the same three);
   call sites and buffers use those full prefixes; the KV cache is
   `kv_k`/`kv_v` with `prefix_*`/`suffix_*` alias views in both Targets.
   Weight names are unchanged: they are the model contract and the checkpoint
   converters' keys.

## Alternatives considered

- Keep the hand-written buffer plans and cost tables and only lift the
  engine boilerplate into the runtime: rejected by the owner. The graph is
  what a new model brings; buffers and costs are consequences of it and
  should not be maintained twice.
- Allocate by tracing a first execution (a runner-owned scratch pool keyed
  by name): rejected. It obtains the graph by running it, so nothing can be
  checked without a device, and the two returning attention ops would have
  had no declared output.
- A stateful TileLang factory closing over the pool, keeping the free
  functions for scripts: adopted in the form of the injected allocator with a
  plain default, which serves both.
- Abbreviated call-site prefixes (`backbone_*`, `expert_*`): rejected by the
  owner as ambiguous; full stage names are used everywhere.
- Renaming weights to the stage prefixes: deferred. It touches the OpenPI
  adapters and the fold; a separate change if wanted.
- Deleting the `Engine` protocol now that it has one implementation:
  rejected; it is the documented surface the harnesses import from.

## Consequences

- Adding a Target is one `target.py` (contract, plans, registry) and one
  `pipeline.py` (`build`), plus a factory entry in `benchmarks/targets.py`
  and an acceptance entry. No engine, buffer plan or cost table.
- `python -m eval.smoke` runs on the login node: graph structure, spec
  arity, declared references, stage outputs, non-zero costs, every plan
  under `lab/plans/`, and the 243 route combinations of Pi0.5's five
  routable call sites against the binding rules.
- Identity plans are keyed by the new call-site names and cover the graph's
  call sites only; reports written before this change are not comparable by
  plan (their workload axes still are).
- `benchmarks profile` still attributes by marker kernels and positional
  matching of an instrumented eager run; attributing by node order is a
  follow-up. `benchmarks kernels` replays recorded node arguments without a
  scratch scope.
- Pi0 gains two graph launches per forward from the stage split and Pi0.5's
  reference route gains one copy per backbone layer (the torch attention
  chain now writes its output buffer); the shipped routes issue the same
  kernels in the same order as before.
- Known debt, unchanged by this note: `vision_encoder_patch_embed` still
  materializes a contiguous patch view inside the wrapper on the shipped
  route; the CUDA attention `Workspace` allocates in its own constructor
  rather than through the injected allocator.
- The experiment scripts still under `eval/correctness/pi05/` call the
  renamed wrappers and move to `lab/` in the next change with the eval
  reorganization; `plan_parity` was deleted, its comparisons being those of
  the generic in-engine check.

## Verification

All on H100 SXM5 (`acd_u`, ACD1-33, clocks unlocked), against the seeded
random weights and inputs of `artifacts/refactor/pr1/before` (job 598393 at
655d7fe).

- Bit-identity (job 598829): PENDING.
- Same-node latency, baseline worktree at 655d7fe then this tree, in one job
  (job 598830): PENDING.
- In-engine checks, kernels, profile, floor and the gate on this tree (job
  598830): PENDING.
- Login node: `python -m eval.smoke` passes 24/24; every module under
  `benchmarks/` and `eval/` imports.

## Related notes

- [runtime/Target boundary and acceptance first](2026-09-06-runtime-target-boundary-and-acceptance-first.md):
  the boundary rule this note implements; its Decision §1 and §7 are
  amended to point here.
- [kernel-design workflow](../process/2026-09-01-kernel-design-workflow.md):
  the kernel-task loop that now reads a call site's cost from the graph.
