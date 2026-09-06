# Agent Note: device-level component packages

Status: implemented

## Problem

The two H100 Targets, Pi0.5 and Pi0, run the same SigLIP vision encoder and
the same Gemma backbone at the same shapes (three views, 768 image tokens,
identical weights layouts), and their action experts share every width. Each
Target carried its own copy of the wrappers and kernels for those components,
and the copies had already diverged: Pi0's vision wrappers route the QKV and
FFN-up projections to different TileLang kernels than Pi0.5's, and the two
Targets' `base.py` and `attention.py` kernel files differ in length and
content. A kernel written for one Target therefore had to be written twice or
shipped once, and the optimization campaign
([campaign plan](../../proposed/performance/2026-09-06-optimization-campaign-plan.md))
opens three lanes whose kernels both Targets need.

Two smaller things stood in the way of sharing. The op vocabulary's two vision
pre-norm GEMMs (`vision_encoder_norm_qkv`, `vision_encoder_norm_ffn_up`)
took a graph buffer for the normalized activation as an auxiliary output that
nothing downstream reads, so a backend fusing the norm into the GEMM would
have had to write a buffer it never materializes. And `eval/smoke.py` hard
coded Pi0.5's backend names for its route-combination check and checked no
combination on Pi0, so a backend registered on both Targets would have been
checked on one.

## Decision

1. **A component package per model component and device.**
   `src/flash_vla/hardware/<vendor>/<device>/<component>/` (`siglip`,
   `gemma_backbone`, `gemma_expert` on H100) holds the kernels, the
   reference mirrors and the backend factories of that component on that
   device, written once. A package's backend satisfies the registry contract
   of `runtime/registry.py` (`NAMES`, `make_wrappers(scratch, selected_names)`,
   optional constraints, graph contract and extension ops). A package imports
   `models/`, `runtime/` and the shared tile library and never a Target.
2. **The Target keeps every routing decision.** A Target registers a
   package backend under a name of its own in its `backends/__init__.py`,
   and its plans, route constraints and reference route stay its own. The
   Target's existing TileLang backends remain; Pi0's forked vision TileLang
   route becomes its reference route once a shared vision backend ships.
3. **Existing kernels move with the promotion that first shares them**, not
   ahead of it: `enc_attn` to `gemma_backbone`, `attn_taskloop` and
   `ffn_taskloop` to `gemma_expert`, each move proven bit-identical on the
   Target it came from in the same PR.
4. **The budget counts per kernel-design task (a lane), not per Target.**
   `eval/acceptance.py` records this; a task's contract may narrow its
   budget, never widen it.
5. **The vision pre-norm GEMMs own their normalized activation as workspace.**
   The two ops lose their `x_norm` parameter; a backend that materializes
   the normalized rows takes them from the runner's injected allocator, and
   one that fuses the norm writes nothing. The graph's `vision_encoder_norm`
   buffer remains for the projector, which still declares its auxiliary
   output.
6. **`eval.smoke` reads backend sets from each Target's registry** and
   checks every route combination over the constrained call sites, plus every
   plan-selectable call site alone on each backend that provides it, against
   a per-Target oracle written in prose independently of the declared
   constraints. A Target whose backends declare constraints must have an
   oracle; a candidate plan under `lab/plans/` is named
   `<target>-<name>.json` so the check reads its Target from the prefix
   (`lab/plans/README.md`).

## Alternatives considered

- **Keep one copy of each component per Target.** Rejected: the copies had
  already diverged without a reason that survives review, and the campaign
  would have written every kernel twice or optimized one Target only.
- **Optimize Pi0.5 alone.** Rejected by the owner: both Targets are in the
  campaign.
- **A `backends/common/` under one Target imported by the other.** Rejected:
  a Target importing another Target's kernels is the dependency the
  architecture forbids, and the owner of such a directory would be
  accidental.
- **Move every existing kernel into packages now.** Rejected: a large diff
  with no gain and a bit-identity proof per kernel; moving each with the
  promotion that shares it keeps every PR one decision.
- **Keep `x_norm` on the vision ops and let a fused backend ignore it.**
  Rejected: an output a wrapper may leave unwritten is not a contract, and
  the cost model would have kept counting a buffer nobody reads.

## Consequences

- Lanes B (SigLIP), C (Gemma backbone) and D0 (the expert chain on Pi0)
  write each kernel once and register it on both Targets.
- `ARCHITECTURE.md` carries the dependency line for component packages and
  the rule that a component package imports no Target; the grep
  `grep -rn "h100\.pi0\b\|h100\.pi05" src/flash_vla/hardware/nvidia/h100/<component>`
  must stay empty for every package.
- The TileLang vision wrappers of both Targets request the normalized rows
  through `scratch("vision_norm", ...)`; the runner records the requesting
  node and freezes the allocator after warmup as for every workspace.
- A new backend on either Target is exercised by `eval.smoke` without
  editing the check, unless it declares route constraints, in which case the
  oracle for that Target must state the new rule.

## Verification

- Login node: `python -m eval.smoke` (both Targets, seven candidate plans,
  both Targets' route combinations: Pi0.5 243 combinations over the five
  constrained action-expert sites plus its selectable backbone attention on
  each backend; Pi0 its three fused sites on each backend);
  `python -c "import flash_vla"`; `grep -rn "lab/" src eval benchmarks` is
  empty.
- GPU, bit identity of the `x_norm` removal (job 599719, ACD1-30, shared
  node): `lab/sbatch/bit_identity.sh` dumped every declared stage output of
  both Targets on both plans from the tree before this change (fccdf94) and
  from this tree (a02bc64) with seed 0 and compared them with `torch.equal`:
  all four Target x plan pairs bit-identical, second forward included.
  `python -m eval.correctness --steps 1 --layers 1` in the same job: Pi0.5
  shipped vs reference min cosine 0.9999994, max rel_rms 1.06e-3; Pi0
  0.9999810 and 6.18e-3; both replay-identical and finite, the same numbers
  the tolerance calibration read (job 599011). Evidence:
  `artifacts/bit_identity/599719/` on this machine.

## Related notes

- [explicit graph and ModelRunner](2026-09-06-explicit-graph-and-model-runner.md):
  the registry contract a package backend satisfies and the workspace
  allocator it uses.
- [deployment configuration and the lab workspace](../process/2026-09-06-deploy-config-and-lab.md):
  where candidate plans live and why they are tracked.
- [acceptance is deployability](2026-09-06-acceptance-is-deployability.md):
  the registry whose budget this note re-scopes to a lane.
