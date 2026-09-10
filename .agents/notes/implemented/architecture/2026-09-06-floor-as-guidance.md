# Agent Note: the floor model is guidance, and its constants live on the hardware axis

Status: implemented

## Problem

`benchmarks floor` called itself "the derived objective of a Target". It
reported a roofline tier and a structural tier and split the distance to the
measured time into a "kernel gap" and a "form gap", both of which read as work
to do. On the Pi0.5 Target the structural tier carried only a launch term, so
the kernel gap was 6.9 ms while the closed optimization loop had shown the
action expert sitting at the cold-delivery ceiling of its own geometry. A
number that says "6.9 ms of kernel work" when there is none is worse than no
number.

The measured constants the model divided by lived inside the
`hardware-unit-test` skill (`.claude/skills/hardware-unit-test/sm90/`), while
the datasheet quantities of the same machine lived in the source tree
(`hardware/nvidia/h100/spec.py`). The two denominators of one floor sat in two
trees, and `floor.py` reached into the skill directory by a hard-coded path.
Three rules the constants stated in prose (the fixed cost behind the streaming
rate, the burst-size curve, the small-grid derating) had no machine-readable
form, so the model could not apply them.

## Decision

1. **Three columns, no objective.** Per call site and per stage the report
   carries `roofline_us` (datasheet: `max(bytes / HBM peak, flops / dense bf16
   peak)`, the peaks from the hardware axis's `spec.py`), `ceiling_us`
   (measured: what this machine delivered for the geometry, from the tagged
   rows of the measured table), and `measured_us` (the in-graph time
   `benchmarks profile` attributes). The structural tier and both gaps are
   gone. Owner's ruling: the roofline is a white-paper limit and guidance
   only; measured constants may carry measurement error. These estimates guide
   kernel work toward hardware SOL; neither alone establishes attainable latency,
   acceptance or convergence.
2. **The ceiling follows the constants' own rules.** Below the burst curve,
   `fixed_us + MB / marginal rate` (`ld.bw.dev.dram`); on the curve, `MB /
   delivered rate` interpolated on `tma.bw.dev.burst`'s end-to-end curve; the
   larger of that and `flops / wgmma.clock.sm`. The rules are machine-readable
   fields on the rows (`fixed_us`, `curve_mb_gbs`, `derate_at_32`), values
   unchanged. A Target may declare a call site's ceiling outright
   (`Invocation.ceiling`, via `VLA.CEILINGS`) when a unit test measured that
   call site's own geometry; the declaration carries its tag and job id and
   is copied into the report. Pi0.5 declares the gate/up GEMM's cold delivery
   (9.41 us, `tma.bw.dev.burst`, job 591174).
3. **Headroom guides iteration.** Each site reports `pct_of_ceiling` and
   `within_ceiling` (measured at most `1 + headroom_pct / 100` times the
   ceiling, `headroom_pct` from the acceptance registry's `stop`); each stage
   and the Target report `all_within_ceiling`. Legacy registry stop logic reads
   that flag. The daily workflow uses it as evidence for the next hypothesis,
   not an automatic stopping rule. Validity stays checked: a ceiling above its measured
   time by more than the table's own noise floor (`machine.noise_floor_pct`,
   6%), or a roofline above its ceiling, marks the report invalid. The
   members of an overlapping atomic group cannot be judged by adding their
   recorded kernel durations. The current floor model has no supported joint
   ceiling for such groups; inspect the chain's timeline and measured span.
4. **The measured table lives on the hardware axis.**
   `src/flash_vla/hardware/nvidia/h100/measured/` holds `constants.yaml`, the
   arch index and the unit references, beside `spec.py`. The skill keeps its
   probes, scripts and arch-independent references; `scripts/constants.py`
   discovers the table by `HUT_CONSTANTS_ROOT`, then by walking up to
   `src/flash_vla/hardware/**/measured/`, then by its own `<arch>/` directories
   so a standalone copy still works. Unit `reference:` paths in the table are
   relative to the measured directory; `probe:` paths stay skill-relative.
5. **Estimate value before implementing a hypothesis.** The daily loop in
   [docs/optimization.md](../../../../docs/optimization.md) proceeds from whole-model
   profiling to focused analysis, kernel design, implementation, validation and
   deployment. Reuse applicable floor reports and hardware measurements; gather
   new evidence only for the selected bottleneck. Price the likely end-to-end
   saving before a trial, without a frozen contract or mandatory promotion gate.
   Reuse existing kernels and continue improving them where evidence supports
   headroom. Fusion candidates are explicitly written CUDA/TileLang kernels;
   existing `torch.compile` implementations may serve as comparison paths.
   Related correctness checks and kernel timing screen candidates before
   deployment into the actual model inference path and plan. Full-model latency
   is then measured with the deployed entry point, assets and execution settings.
   This post-deployment evidence determines retention or rollback and the next
   optimization hypothesis.

## Alternatives considered

- Keep the structural tier and add the missing terms (under-one-wave
  derating, protocol floors): rejected; every added term would again be read
  as a target, and the terms that mattered on Pi0.5 were per-kernel findings,
  not machine constants.
- Leave the table in the skill and only change discovery: rejected by the
  owner; a floor's two denominators belong together, and the source tree is
  what a new hardware axis brings.
- A symbolic link from the skill to the source tree: rejected; the skill's
  standalone contract is discovery, not a link that breaks on copy.
- Apply the CTA-knee derating to the ceiling: deferred; the constant is a
  ratio at one size, and applying it needs the per-launch grid the profiler
  reports but the cost model does not; it is carried as information.

## Consequences

- A floor report reads as three columns and a flag; `eval/gate` prints them as
  context and never gates on them.
- The daily kernel-design loop stops when the important hotspots approach an
  evidence-supported attainable capability with no worthwhile hypothesis left,
  or the user's scope or budget ends. A floor ratio or one successful candidate
  is insufficient to establish convergence.
- `python .claude/skills/hardware-unit-test/scripts/constants.py` run from
  anywhere under the repository finds the moved table; from outside, only a
  standalone copy's own table.
- `kernel-wiki/scripts/validate.py` treats a missing table as a skipped tag resolution, not an
  abort.
- Reports from form "1" are not comparable to form "2"; the version string
  changed.

## Verification

- Login node: `constants.py --validate` on the moved table passes;
  `validate.py` and `check_templates.py` pass; the three columns computed
  on the declared graphs of both Targets without a device, with the declared
  Pi0.5 ceiling attached to `action_expert_norm_gated_ffn`.
- GPU, job 598949 (ACD1-1), the first form: `eval.gate` printed the new floor
  line and the skill's `constants.py` discovered the moved table from the
  job's tree; the reports were invalid on three sites, which produced the
  noise-floor tolerance and the group rule above. Job 598964 (ACD1-55),
  the form that shipped: both Targets `valid`, every site with its three
  columns and `roofline <= ceiling` everywhere.

  | Target | roofline | ceiling | measured | sites within ceiling |
  |---|---:|---:|---:|---|
  | Pi0.5 | 6.56 ms | 9.88 ms | 16.94 ms | `llm_backbone_ffn_down_residual` 110%, `action_expert_action_in_proj` 105% |
  | Pi0 | 5.75 ms | 9.01 ms | 16.20 ms | `action_expert_norm_gated_ffn` 99%, `action_expert_state_proj` 95% |

  Neither Target is within its ceiling as a whole: the vision encoder
  measures 2.0-2.5x its ceiling on both, Pi0.5's attention chain 5.7x and
  its FFN chain 1.7x. That is the guidance this model exists to give.

## Related notes

- [acceptance is deployability](2026-09-06-acceptance-is-deployability.md):
  the registry's `stop.headroom_pct` this report's flag serves.
- [runtime/Target boundary and acceptance first](2026-09-06-runtime-target-boundary-and-acceptance-first.md):
  its Decision §4 described the two-tier objective this note withdraws.
- [gate/up copy column](../../proposed/architecture/2026-09-03-ffn-gu-copy-column.md):
  the measurement behind the declared Pi0.5 ceiling.
