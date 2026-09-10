# Must-Read Skills Before Modifying Components

Before modifying a component, read the skill that defines its contract. This
is intentionally a small project-specific adaptation of SGLang's component
gate: add a row only when a stable boundary and a maintained skill exist.

- **Writing a new kernel, porting an op to CUDA/TileLang, fusing ops, or an
  optimization pass on an existing kernel**
  -> [`kernel-design`](../skills/kernel-design/SKILL.md).
- **Per-kernel latency, benchmark harnesses, or timing claims**
  -> [`benchmark-kernel`](../skills/benchmark-kernel/SKILL.md).
- **Machine constants, hardware microbenchmarks, or the denominator under a
  floor / roofline target**
  -> [`hardware-unit-test`](../skills/hardware-unit-test/SKILL.md) and the unit
  reference for the primitive concerned. The floor model reports a datasheet
  roofline and a measured ceiling side by side: the ceiling divides only by
  tagged measured constants of the hardware axis's `measured/` table, the
  roofline only by `spec.py` peaks, and neither is an objective.
- **Model/module timelines, trace export, or the shared capture runner**
  -> [`gpu-profiler-analysis`](../skills/gpu-profiler-analysis/SKILL.md) and
  its relevant capture-mode reference.
- **Selected-kernel NCU capture settings, counters or diagnosis**
  -> [`ncu-report`](../skills/ncu-report/SKILL.md).
- **Graph, backend, or buffer ownership changes** -> read
  [`ARCHITECTURE.md`](../../ARCHITECTURE.md) and the Target's `target.py` and
  `pipeline.py` before editing.

Two repository hygiene gates apply to every row:

- Inspect `git status` before and after a scoped edit; preserve unrelated user
  changes.
- Do not edit kernel source while a TileLang compile or profiling job is active;
  wait for the job to finish and keep its artifact directory separate.
