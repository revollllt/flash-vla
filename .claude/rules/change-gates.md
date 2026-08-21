# Must-Read Skills Before Modifying Components

Before modifying a component, read the skill that defines its contract. This
is intentionally a small project-specific adaptation of SGLang's component
gate: add a row only when a stable boundary and a maintained skill exist.

- **Tile kernels, tiling, stages, warp roles, barriers, or instruction shape**
  -> [`tile-dataflow`](../skills/tile-dataflow/SKILL.md) and the matching
  backend reference.
- **Per-kernel latency, benchmark harnesses, or timing claims**
  -> [`benchmark-kernel`](../skills/benchmark-kernel/SKILL.md).
- **Profiler capture, trace export, offline analysis, or Nsight wrappers**
  -> [`gpu-profiler-analysis`](../skills/gpu-profiler-analysis/SKILL.md) and
  its relevant capture-mode reference.
- **Pipeline, backend, or buffer ownership changes** -> read
  [`ARCHITECTURE.md`](../../ARCHITECTURE.md) and the target pipeline/backend
  reference before editing.

Two repository hygiene gates apply to every row:

- Inspect `git status` before and after a scoped edit; preserve unrelated user
  changes.
- Do not edit kernel source while a TileLang compile or profiling job is active;
  wait for the job to finish and keep its artifact directory separate.
