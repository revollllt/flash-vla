---
name: ncu-report
description: Read an Nsight Compute report and turn it into a named bottleneck and a ranked next optimization move, on H100 sm90. Use when a .ncu-rep exists or was just captured and someone asks why a kernel is slow, what its bottleneck is, what to optimize next, whether tensor cores / TMA / occupancy are the problem, or to interpret NCU output ("ncu 报告", "为什么慢", "下一步优化什么"). Covers querying reports on the login node, per-line and per-PC stall hotspots, NCU's rule engine, and the sm90 metric-name and stall-reason vocabulary this repo has verified. Capture itself belongs to gpu-profiler-analysis; latency claims to benchmark-kernel.
---

# NCU Report Interpretation (H100 / sm90)

Profile first, diagnose second, plan third — never guess. An underperforming
kernel is usually slow for one dominant reason the report already names; the
job is to find it, cite the metric values that prove it, and rank the fixes.
A conclusion without its numbers ("it's memory bound") is not a finding.

## Boundaries

- **Capture** is owned by `gpu-profiler-analysis` (its `references/ncu-backend.md`)
  through the host's capture wrappers (see Host integration). Two host facts
  gate it: counters work only on the ncu-capable nodes the wrappers name, and
  captures run with `--clock-control=none` because the host cluster denies
  clock locking.
- **Latency claims** come from `benchmark-kernel`, never from NCU: replay
  passes serialize and perturb, and on microsecond-scale kernels the
  distortion is most of the number. NCU explains *why*; CUPTI says *how fast*.
- **Ceilings** come from `hardware-unit-test` tags, not from NCU percentages:
  NCU's speed-of-light is the datasheet frame, and this machine's reachable
  ceilings are the measured constants (`scripts/constants.py`).

## Quickstart

1. Have a report. Fresh capture: submit a host capture wrapper on an
   ncu-capable node. For per-source-line stalls the build needs `-lineinfo`,
   passed through the host's extra-nvcc-flags hook (Host integration) so the
   instrumented build caches separately and production stays untouched.
2. Query it — login node, no GPU:

   ```bash
   .venv/bin/python .claude/skills/ncu-report/scripts/report_query.py \
       summary  profiles/<area>/<report>.ncu-rep --action 0
   #   rules    <rep>            NCU's rule engine, sorted by Est. Speedup
   #   hotspots <rep> --top 15   per-PC/per-line stall attribution
   #   compare  <a> <b>          before/after on the headline metrics
   ```

3. Read `rules` first — the engine is right about *what* it sees — then walk
   the six lenses in `references/dimensions-sm90.md` and match the signals in
   `references/playbook-sm90.md`, which also says when a rule must be
   overruled (see lesson 1).
4. Deliver findings as: metric = value → meaning → move, ranked by expected
   impact, at most 3–5 items. Everything else is supporting artifacts.

## Critical lessons (sm90 / this repo)

1. **NCU's occupancy rules misfire on persistent kernels.** On a by-design
   1 CTA/SM persistent kernel holding a large smem ring, the top rules will
   demand double-digit occupancy gains — that is the design, not a defect.
   Judge warp-specialized persistent kernels by barrier/stall structure and
   pipe activity, never by occupancy percent.
2. **TMA traffic is invisible to the LSU load metrics.** A kernel whose real
   traffic rides TMA barely registers in `global_op_ld` sectors; read
   `dram__bytes_*` and `lts__*` for what actually moved. Coalescing metrics
   only judge the scalar-load side path.
3. **sm90 has a `gmma` stall reason** — warps waiting on wgmma. It exists in
   the aggregate ratios and is the first place to look when a math warpgroup
   seems starved.
4. **Metric names drift by ncu version.** Everything this skill names is
   verified under the host toolchain's ncu (**2025.4.1**); on any other
   version, re-enumerate before trusting a name
   (`references/metrics-sm90.md` shows how).
5. **Reports parse on the login node.** The venv python + the ncu_report
   module inside any Nsight install read `.ncu-rep` without a GPU;
   `report_query.py` auto-locates a compatible module. Don't burn a compute
   job to read a report.

## Host integration (swap these when porting)

- **Capture wrappers**: `sbatch/profile_attn.sh` / `profile_ffn.sh`; counter
  access is permission-gated to the ncu-capable nodes their headers name.
- **`-lineinfo` hook**: the host build takes extra nvcc flags via an env hook
  (`ATTN_NVCC_DEFINES` on the attention backend); the build cache is keyed on
  flags, so instrumented builds never displace production ones.
- **Query interpreter**: the repo venv python; `scripts/report_query.py`
  scans the host's Nsight installs for `ncu_report`, `NCU_PYTHON_DIR`
  overrides.
- **Output homes**: reports under `profiles/<area>/` or `artifacts/` (both
  git-ignored), named with the Slurm job id so provenance survives.

## Files

| File | Read when |
|---|---|
| `references/workflow.md` | running a full capture→diagnose→report pass |
| `references/dimensions-sm90.md` | walking a report: the six lenses, thresholds, wave math |
| `references/playbook-sm90.md` | mapping a signal to a cause and the next move |
| `references/metrics-sm90.md` | needing an exact metric name, or on a new ncu version |
| `references/python-api.md` | writing custom extraction beyond `report_query.py` |
| `scripts/report_query.py` | the query tool the quickstart uses |
