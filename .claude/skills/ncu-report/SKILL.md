---
name: ncu-report
description: Profile a CUDA kernel with Nsight Compute on H100 / sm90 and turn the report into a named bottleneck and a ranked optimization plan. Use this whenever a .ncu-rep exists or is about to be captured and someone asks why a kernel is slow, what its bottleneck is, what to optimize next, whether tensor cores / TMA / occupancy / coalescing are the problem, or to read NCU output — including Chinese phrasings ("profile 一下", "ncu 报告", "为什么慢", "下一步优化什么", "瓶颈在哪"). Also use it when writing a profiling harness, choosing ncu flags, parsing reports with the ncu_report Python API, or comparing two reports before/after a change. Covers the full loop (harness → collect on an ncu-capable Slurm node → parse on the login node → six analysis dimensions → diagnosis playbook → REPORT.md), the sm90 metric-name and stall-reason vocabulary verified on this cluster's ncu 2025.4.1, the Hopper-specific traps (wgmma `gmma` stalls, TMA traffic invisible to LSU metrics, persistent-kernel occupancy rules to overrule), and a B200 / sm100 appendix. Latency claims stay with benchmark-kernel; machine ceilings with hardware-unit-test.
---

# Skill: CUDA Kernel Profiling with Nsight Compute (H100 / sm90)

**When to use:** the user asks to profile a CUDA kernel, analyze its performance, find its bottleneck, or write an optimization plan backed by Nsight Compute data. Triggers include "profile X", "为什么这个 kernel 慢", "ncu report 说...", "下一步怎么优化", "帮我看一下这份 ncu 报告", "is it memory bound", "are the tensor cores busy".

**Target hardware (this repo):** NVIDIA H100 80GB HBM3 (sm_90a, CC 9.0, 132 SMs, 50 MB L2, 228 KB smem/SM). Nsight Compute **2025.4.1** from the `cuda/13.1` module captures every report here; all metric names in this skill are verified against real full-set sm90 reports from that toolchain. Generic advice is unmarked; sm90-specific notes are marked **sm90**; B200 / sm100 material lives in `references/08-b200-metric-names.md` and `blackwell-cuda-programming.md` for when a report from that GPU shows up.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing kernels are slow for exactly one reason that ncu names in ten seconds. Don't invent hypotheses before the report exists. Don't code a fix before the observed pattern has been matched to a known diagnosis. Don't write a wall of suggestions — rank them by evidence and expected impact. A conclusion without its metric values ("it's memory bound") is not a finding.

---

## Boundaries (who owns what)

| Concern | Owner | Why it is not this skill |
|---|---|---|
| Latency numbers | `benchmark-kernel` | ncu replays serialize and perturb; on a 10–30 µs kernel the distortion is most of the number. NCU explains *why*, CUPTI says *how fast*. |
| Machine ceilings | `hardware-unit-test` | NCU's speed-of-light is the datasheet frame; the reachable ceiling on this machine is a tagged measured constant (`scripts/constants.py --tag <t>`). |
| Trace / timeline capture (nsys, torch) | `gpu-profiler-analysis` | Its runner also drives `ncu` (`--backend ncu`) and records the manifest; this skill decides *what* to capture and reads the result. |
| Next optimization move | `kernel-design` (`references/wiki/README.md`) | The wiki is symptom-indexed; this skill produces the symptom. |

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** — `artifacts/profile/<run_name>/` with `harness/`, `reports/`, `analysis/`, `REPORT.md`. One directory per run, never reuse one. See [`references/00-directory-layout.md`](references/00-directory-layout.md). Existing reports under `profiles/<area>/` and `artifacts/ncu-verify/` are the legacy homes; read them in place.

1. **Frame the question.** Which kernel, which production shape, which dispatch path, answering what? This repo is fixed-workload: the shapes the Target runs are the only shapes worth profiling. See [`references/01-workflow.md`](references/01-workflow.md) Phase 0.5.

2. **Pick the profiled process.** Prefer the engine's own drivers (`python -m eval.correctness` / `python -m benchmarks kernels --site <site>`) so the kernel sees production inputs; build a standalone harness only for a kernel outside the pipeline. Get `-lineinfo` into the build through the backend's extra-nvcc-flags hook (`ATTN_NVCC_DEFINES` / `FFN_NVCC_DEFINES` / `ENC_ATTN_NVCC_DEFINES`) — the build cache is keyed on flags, so the instrumented `.so` never displaces production. See [`references/02-harness-guide.md`](references/02-harness-guide.md).

3. **Collect on an ncu-capable node.** Counters are permission-gated to `ACD1-10/20/21/31/40/62`; clock locking is denied, so pass `--clock-control=none`. `--set full` on 2025.4.1 already carries per-PC stall samples and the PM-sampling timelines. See [`references/03-collection.md`](references/03-collection.md).

4. **Parse on the login node** — no GPU, no job. Start with the quick look:

   ```bash
   .venv/bin/python .claude/skills/ncu-report/scripts/report_query.py summary  <rep> --action 0
   .venv/bin/python .claude/skills/ncu-report/scripts/report_query.py rules    <rep>            # engine, by Est. Speedup
   .venv/bin/python .claude/skills/ncu-report/scripts/report_query.py hotspots <rep> --top 15   # per-PC / per-line stalls
   .venv/bin/python .claude/skills/ncu-report/scripts/report_query.py compare  <a> <b>
   ```

   then archive with the run-dir helpers (`analyze_reports.py`, `extract_stall_hotspots.py`, `plot_timeline.py`). See [`references/04-python-api.md`](references/04-python-api.md).

5. **Walk the six analysis dimensions** — launch/occupancy, balance, stalls, tensor pipe, timeline, memory. Every one, every time; only one or two dominate, but you don't know which until you've looked. See [`references/05-analysis-dimensions.md`](references/05-analysis-dimensions.md).

6. **Match patterns to the diagnosis playbook.** Signal → cause → first fix → exception, with the sm90 entries that say when to *overrule* an NCU rule. See [`references/06-diagnosis-playbook.md`](references/06-diagnosis-playbook.md).

7. **Write `REPORT.md`** in the run directory: capture conditions, headline table, per-dimension evidence, 3–5 ranked priorities, caveats. See [`references/07-report-template.md`](references/07-report-template.md). Findings that change a decision get promoted into an Agent Note; the run directory is evidence, not the record.

---

## Critical lessons (don't skip)

1. **NCU's occupancy rules misfire on persistent kernels.** On a by-design 1 CTA/SM task loop holding a large smem ring, `TheoreticalOccupancy` / `IssueSlotUtilization` / `LaunchConfiguration` will estimate double-digit gains from what is a design property. Judge warp-specialized persistent kernels by stall structure, `gmma`/tensor activity and the benchmark — never by occupancy percent. (Playbook pattern O.)

2. **TMA traffic is invisible to the LSU metrics** (**sm90**). A kernel whose real traffic rides TMA barely registers in `l1tex__t_sectors_pipe_lsu_mem_global_op_ld`; the coalescing rules then judge only the scalar side path. Read what actually moved from `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum`, `dram__bytes_*` and `lts__*`.

3. **Hopper has two wgmma stall names** (**sm90**). The aggregate ratio family has `gmma` (warps waiting on wgmma completion); the per-PC sampler has `warpgroup_arrive` (`WARPGROUP.ARRIVE` / `WARPGROUP.WAIT`) instead. Neither exists on the other side. Both point at the wgmma group discipline, not at memory.

4. **`sm__ops_path_tensor_op_hmma_*` does not count wgmma** (**sm90**). It reads 0 on a kernel that is 10 % busy in `sm__pipe_tensor_op_hmma_cycles_active`. Judge wgmma by the pipe-cycles metrics and `sm__inst_executed_pipe_tensor_op_gmma`, never by the ops-path counters.

5. **Metric names drift by ncu version.** Everything here is verified under 2025.4.1 on sm90 (2 320 names in a full-set report). On any other version, re-enumerate with `action.metric_names()` before trusting a name (`references/08-sm90-metric-names.md` shows how, and what differs from B200 / 2026.1).

6. **Reports parse on the login node.** The repo venv plus the `ncu_report` module inside any Nsight install reads `.ncu-rep` without a GPU; the helpers auto-locate a compatible module. Never burn a GPU job to read a report.

7. **PM sampling is the only way to see tail effects, and it is already in your report.** On 2025.4.1 `--set full` includes `PmSampling` and `PmSampling_WarpStates`; the timelines appear as `<UNIT>.TriageCompute.<metric>` and `warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>` series, not under the `pmsampling:` prefix the B200 docs use. On a 12 µs kernel the default 1.5 µs interval yields under ten in-kernel samples — shorten it with `--pm-sampling-interval` when the shape over time matters.

8. **Read the rule engine first, then overrule it deliberately.** `rules` sorts by `Est. Speedup`; the engine is right about what it *sees*. Never sum the estimates (rules overlap), and never chase a 1 % rule while an 87 % rule sits unexplained.

9. **Don't delegate understanding.** Run the queries yourself, cite the metric values, name the source line. "`long_scoreboard = 12.2` warps/issue, 230 of 464 samples at `barrier.h:426`, DRAM read at 40 % of peak → latency-bound on the ring wait, not bandwidth-bound" is a finding; "the profile shows it's memory-bound" is not.

---

## File index

### Reference docs

| File | Purpose |
|---|---|
| [`references/00-directory-layout.md`](references/00-directory-layout.md) | **Read first.** Run-directory convention (`artifacts/profile/<run>/`), naming, what not to keep |
| [`references/01-workflow.md`](references/01-workflow.md) | End-to-end checklist from "user request" to "final report", with this cluster's phases |
| [`references/02-harness-guide.md`](references/02-harness-guide.md) | Engine drivers vs standalone harness; `-lineinfo` hooks; the `sm_90a` template |
| [`references/03-collection.md`](references/03-collection.md) | ncu recipes: Slurm job, node list, `--clock-control=none`, sets, sampling, A/B |
| [`references/04-python-api.md`](references/04-python-api.md) | `ncu_report` patterns, with the 2025.4.1 API shapes (rule dicts, `sass_by_pc(pc)`, timeline instances) |
| [`references/05-analysis-dimensions.md`](references/05-analysis-dimensions.md) | Six dimensions with sm90 metric names, thresholds, wave math, stall table |
| [`references/06-diagnosis-playbook.md`](references/06-diagnosis-playbook.md) | Pattern → cause → fix (A–N generic, O–U sm90 / this repo's kernel styles) |
| [`references/07-report-template.md`](references/07-report-template.md) | Structure of `REPORT.md` |
| [`references/08-sm90-metric-names.md`](references/08-sm90-metric-names.md) | Verified sm90 vocabulary: names that work, names that don't, diff vs B200 |
| [`references/08-b200-metric-names.md`](references/08-b200-metric-names.md) | Upstream sm100 reference, kept for cross-GPU reports |
| [`references/09-common-issues.md`](references/09-common-issues.md) | Permissions, clock control, sampling, JIT `-lineinfo`, version mismatches |
| [`hopper-cuda-programming.md`](hopper-cuda-programming.md) | Companion: H100 architecture, Hopper features, 16 principles ↔ NCU signals |
| [`blackwell-cuda-programming.md`](blackwell-cuda-programming.md) | Companion (upstream, Chinese): B200 principles, for a sm100 report |

### Scripts

| File | Purpose |
|---|---|
| [`scripts/report_query.py`](scripts/report_query.py) | Login-node quick look: `summary` / `rules` / `hotspots` / `compare`, any `--action` |
| [`scripts/analyze_reports.py`](scripts/analyze_reports.py) | Archive all metrics + curated key metrics per tag; side-by-side compare |
| [`scripts/extract_stall_hotspots.py`](scripts/extract_stall_hotspots.py) | Per-line stall aggregation via `action.source_info(pc)` |
| [`scripts/plot_timeline.py`](scripts/plot_timeline.py) | ASCII timelines from the PM / warp sampling series (tail effects, sawtooth) |
| [`scripts/ncu_utils.py`](scripts/ncu_utils.py) | Shared: module discovery, safe access, key-metric lists, version-tolerant rule parsing |
| [`scripts/harness_template.cu`](scripts/harness_template.cu) | Standalone harness skeleton (`sm_90a`, `-lineinfo`) |
| [`scripts/safetensors_loader.h`](scripts/safetensors_loader.h) | Header-only safetensors reader for real-tensor harness inputs |

---

## Host integration (swap these when porting)

- **Nodes**: ncu counters only on `ACD1-10/20/21/31/40/62` (`sbatch -w`); elsewhere `ERR_NVGPUCTRPERM`. Clock locking is denied cluster-wide → `--clock-control=none`, so cross-run deltas under ~5 % are noise.
- **Toolchain**: `sbatch/_common.sh` loads `cuda/13.1` (ncu 2025.4.1) and the repo venv; `sbatch/run.sbatch` runs any `CMD`; `sbatch/kernel_template.sh` builds the `kernel-design` templates with `sm_90a`.
- **Capture wrappers**: `lab/sbatch/profile_attn.sh`, `lab/sbatch/profile_ffn.sh` (`PROFILE_KIND=ncu`) and the verified `artifacts/ncu-verify/job.sh`; the job template in `references/03-collection.md` is the generic form.
- **`-lineinfo` hooks**: `ATTN_NVCC_DEFINES`, `FFN_NVCC_DEFINES`, `ENC_ATTN_NVCC_DEFINES` (space-separated extra nvcc flags; hash-keyed build cache).
- **Query interpreter**: `.venv/bin/python` (3.12); `ncu_utils.py` scans `/data/apps/cuda/*/nsight-compute-*` and `/usr/local/cuda-*/nsight-compute-*` for `ncu_report`, `NCU_PYTHON_DIR` overrides.
- **Output homes**: `artifacts/profile/<run>/` (new runs), `profiles/<area>/`, `artifacts/ncu-verify/` (legacy) — all git-ignored; name reports with the Slurm job id so provenance survives.

---

## Related skills

- `gpu-profiler-analysis` for nsys / torch timelines and the manifest-writing ncu runner.
- `benchmark-kernel` for the latency number the report must never replace.
- `hardware-unit-test` for the ceiling under any "X % of peak" claim.
- `kernel-design` for the move that answers the symptom, and the compilable sm90 templates.
