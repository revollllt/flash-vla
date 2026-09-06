# Profiling Workflow — End-to-End

The complete checklist from "user asks to profile" to "final report", with the phases this cluster imposes (Slurm, restricted counters, no clock locking, GPU-less login node). Every step has a short rationale and a pointer to the detailed doc.

---

## Phase 0 — Create a new run directory

**Always start here.** See [`00-directory-layout.md`](00-directory-layout.md).

```bash
PROFILE_RUN_DIR=artifacts/profile/<descriptive_run_name>      # e.g. ffn-gu-baseline
mkdir -p "$PROFILE_RUN_DIR"/{reports,analysis}
```

A new version of a kernel is a **new** run. The same kernel on a different plan or call site is a new run, or at minimum a new `<tag>` inside the same run. Everything produced later is written only under `$PROFILE_RUN_DIR`.

---

## Phase 0.5 — Frame the problem (before any tools)

Write one sentence: *which kernel, at which shape, answering what.* "Why is this kernel slow" is not answerable; "at the production shape, is the fused FFN task loop starved by its TMA producer or by wgmma issue" is.

1. **Which kernel?** Exact name for `--kernel-name`. On this host the persistent kernels are plain C symbols (`ffn_taskloop_kernel`, `attn_taskloop_kernel`), TileLang kernels carry generated names (`cuobjdump --dump-function-names` on the cached `.so`, or a first `--set basic` pass, tells you).
2. **Which shape?** This repo is fixed-workload by design: the Target's production shapes are the only ones worth profiling, and the engine's own drivers already run them. Profile *through* those drivers, not a synthetic shape.
3. **Which dispatch path / mode?** A parity script that runs several modes profiles each launch as its own action. A task-loop kernel with several task kinds averages them in one action — say which task mix the capture represents.
4. **What question?** "Latency-bound or bandwidth-bound?", "did the ring-depth change move the barrier stall?", "is wgmma issuing or starved?".
5. **What is the baseline?** A previous run, the reference plan, or a cuBLAS/TileLang sibling — profile it too when the question is comparative.

If any of 1–4 is unclear, ask before profiling. A wrong capture costs a GPU job and an hour.

---

## Phase 1 — Environment check (once per toolchain, on the login node)

```bash
# the venv python that parses reports (3.12)
.venv/bin/python --version

# ncu_report modules available on this host — the helpers pick the newest that loads the report
ls -d /data/apps/cuda/*/nsight-compute-* /usr/local/cuda-*/nsight-compute-*
#   /data/apps/cuda/13.1/nsight-compute-2025.4.1   ← what the cuda/13.1 module captures with
#   /usr/local/cuda-13.3/nsight-compute-2026.2.1   ← newer; reads 2025.4.1 reports too

# which nodes can run ncu right now
sinfo -N -p acd_u -n ACD1-10,ACD1-20,ACD1-21,ACD1-31,ACD1-40,ACD1-62 -o "%N %t"
```

Counters are permission-gated: outside those six nodes ncu fails with `ERR_NVGPUCTRPERM`. Clock locking is denied everywhere (`The current user does not have permission to change clocks`), so every capture runs `--clock-control=none` and reports carry the "unmodified GPU clocks" warning. Consequence: cross-run deltas under ~5 % are noise; compare structure, not durations. See [`09-common-issues.md`](09-common-issues.md).

---

## Phase 2 — Choose the profiled process

**Option A (default here): the engine's drivers.** `python -m eval.correctness --target h100/pi05 --plan <plan>` or `python -m benchmarks kernels --target h100/pi05 --plan <plan> --site <site>` launch the kernel on production inputs; `--kernel-name` + `--launch-skip/--launch-count` pick one launch out of the warmup/repeat loop. Get `-lineinfo` into the build through the backend's extra-flags hook (`ATTN_NVCC_DEFINES="-lineinfo"` etc.); the build cache is hash-keyed on flags so production stays untouched.

**Option B: a standalone harness.** For a kernel that is not yet in the pipeline, a `kernel-design` template, or an ablation that needs inputs the drivers can't produce. `sbatch/kernel_template.sh` builds templates with `-gencode arch=compute_90a,code=sm_90a`; the skeleton in `scripts/harness_template.cu` is the starting point for anything else. See [`02-harness-guide.md`](02-harness-guide.md).

Either way, **make sure `-lineinfo` is in the nvcc command** when you want per-line attribution. Without it hotspots still work, attributed to PCs/SASS.

---

## Phase 3 — Collect (a Slurm job on an ncu-capable node)

One `--set full` pass is the mandatory first run; on 2025.4.1 it already includes per-PC warp sampling and the PM-sampling timelines, so the separate `--set source` pass the B200 workflow needs is optional here. Details and the job template in [`03-collection.md`](03-collection.md).

```bash
ncu --set full --clock-control=none --target-processes=all \
    --kernel-name-base=function --kernel-name="regex:<kernel>" \
    --launch-skip=<N> --launch-count=1 \
    --force-overwrite --export="$PROFILE_RUN_DIR/reports/full_<tag>" \
    "$PYTHON" -u -m eval.correctness --target h100/pi05 --plan <plan>
# export the human pages while still on the node
ncu -i "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page details > "$PROFILE_RUN_DIR/reports/full_<tag>.details.txt"
ncu -i "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page raw --csv > "$PROFILE_RUN_DIR/reports/full_<tag>.raw.csv"
```

Budget: a full-set pass is ~39–45 replays; on a 10–30 µs kernel the capture itself is seconds, the engine build and model load around it are minutes. Every extra matched launch multiplies the replay cost — filter tight.

---

## Phase 4 — Extract structured data (login node)

Do not eyeball the CLI output. Parse in Python so you can compare, aggregate and archive. See [`04-python-api.md`](04-python-api.md) and the helpers in [`../scripts/`](../scripts/).

| Artifact | Tool | What it tells you |
|---|---|---|
| terminal quick look | `report_query.py summary / rules / hotspots` | headline SOL, launch, stall mix, memory pattern; the engine's ranked rules; per-line stalls |
| `metrics_key_<tag>.txt` | `analyze_reports.py` | ~110 curated metrics (launch, SOL, occupancy, stalls, sectors, TMA, wgmma) |
| `metrics_all_<tag>.json` | `analyze_reports.py` | all ~2 300 metrics, archive |
| `compare_<a>_vs_<b>.txt` | `analyze_reports.py` (two tags) or `report_query.py compare` | side-by-side |
| `stall_hotspots_<tag>.txt` | `extract_stall_hotspots.py` | source lines ranked by stall samples, per-reason top lines |
| `pm_timeline_plots.txt` | `plot_timeline.py` | ASCII time-series of warp states and unit throughput |
| `full_<tag>.details.txt` | exported on the node | NCU's rule text with `Est. Speedup` |

Save everything under `$PROFILE_RUN_DIR/analysis/`.

---

## Phase 5 — Diagnose

Read `rules` first — the engine names what it *sees* — then walk the six dimensions of [`05-analysis-dimensions.md`](05-analysis-dimensions.md) in order:

1. **Launch geometry & occupancy** — enough CTAs? occupancy limited by registers / smem / barriers? *Persistent kernels pin this by design.*
2. **Balance across SMs (tail effect)** — per-SM active cycles, timeline shape, task-queue drain.
3. **Stall reasons + per-line hotspots** — which reason dominates, at which line; on sm90 read `gmma` / `warpgroup_arrive` alongside `barrier` and `long_scoreboard`.
4. **Tensor pipe** — is wgmma issuing, and against which ceiling.
5. **Utilization over time** — flat, tail, sawtooth, ramp.
6. **Memory pattern** — TMA bytes vs LSU sectors, L1/L2 hit, DRAM %, store fill, spill.

For each dimension write down the observed signal *and the metric value that produced it*. Then map the signal set to [`06-diagnosis-playbook.md`](06-diagnosis-playbook.md), which carries the cases where the rule engine's advice must be overruled for warp-specialized persistent kernels.

Two ranking rules: fix the biggest signal first (an idle SM or a tail dwarfs a coalescing nit), and never sum `Est. Speedup` values — rules overlap.

---

## Phase 6 — Write the report

Structure in [`07-report-template.md`](07-report-template.md). Key elements:

1. **Setup**: node, job id, ncu version, driver command, plan/site/mode, whether `-lineinfo` was on, `--action` → mode map. A number whose capture cannot be reproduced is not evidence.
2. **Headline table**: duration (diagnostic only), SM / memory / DRAM SOL, occupancy, tensor pipe, waves.
3. **Per-dimension analysis** with evidence (metric values + rule text).
4. **Optimization directions** ranked by expected impact — `metric = value → meaning → move`, 3–5 items, each naming the next capture that would confirm it.
5. **Confidence & caveats** — including "unpinned clocks, < 5 % is noise" whenever two runs are compared.

Findings that change a decision go into an Agent Note (`.agents/notes/`), with the job id and command; the run directory stays as evidence.

---

## Anti-patterns to avoid

- ❌ **"ncu says memory throughput is 14 %"** — without the metric name, mode and kernel this is un-actionable. Metric + value + meaning, always.
- ❌ **Profiling a shape the pipeline never runs.** The parity and benchmark drivers exist so you don't have to invent one.
- ❌ **Obeying an occupancy rule on a kernel whose design pins occupancy** (playbook pattern O).
- ❌ **Dumping the details page into the report.** Extract, interpret, write.
- ❌ **Reading an ncu duration as a latency claim.** Replay serializes and perturbs; `benchmark-kernel` owns the number.
- ❌ **Comparing reports captured under different ncu versions or nodes as one series.** Say which node and version each came from.
- ❌ **Missing the #1 finding because a smaller one was easier to explain.** Rank by magnitude.
