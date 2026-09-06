# Scripts

Reusable code for profiling harnesses and report analysis. See `../SKILL.md` for context. Everything Python runs on the **login node** with the repo venv (`.venv/bin/python`, 3.12) — no GPU, no job.

## Python

| File | Purpose |
|---|---|
| `report_query.py` | Terminal quick look at one report: `summary` (headline SOL, launch, stall mix, memory pattern), `rules` (engine by `Est. Speedup`), `hotspots` (per-PC / per-line stalls), `compare` (A/B on headline + top stalls). `--action N` picks the launch. |
| `ncu_utils.py` | Shared: `ncu_report` discovery (newest install first, `NCU_PYTHON_DIR` override), `load_action`, `safe`, per-instance / per-PC / timeline accessors, `key_metrics_for(action)` (portable core + sm90 or sm100 additions), version-tolerant `rule_speedups`. |
| `analyze_reports.py` | Archive all metrics + curated key metrics + rules per tag into `<run>/analysis/`; side-by-side compare across tags. |
| `extract_stall_hotspots.py` | Per-PC samples → per-source-line hotspots with a per-reason breakdown (`--sass` to see the instruction at each hot PC when there is no `-lineinfo`). |
| `plot_timeline.py` | ASCII timelines of the sampled series (`warpsampling:*`, `*.TriageCompute.*`, or `pmsampling:*` on other ncu versions), trimmed to the kernel's window. `--list` shows what a report holds. |

### Typical workflow

```bash
HELPERS=.claude/skills/ncu-report/scripts
export PROFILE_RUN_DIR=artifacts/profile/<run_name>

# quick look
.venv/bin/python $HELPERS/report_query.py summary  $PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep --action 0
.venv/bin/python $HELPERS/report_query.py rules    $PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep
.venv/bin/python $HELPERS/report_query.py hotspots $PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep --top 15

# archive + compare
.venv/bin/python $HELPERS/analyze_reports.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep --tag <tag1> \
    --report $PROFILE_RUN_DIR/reports/full_<tag2>.ncu-rep --tag <tag2>

# per-line stall hotspots (full-set report; -lineinfo build for source lines)
.venv/bin/python $HELPERS/extract_stall_hotspots.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep --tag <tag1>

# ASCII timelines
.venv/bin/python $HELPERS/plot_timeline.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep --tag <tag1>
```

`ncu_utils.py` scans `/data/apps/cuda/*/nsight-compute-*` and `/usr/local/cuda-*/nsight-compute-*` for `ncu_report`, newest first, and falls through when a module cannot read the report. Pin one with:

```bash
export NCU_PYTHON_DIR=/data/apps/cuda/13.1/nsight-compute-2025.4.1/extras/python
```

## C++ / CUDA

| File | Purpose |
|---|---|
| `harness_template.cu` | Starting point for a standalone profiling harness (`sm_90a`, `-lineinfo`). Copy into the run's `harness/`, fill in the `TODO(you)` sections. |
| `safetensors_loader.h` | Header-only safetensors reader (no external deps) for real workload tensors. |

### Typical harness setup

```bash
mkdir -p artifacts/profile/<run>/harness && cd artifacts/profile/<run>/harness
cp .claude/skills/ncu-report/scripts/harness_template.cu my_kernel_harness.cu
cp .claude/skills/ncu-report/scripts/safetensors_loader.h .
# edit my_kernel_harness.cu: include the kernel, fill in main()
# build ON a compute node (the login node has no GPU and a different driver):
nvcc -ccbin "$(command -v g++)" -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -lineinfo \
     -I third_party/cutlass/include my_kernel_harness.cu -o my_kernel_harness -lcuda
```

For a `kernel-design` template, `sbatch/kernel_template.sh` already builds and runs it with the right line.
