# Profile Collection Commands

The exact `ncu` invocations to run on this cluster, in what order, and what each flag does. Everything here was exercised on the `acd_u` partition with Nsight Compute 2025.4.1 (`cuda/13.1` module).

---

## Prerequisites recap

- A run directory (`00-directory-layout.md`).
- `-lineinfo` in the build when per-line attribution is wanted (`02-harness-guide.md`).
- An **ncu-capable node**: `ACD1-10`, `ACD1-20`, `ACD1-21`, `ACD1-31`, `ACD1-40`, `ACD1-62`. Anywhere else: `ERR_NVGPUCTRPERM`.
- Kernel name known (`--kernel-name-base=function` matches the bare function name, so `regex:ffn_taskloop_kernel` works without the mangled signature).

Quick permission / name test, when in doubt (3–5 replays):
```bash
ncu --set basic --clock-control=none -k "regex:<kernel>" -c 1 "$PYTHON" -u <driver> ...
```

---

## Recipe 0: the Slurm job (this cluster's capture wrapper)

Every capture is a batch job. The pattern below is the one the skill was verified with (`artifacts/ncu-verify/job.sh`); copy it into the run dir as `job.sh` and edit the marked lines. Two ready-made wrappers of the same shape exist for the pi05 task-loop kernels — `lab/sbatch/profile_attn.sh` (one launch per mode in `ATTN_MODES`) and `lab/sbatch/profile_ffn.sh` (`PROFILE_KIND=ncu`) — writing to `profiles/<area>/`; point `PROFILE_OUT` at the run dir to keep the new convention.

```bash
#!/bin/bash
# Submit:  sbatch -w ACD1-20 artifacts/profile/<run>/job.sh     (ncu-capable nodes only)
#SBATCH --job-name=ncu-<run>
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/sbatch/_common.sh"     # cuda/13.1, gcc/13.3, venv, compat-lib rule
require_cuda
report_env
ncu --version | tail -1

RUN="${REPO_DIR}/artifacts/profile/<run>"                   # EDIT
TAG="<tag>"                                                 # EDIT
mkdir -p "${RUN}/reports"

export FFN_NVCC_DEFINES="-lineinfo"                         # EDIT: the backend's extra-flags hook

ncu --set full --clock-control=none --target-processes=all \
    --kernel-name-base=function --kernel-name="regex:ffn_taskloop_kernel" \
    --launch-skip=0 --launch-count=1 \
    --force-overwrite --export="${RUN}/reports/full_${TAG}" \
    "${PYTHON}" -u -m eval.correctness --target h100/pi05 --plan shipped \
  || echo "[ncu] non-zero exit; the report may still exist"

REP="${RUN}/reports/full_${TAG}.ncu-rep"
ls -la "${REP}"
ncu -i "${REP}" --page details > "${RUN}/reports/full_${TAG}.details.txt"
ncu -i "${REP}" --page raw --csv > "${RUN}/reports/full_${TAG}.raw.csv"
echo "[job] done $(date)"
```

Notes on the wrapper:

- `sbatch/_common.sh` is what makes the job survive both driver generations on the partition and points nvcc at the module GCC; do not bypass it.
- `--target-processes=all` is needed because the engine may spawn workers.
- Do not request clock changes here; repository job scripts inherit device state, and NCU uses `--clock-control=none`.
- Export `details.txt` / `raw.csv` in the same job: the login node's ncu differs from the one that wrote the report.
- `gpu-profiler-analysis`'s `run_local_profile.py --backend ncu` is the alternative when a manifest (git commit, driver, node) should be recorded automatically; pass ncu flags through its `--tool-arg`.

---

## Recipe 1: Full overview (first pass — mandatory)

```bash
ncu --set full --clock-control=none --target-processes=all \
    --kernel-name-base=function --kernel-name="regex:KERNEL" \
    --launch-skip=N --launch-count=1 \
    --force-overwrite --export="$PROFILE_RUN_DIR/reports/full_<tag>" \
    "$PYTHON" -u <driver> [args]
```

| Flag | Meaning |
|---|---|
| `--set full` | All built-in sections. **On 2025.4.1 this includes `PmSampling`, `PmSampling_WarpStates` and the per-PC warp sampler** — the report carries timelines and per-line stalls without extra sections. |
| `--clock-control=none` | Don't try to lock clocks (denied here; letting ncu try aborts the capture). Reports then warn "unmodified GPU clocks". |
| `--target-processes=all` | Follow child processes. |
| `--kernel-name-base=function` | Match on the bare function name, not the demangled signature. |
| `--kernel-name="regex:..."` | Only profile matching kernels. Every extra match multiplies replay time. |
| `--launch-skip N` / `--launch-count 1` | Skip warmup launches, profile one. |
| `--export` | Output path; `.ncu-rep` appended. `--force-overwrite` for reruns. |

Replay count on this host: 39–45 passes for a full-set capture of a persistent task-loop kernel. The unavailable `ctc__*` (NVLink) metrics warning is harmless on a single-GPU job.

**Which launch is which.** The parity drivers launch each mode once; a benchmark driver launches warmup + repeats. `--launch-skip` picks the steady-state launch; a report with several actions (several modes) is addressed with `--action N` in the helpers.

---

## Recipe 2: Source-level counters (optional on 2025.4.1)

The B200 workflow needs a separate `--set source --section SourceCounters` pass for per-PC samples. Here, a `--set full` report already has `smsp__pcsamp_*` with correlation ids (45 metrics, `warpgroup_arrive` included), and `hotspots` works on every existing report. Add the section only when the details page's Source Counters tables (L2 theoretical sectors excessive, per-instruction tables) are wanted:

```bash
ncu --set full --section SourceCounters --clock-control=none ...
```

Per-line attribution still requires `-lineinfo` in the build; without it, samples attribute to PC addresses and SASS.

---

## Recipe 3: Details page (rule summary)

No new capture — import an existing report (same ncu major version as the writer):

```bash
ncu -i reports/full_<tag>.ncu-rep --page details > reports/full_<tag>.details.txt
```

Each rule is rendered as

```
OPT   Est. Speedup: 35.27%
      On average, each warp of this workload spends 4.0 cycles being stalled waiting for
      sibling warps at a CTA barrier. ...
```

`report_query.py rules` gives the same list sorted by estimate, from Python, on the login node. **Read it first.** The engine is right about what it sees; the playbook says when its advice does not apply.

---

## Recipe 4: CSV / raw export (scripting)

```bash
ncu -i reports/full_<tag>.ncu-rep --page raw --csv > reports/full_<tag>.raw.csv
ncu -i reports/full_<tag>.ncu-rep --page source   > reports/full_<tag>.source.txt   # needs -lineinfo
```

Handy for `grep`/`awk`; the Python API (`04-python-api.md`) is easier for anything structured.

---

## Recipe 5: Targeted metrics only (fast recheck)

When you already know which metrics answer the question (did the fix land?), collect just those — one or two replay passes instead of ~40:

```bash
ncu --clock-control=none --target-processes=all \
    --kernel-name-base=function --kernel-name="regex:KERNEL" --launch-count=1 \
    --metrics \
    gpu__time_duration.sum,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__bytes_read.sum.pct_of_peak_sustained_elapsed,\
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
    smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio,\
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
    l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum \
    "$PYTHON" -u <driver> [args]
```

This answers "did the barrier ratio drop", not "what is the bottleneck".

---

## Recipe 6: A/B comparison (before vs after)

Two captures, same node class, same driver, same `--launch-skip`, different `<tag>`:

```bash
# before
... --export="$PROFILE_RUN_DIR/reports/full_v1" <driver with plan v1>
# after
... --export="$PROFILE_RUN_DIR/reports/full_v2" <driver with plan v2>
```

Then on the login node:

```bash
.venv/bin/python .claude/skills/ncu-report/scripts/report_query.py compare \
    $PROFILE_RUN_DIR/reports/full_v1.ncu-rep $PROFILE_RUN_DIR/reports/full_v2.ncu-rep
.venv/bin/python .claude/skills/ncu-report/scripts/analyze_reports.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/full_v1.ncu-rep --tag v1 \
    --report $PROFILE_RUN_DIR/reports/full_v2.ncu-rep --tag v2
```

Compare *counts and ratios* (sectors, instructions, stall mix, TMA bytes) before durations: with unpinned clocks, duration deltas under ~5 % are noise, and replay perturbation on a 10–30 µs kernel is comparable to the kernel itself.

---

## What each `--set` contains (2025.4.1, sm90)

```bash
ncu --list-sets ; ncu --list-sections
```

| Set | Sections | Replays | Use when |
|---|---|---|---|
| `basic` (default) | SpeedOfLight, LaunchStats, Occupancy | ~3–5 | Smoke test — does the name match, is the node allowed |
| `detailed` | basic + Scheduler, WarpState, Compute, Memory, InstructionStats | ~15 | Rarely the right middle ground |
| `full` | everything incl. PmSampling, PmSampling_WarpStates, SourceCounters' warp sampler | ~39–45 | **First pass. Always start here.** |
| `--section SourceCounters` on top | adds the per-instruction source tables | +few | Source-page tables wanted |

---

## Sampling controls

```bash
--pm-sampling-interval <ns>        # default auto → 1.5 µs here; ≥ 1000 ns on GA10x+; shorten for < 30 µs kernels
--warp-sampling-interval auto      # PC sampler period 2^(5+n) cycles; auto avoids buffer overflow
--warp-sampling-max-passes 5
```

The `PMSamplingData` rule fires when the interval exceeds 10 % of the workload duration — on this repo's 12–20 µs kernels that is the default, so expect under ten in-kernel timeline samples unless you shorten the interval.

---

## Profiling multiple kernel launches

```bash
ncu ... --launch-skip 5 --launch-count 3 ...     # skip 5 matches, profile the next 3 (three actions)
```

Under `--graph-profiling node` (the runner's default) each captured graph kernel is its own launch; use `--graph-profiling graph` only for whole-graph metrics.

---

## GPU frequency locking

Not available to users on this cluster (`nvidia-smi -lgc` → "does not have permission to change clocks"); ncu's own attempt (`--clock-control base`, the default) aborts the capture. Always pass `--clock-control=none`, state it in the report, and compare structure rather than time.

---

## Gotchas

- **`regex:` matches nothing**: check `cuobjdump --dump-function-names` on the cached `.so` (TileLang / torch extension caches under `.cache/`), and remember `--kernel-name-base=function` vs the default demangled base.
- **Report is 0 KB / missing**: the kernel never launched under the filter, the job landed on a non-ncu node, or clock control aborted it. Read `sbatch/logs/<name>_<jobid>.err`.
- **Two driver generations**: `_common.sh` handles the compat-library rule; if torch cannot see the GPU the job exits 75 in seconds — resubmit on another ncu-capable node.
- **Run time blows up**: every replay re-runs the kernel; a kernel that launches inside a long model step still replays only the kernel, but the engine build and weight load around it run once per job.
- **"Could not deploy stock section files to $HOME"**: set `HOME` writable.
