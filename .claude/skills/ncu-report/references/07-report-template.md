# Final Report Template

The report is the deliverable. Everything else (`.ncu-rep`, Python artifacts, CSVs) is evidence. Structure matters: a busy reader should see the top findings in 30 seconds and be able to drill into details if they want.

Save as `$PROFILE_RUN_DIR/REPORT.md` (`artifacts/profile/<run_name>/REPORT.md`). Findings that change a decision are then promoted into an Agent Note with the job id and command; the run directory stays as evidence.

---

## Template

```markdown
# `<kernel_name>` Profiling Report

**Kernel:** `<exact kernel name or template instantiation>`
**Target GPU:** NVIDIA H100 80GB HBM3 (132 SM, CC 9.0)   (or whatever is actually being profiled)
**Node / job:** ACD1-<nn> / Slurm <jobid>   (ncu-capable nodes only; clocks unpinned, `--clock-control=none`)
**Nsight Compute:** 2025.4.1 (from the job log's `ncu --version`; `report.get_version()` only names the parsing module)
**Build:** `-lineinfo` via `<HOOK>_NVCC_DEFINES` (or the exact nvcc line for a standalone harness)
**Profile date:** YYYY-MM-DD
**Run directory:** `artifacts/profile/<run_name>/`

---

## 0. Profiling setup

> How exactly did we get these numbers? Required for reproducibility.

- Profiled process: the engine driver (`eval.correctness` / `benchmarks kernels --site`) with plan and options, or `artifacts/profile/<run_name>/harness/*.cu`. Why this one.
- Shapes / modes: the Target shape tuple; for a multi-mode capture the `--action` index → mode map.
- Launch selection: `--kernel-name`, `--launch-skip`, `--launch-count`.
- Metric-name caveats: names that differ from the stock docs (sm90 / 2025.4.1: `references/08-sm90-metric-names.md`).
- Ceilings used: the `hardware-unit-test` tags behind every "% of peak" judgement.

Minimal runnable command listing:

    # Submit (ncu-capable node)
    sbatch -w ACD1-20 artifacts/profile/<run_name>/job.sh

    # Inside job.sh (after sourcing sbatch/_common.sh; -lineinfo via the backend hook)
    export FFN_NVCC_DEFINES="-lineinfo"
    ncu --set full --clock-control=none --target-processes=all \
        --kernel-name-base=function --kernel-name="regex:<kernel>" \
        --launch-skip=<N> --launch-count=1 \
        --force-overwrite --export=artifacts/profile/<run_name>/reports/full_<tag> \
        "$PYTHON" -u -m eval.correctness --target h100/pi05 --plan <plan>
    ncu -i artifacts/profile/<run_name>/reports/full_<tag>.ncu-rep --page details > .../full_<tag>.details.txt

    # Parse (login node)
    .venv/bin/python .claude/skills/ncu-report/scripts/report_query.py summary  .../full_<tag>.ncu-rep
    .venv/bin/python .claude/skills/ncu-report/scripts/analyze_reports.py --run-dir artifacts/profile/<run_name> \
        --report .../full_<tag>.ncu-rep --tag <tag>

### Artifacts

    artifacts/profile/<run_name>/
    ├── REPORT.md                       ← this file
    ├── job.sh                          ← the capture job (or manifest.json from the runner)
    ├── harness/...                     ← only for a standalone harness
    ├── reports/                        ← raw .ncu-rep + details.txt + raw.csv
    └── analysis/                       ← extracted metrics, hotspots, timelines

---

## 1. Headline numbers

> A single table that tells the whole story at a glance.

| Metric | `<tag1>` | `<tag2>` | Source |
|---|---:|---:|---|
| **Duration** (diagnostic only — not a latency claim) | X µs | Y µs | `gpu__time_duration.sum` |
| SM throughput (% peak) | X% | Y% | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory throughput (% peak) | X% | Y% | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM throughput (% peak) | X% | Y% | `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` |
| L1 hit rate | X% | Y% | `l1tex__t_sector_hit_rate.pct` |
| L2 hit rate | X% | Y% | `lts__t_sector_hit_rate.pct` |
| Tensor pipe (`_active`) | X% | Y% | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| TMA bytes loaded | X MB | Y MB | `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` |
| Reg / thread | X | Y | `launch__registers_per_thread` |
| Theoretical / Achieved occupancy | X% / Y% | ... | |
| Waves / SM | X | Y | `launch__waves_per_multiprocessor` (1 CTA/SM persistent = by design) |

**One-line read:** <"The kernel runs at X% of peak SM throughput — it's latency-bound on Y, not DRAM-BW-bound."> — this is the punchline.

---

## 2. Per-dimension analysis

> Walk through the six analysis dimensions, cite metrics, state findings.

### 2.1 SM occupancy & launch geometry
<grid size, block size, waves/SM, occupancy, register/shared-mem limits, wave math>

### 2.2 Thread-block balance (tail effect)
<per-SM active cycles, PM timeline shape, input distribution imbalance ratios>

### 2.3 Instruction-level stall analysis
<stall breakdown %, top source-line hotspots (cite file:line + samples + stall type); on sm90 name `gmma` / `warpgroup_arrive` alongside barrier / long_scoreboard>

### 2.4 Tensor Core utilization
<`_active` value vs the mma unit's tagged ceiling for this tile shape, or "0%, n/a">

### 2.5 SM utilization timeline
<shape: flat-high / flat-low / tail / sawtooth — reference the ASCII plot in analysis/>

### 2.6 Memory access pattern
<TMA bytes vs LSU sectors, L1/L2 hits, DRAM % vs the measured ceiling tag, store efficiency, register spill>

### 2.7 Additional findings
<items from NCU rule engine not otherwise mentioned — each with the rule's `Est. Speedup: X%`>

### 2.8 Rules overruled
<rules whose advice does not apply to this kernel style (e.g. occupancy rules on a persistent task loop) and why — playbook pattern letter>

---

## 3. Summary diagnosis

| Factor | `<tag1>` | `<tag2>` | Impact |
|---|---|---|---|
| <factor 1> | <status> | <status> | <ranked impact> |
| <factor 2> | ... | ... | ... |

---

## 4. Optimization directions (ranked by impact)

> Each priority: name the change, cite evidence, estimate magnitude, flag effort.

### Priority 1 — <one-line name>

<what to do, concretely, with line numbers / function names from the existing kernel>

**Evidence:**
- <metric + value>
- <NCU rule + est. speedup>

**Expected impact:** <X% on this kernel, bounded by `[tag]`>, <which call sites benefit>

**Confirm with:** <the targeted `--metrics` recapture or the benchmark case that shows it landed>

**Effort:** <low/medium/high + rough description of the code change>

### Priority 2 — ...

<same structure>

### Priority 3 — ...

(Stop at 3-5. More dilutes the signal.)

---

## 5. Confidence & caveats

- What I'm sure about: <list>
- What I'm uncertain about: <list + what would resolve the uncertainty>
- Anything the profile couldn't answer that the user should know: <list>
- Noise floor: clocks unpinned on this cluster — cross-run deltas under ~5 % are noise; ncu durations are not latency claims (`benchmark-kernel` owns those)

---

## 6. Reproduction

    cd /data/user/jzou521/codes/cuda/flash-vla
    sbatch -w ACD1-20 artifacts/profile/<run_name>/job.sh          # capture
    .venv/bin/python .claude/skills/ncu-report/scripts/analyze_reports.py --run-dir artifacts/profile/<run_name> \
        --report artifacts/profile/<run_name>/reports/full_<tag>.ncu-rep --tag <tag>   # parse
```

---

## Style rules

- **Cite specific metric values for every claim.** "SM throughput X.X%" (with the actual number from your report) > "SM throughput is low".
- **Name files and line numbers.** "Line L of `harness.cu`" (pasting the actual file/line) > a high-level description like "the main memory load".
- **Use NCU's own estimates — except where the playbook overrules them.** `Est. Speedup: X%` ranks magnitude well for kernels NCU understands; on a persistent warp-specialized kernel the occupancy rules estimate a design property, and citing them is a mistake.
- **Cite the ceiling.** Every "X % of peak" gets the `hardware-unit-test` tag it is judged against.
- **Rank by magnitude.** Fix the 50% problem before the 5% problem.
- **Keep the top-line summary dense.** A reader should be able to get the #1 finding in 10 seconds of reading.
- **Link to artifacts.** Don't paste huge tables into the prose — link to `analysis/compare_<tag1>_vs_<tag2>.txt` etc.

## Anti-patterns

- ❌ Generic advice without evidence ("you might consider using shared memory").
- ❌ More than 5 "priorities" — you're probably padding.
- ❌ Re-running the same profile with different tags and copy-pasting the same analysis — consolidate.
- ❌ Reporting from the CLI table directly. Extract, interpret, write — don't dump.
- ❌ Omitting the setup section. Without it, nobody can reproduce or trust the numbers.
