# Profile Directory Layout & Naming

**Read this first, before any collection.** Bad directory layout is the single most common cause of mixing results from different runs, overwriting prior profiles, or losing track of which `.ncu-rep` belongs to which kernel version. The rules below are non-negotiable for work in this repo.

---

## Top-level rule

**Every new profiling run lives in its own directory under `artifacts/profile/`.** `artifacts/` is git-ignored, which is right: reports are evidence, and evidence that matters gets promoted into an Agent Note (with the job id and the exact command) rather than committed as binaries. Never scatter `.ncu-rep` files under `src/`, `lab/`, `eval/` or the skill directories.

```
<repo_root>/
├── artifacts/
│   ├── profile/                    ← every new run: one subdirectory each
│   │   ├── <run_1>/
│   │   └── <run_2>/
│   └── ncu-verify/                 ← legacy: the skill's own verification capture
├── profiles/<area>/                ← legacy: reports the older sbatch wrappers wrote
│   ├── ncu_<jobid>.ncu-rep         (git-ignored; read in place, do not move)
│   ├── ncu_<jobid>.details.txt
│   └── ncu_<jobid>.raw.csv
├── src/                            ← untouched by profiling
└── ...
```

`gpu-profiler-analysis`'s runner (`run_local_profile.py --backend ncu --output-dir artifacts/profile/<run>`) already writes `profile.ncu-rep` + `manifest.json` into exactly this convention; a hand-written sbatch job should do the same.

---

## One run = one subdirectory

Every time you profile a kernel — a new kernel, a new version of the same kernel, or the same kernel on a different plan or call site — **create a new subdirectory**. Never write into an existing run's directory.

Rationale:

- Profiles of different implementations must not overwrite each other. `v1` today and `v2` tomorrow both need to exist for A/B comparison.
- The capture command is part of the profile: it encodes which plan, which site, which `-lineinfo` hook, which node. Keeping it in the run dir (`job.sh` or `manifest.json`) pins the provenance.
- Analysis artifacts (`metrics_*.json`, `compare_*.txt`, ASCII plots) are tied to one set of `.ncu-rep` files; they must not be mixed.

---

## Run directory naming

Descriptive, short, kebab-case. Include **what** was profiled and **why**, and the Slurm job id when there is one — with unpinned clocks and two driver generations on the partition, the job id is how you recover the node later.

Good:
```
artifacts/profile/ffn-gu-baseline-584074/
artifacts/profile/ffn-gu-ring4-585120/
artifacts/profile/attn-taskloop-lineinfo-580593/
artifacts/profile/attn-v2-vs-v1/                 # comparison run, no reports of its own
```

Bad:
```
artifacts/profile/test/                # too vague
artifacts/profile/run1/                # meaningless
artifacts/profile/20260906/            # a date with no context
artifacts/profile/final/               # there is never a "final"
```

---

## Standard run layout

```
artifacts/profile/<run_name>/
├── REPORT.md                       ← human-readable final report
├── job.sh                          ← the sbatch script that captured it (or manifest.json from the runner)
├── harness/                        ← only when a standalone harness was built
│   ├── <kernel>_harness.cu         ← the exact source that was compiled
│   └── build_command.sh
├── reports/
│   ├── full_<tag>.ncu-rep          ← ncu --set full output (per-PC samples included on 2025.4.1)
│   ├── full_<tag>.details.txt      ← ncu -i ... --page details, exported on the node
│   └── full_<tag>.raw.csv          ← ncu -i ... --page raw --csv, exported on the node
└── analysis/
    ├── metrics_all_<tag>.json      ← every metric, archive
    ├── metrics_key_<tag>.{txt,json}← curated key metrics
    ├── compare_<a>_vs_<b>.txt      ← side-by-side
    ├── stall_hotspots_<tag>.txt    ← per-line stall aggregation
    ├── pm_timeline_plots.txt       ← ASCII time-series
    └── notes.md                    ← optional working notes
```

Notes:

- `<tag>` is the per-plan / per-site / per-mode label, e.g. `gu`, `attn-alias`, `dr-ring4`. Short, and named for the workload, not the job.
- A parity script that runs several modes produces **one report with several actions** (one per profiled launch). Note the `--action` index → mode mapping in `REPORT.md`; the helpers take `--action`.
- Export `details.txt` and `raw.csv` **on the compute node while the report is fresh** (the wrappers do): the CLI's `--page` export needs the same ncu that wrote the report, and the login node runs a different toolchain.

---

## Comparing two runs

For A/B comparisons (before vs after, two plans on one site), create a comparison run that *references* both underlying runs:

```
artifacts/profile/<kernel>-v2-vs-v1/
├── REPORT.md                       ← describes both runs + the comparison
└── analysis/
    ├── compare_key_metrics.txt     ← report_query.py compare / analyze_reports.py output
    └── compare_stalls.txt
    (No ncu-rep files — they live in the referenced runs)
```

The comparison run does not re-profile; it produces comparison artifacts and prose. Remember the noise floor: clocks are unpinned on this cluster, so treat cross-run deltas below ~5 % as noise and compare *structure* (stall mix, sectors, instruction counts) rather than durations.

---

## What does NOT go in a run directory

- `.ncu-rep.old` backups — a prior version is a separate run.
- Scratch files — use the session scratchpad.
- The model weights / workload tensors — reference them by path.
- Compiler intermediates (`*.o`, `*.d`) and the TileLang / torch-extension caches (`.cache/`).
- `ncu_home/` or ncu cache directories — set `HOME` to something writable before running ncu rather than letting it cache inside the run dir.

---

## Environment variable convention

```bash
export PROFILE_RUN_DIR=artifacts/profile/<run_name>
mkdir -p "$PROFILE_RUN_DIR"/{reports,analysis}

# on the login node, after the job finished:
HELPERS=.claude/skills/ncu-report/scripts
.venv/bin/python $HELPERS/analyze_reports.py --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag>
```

All helpers accept `--run-dir` and write under `<run-dir>/analysis/`.

---

## Checklist before starting a profile run

1. `mkdir -p artifacts/profile/<new_run_name>/{reports,analysis}` (and `harness/` if you build one).
2. Write the sbatch job into the run dir (`job.sh`) — node list, `--clock-control=none`, the `-lineinfo` hook, `--export` pointing at `reports/full_<tag>`.
3. Submit; check `sbatch/logs/<name>_<jobid>.out` for `==PROF== Report:` and the pass count.
4. Parse on the login node into `analysis/`.
5. Write `REPORT.md`; promote decision-changing findings into an Agent Note.
6. For the next capture, go back to step 1 with a new name — never write into the existing one.
