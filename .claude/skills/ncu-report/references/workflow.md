# Capture → Diagnose → Report, end to end

## 0. Frame the question before any tool

Write one sentence: *which kernel, at which shape, answering what.* "Why is
this kernel slow" is not answerable; "at the production shape, is this
warp-specialized attention kernel starved by its producer or by wgmma" is. If the
kernel dispatches differently by shape or mode, each active path is its own
capture — averaging two paths hides both.

The production shapes are the only shapes worth profiling here (the host
repo is fixed-workload by design); the host's parity scripts already run
them, which is why its capture wrappers profile *through* a parity script
rather than a synthetic driver.

## 1. Capture

Owned by `gpu-profiler-analysis`, through the host's capture wrappers
(named in SKILL.md's Host integration — swap on port). What matters when
adapting one:

- **Node**: counters are permission-restricted on most of the partition;
  submit with `-w` to one of the ncu-capable nodes listed in the wrapper
  header. A wrong node fails with `ERR_NVGPUCTRPERM`.
- **Filter tight**: `--kernel-name-base=function --kernel-name="regex:<name>"
  --launch-count=N` — every extra matched launch multiplies replay time.
- **`--clock-control=none`**: the host cluster denies clock locking; letting
  ncu try aborts the capture. Consequence: cross-run deltas below ~5% are
  noise (same rule as the benchmark skill).
- **Source lines need `-lineinfo`** at compile time, passed through the
  host's extra-nvcc-flags hook (the build dir is hash-keyed on flags, so the
  instrumented .so caches separately). Without it, hotspots still work but
  attribute to bare PCs/SASS.
- Export the human pages next to the report while still on the node, as
  the wrappers do:
  `ncu -i out.ncu-rep --page details > out.details.txt` and
  `--page raw --csv > out.raw.csv`.
- For a quick did-the-fix-land recheck, collect only the few metrics in
  question with `ncu --metrics a,b,c` (1–2 replay passes instead of ~45).
  That answers "did sectors/request drop", not "what is the bottleneck".

Reports land in the host's ignored report homes, named with the job id so
provenance survives.

## 2. Extract

`scripts/report_query.py` on the login node. `summary` per profiled launch
(`--action N` — a multi-mode capture holds several), `rules` for the engine's
ranked suggestions, `hotspots` for stall attribution, `compare` for A/B.
For anything bespoke, `references/python-api.md`.

## 3. Diagnose

Read `rules` output first: the engine reliably names what it *sees*. Then
walk the six lenses of `references/dimensions-sm90.md` in order — launch and
occupancy, balance, stalls, tensor pipe, timeline, memory pattern — writing
down for each the metric value observed, not just the verdict. Then map the
signal set to `references/playbook-sm90.md`, which carries the cases where
the rule engine's advice must be overruled for warp-specialized persistent
kernels.

Two ranking rules: fix the biggest signal first (a tail or an idle SM dwarfs
a coalescing nit), and never sum `Est. Speedup` values — rules overlap.

## 4. Report

A finding is four parts: **metric = value → meaning → move**, e.g.
"`stall barrier = <value>` warps/issue, concentrated at one ring-wait PC →
consumers wait on the producer → deepen the ring or widen the producer,
ceiling per the TMA unit". 3–5 findings ranked by expected impact; NCU's per-rule
estimate ranks magnitude, your judgement ranks effort. State the capture
conditions (node, job id, ncu version, workload script and mode, whether
`-lineinfo` was on) once at the top — a number whose capture cannot be
reproduced is not evidence, and profiler numbers are never latency claims
(`rules/performance-and-validation.md`).

## Anti-patterns

- A verdict with no metric value behind it.
- Profiling a shape the pipeline never runs.
- Pasting raw CLI pages into the report instead of extracting and reading.
- Chasing a 1% rule while an 87% rule sits unexplained — or obeying an
  occupancy rule on a kernel whose design pins occupancy (playbook, entry 1).
- Comparing numbers captured under different ncu versions or nodes as if
  they were one series.
