# Agent Note: the ncu-report skill adopts the upstream profiling template with an sm90 overlay

Status: implemented

Related: the [kernel-design workflow note](2026-09-01-kernel-design-workflow.md)
places this skill in the candidate loop; that placement is unchanged.

## Problem

The `ncu-report` skill shipped as five short pages and one query script: enough
to read a report that already existed, too thin to carry an agent through a
whole profiling pass (which process to profile, how to get `-lineinfo` in, which
ncu flags this cluster tolerates, where a run's artifacts live, how to write the
report). Its host knowledge was also stale — it named capture wrappers that no
longer exist — and it recorded the timeline lens as a gap because the
`pmsampling:` names it looked for do not exist on this toolchain.

## Decision

Rebuild the skill on the structure and depth of
[mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill)
(MIT; `LICENSE` and provenance kept in the skill's `README.md`), and layer this
repo's sm90 / H100 knowledge on top rather than beside it:

- **Layout**: the upstream ten numbered references, helper scripts and companion
  document, in this repo's `references/` + `scripts/` naming. The skill's name
  and the `report_query.py` interface are unchanged, so `kernel-design`,
  `gpu-profiler-analysis` and `ARCHITECTURE.md` keep pointing at the same thing.
- **Primary target is H100 / sm90.** Every metric name the skill states is
  verified by enumeration on this repo's own full-set reports (Nsight Compute
  2025.4.1, 2 320 names); `references/08-sm90-metric-names.md` is the vocabulary
  and the diff against the upstream B200 list. The upstream B200 reference and
  Blackwell companion stay as an appendix for a report from that GPU.
- **Hopper facts that change a diagnosis are first-class**: the `gmma`
  (aggregate) / `warpgroup_arrive` (per-PC) stall pair, TMA traffic invisible to
  the LSU counters, `sm__ops_path_tensor_op_hmma_*` reading 0 on wgmma work, the
  occupancy rules to overrule on a persistent warp-specialized kernel. They live
  in the dimensions doc, the playbook (patterns O–V) and the new Hopper
  companion (`hopper-cuda-programming.md`), which maps the sixteen principles to
  sm90 NCU signals and cites machine numbers only by `hardware-unit-test` tag.
- **Timeline lens closed.** On 2025.4.1 `--set full` already carries
  `PmSampling` and `PmSampling_WarpStates`; the series are
  `warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>` and
  `<UNIT>.TriageCompute.<metric>`, keyed by GPU timestamp, bounded by
  `profiler__timestamp_workload_{start,end}_<g>`. `plot_timeline.py` reads them
  and trims to the kernel window.
- **Host binding** is explicit and swappable: ncu-capable node list,
  `--clock-control=none`, `sbatch/_common.sh`, the `<HOOK>_NVCC_DEFINES`
  `-lineinfo` hooks, the engine drivers (`eval.correctness`,
  `benchmarks kernels --site`) as the default profiled process, and
  `artifacts/profile/<run>/` (git-ignored) as the one run-directory home, with
  `profiles/<area>/` and `artifacts/ncu-verify/` read in place as legacy.
- **Helpers are version-tolerant**: `ncu_utils.py` discovers the newest
  `ncu_report` that loads the report, accepts both rule-dict shapes
  (`rule_identifier`/`rule_message`/`speedup_estimation` here,
  `rule_name`/`estimated_speedup_pct` upstream), and picks the curated key-metric
  list by compute capability.

## Alternatives considered

- **Keep the five-page skill and patch it.** Rejected: the missing parts were
  whole phases (harness, collection recipes, run layout, report template,
  common issues), not sentences.
- **Symlink or submodule the upstream and keep sm90 notes separately.**
  Rejected: an agent would read B200 names and 148-SM arithmetic first; the
  sm90 facts have to be in the same sentence as the generic advice they
  qualify.
- **Copy upstream verbatim including its B200-first framing and the
  flashinfer-trace browser.** Rejected: this repo is H100-only and
  fixed-workload; the browser has no dataset to browse. The B200 material is
  kept as an appendix; the browser is referenced, not shipped.

## Consequences

- One skill covers capture-planning through report-writing; `gpu-profiler-analysis`
  still owns the capture runner, `benchmark-kernel` the latency numbers,
  `hardware-unit-test` the ceilings, `kernel-design` the next move.
- Metric names are a per-version contract: on any ncu other than 2025.4.1 the
  vocabulary page says to re-enumerate before trusting a name.
- The upstream's Chinese Blackwell companion is carried unmodified; the Hopper
  companion is English like the rest of the repo.

## Verification

On the login node with the repo venv, against existing reports
(`profiles/ffn/ncu_lineinfo_584074.ncu-rep`, `profiles/attn/ncu_555126.ncu-rep`,
`artifacts/ncu-verify/580593_full.ncu-rep`):

- `report_query.py summary | rules | hotspots` unchanged and passing; hotspots
  resolve to kernel and CUTLASS source lines on the `-lineinfo` report.
- `analyze_reports.py` writes `metrics_all_*`, `metrics_key_*`, `rules_*` and
  a two-tag compare; every curated sm90 metric returns a value.
- `extract_stall_hotspots.py` on the `-lineinfo` report resolves sites to
  source lines; with `--sass` on a plain report it prints the instruction at
  each hot PC.
- `plot_timeline.py --list` enumerates the `warpsampling:` and `Triage` series;
  the plot is trimmed to the recorded workload window.
- Metric enumeration confirmed every name in the upstream key list on sm90 and
  the absences listed in `08-sm90-metric-names.md` (tensor `subpipe` names,
  `pmsampling:*`, L2 atomic sector counters, `dram__throughput`).
- The `ncu_report` module of 2026.2.1 reads a 2025.4.1 report (fallback path).
