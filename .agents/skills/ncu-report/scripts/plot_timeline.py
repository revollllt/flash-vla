#!/usr/bin/env python3
"""ASCII-plot the sampled timelines of a .ncu-rep (offline, no GPU).

Aggregate metrics average over the whole kernel; only the time series show the
shape -- tail effects, pipeline bubbles, sawtooth, ramp.  Nsight Compute stores
them as instanced metrics keyed by GPU timestamp:

    warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>    warp states (PmSampling_WarpStates)
    <UNIT>.TriageCompute.<metric>                              PM counters (PmSampling) -- ncu 2025.x
    pmsampling:<metric>                                        PM counters on other ncu versions

On this host's ncu 2025.4.1 all of these come with `--set full`.  The sampling
window is longer than the kernel; each series is trimmed to the pass group's
workload window (profiler__timestamp_workload_{start,end}_<g>) before plotting.

Produces `<run-dir>/analysis/pm_timeline_plots.txt`.

Usage:
    python3 plot_timeline.py --run-dir artifacts/profile/myrun \\
            --report artifacts/profile/myrun/reports/full_<tag>.ncu-rep --tag <tag> [--action N]
    python3 plot_timeline.py --list --report <rep>        # which series does this report hold?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    describe, has_metric, load_action, timeline_metrics, timeline_series, workload_windows,
)

# Each entry: (label, [candidate names, first present wins]).  Candidates cover
# ncu 2025.x (Triage / warpsampling) and the `pmsampling:` spelling of other versions.
DEFAULT_SERIES = [
    ("SM instructions executed (% peak)", [
        "TPC.TriageCompute.sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
        "pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "pmsampling:sm__inst_executed.avg.pct_of_peak_sustained_elapsed"]),
    ("Tensor pipe active (% peak)", [
        "TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
        "pmsampling:sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]),
    ("SM cycles active (per elapsed cycle)", [
        "TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed",
        "pmsampling:sm__warps_active.avg.pct_of_peak_sustained_active"]),
    ("DRAM read throughput (% peak)", [
        "FBSP.TriageCompute.dramc__read_throughput.avg.pct_of_peak_sustained_elapsed",
        "pmsampling:dram__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("L2 throughput (% peak)", [
        "LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "pmsampling:lts__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("L1TEX throughput (% peak)", [
        "SM_A.TriageCompute.l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "pmsampling:l1tex__throughput.avg.pct_of_peak_sustained_active"]),
    ("CTAs launched (per cycle)", [
        "FE_B.TriageCompute.gr__ctas_launched_realtime.avg.per_cycle_elapsed"]),
    ("stall: long_scoreboard (warps)", [
        "warpsampling:smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg"]),
    ("stall: barrier (warps)", [
        "warpsampling:smsp__pcsamp_warps_issue_stalled_barrier",
        "pmsampling:smsp__warps_issue_stalled_barrier.avg"]),
    ("stall: short_scoreboard (warps)", [
        "warpsampling:smsp__pcsamp_warps_issue_stalled_short_scoreboard",
        "pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg"]),
    ("stall: wait (warps)", [
        "warpsampling:smsp__pcsamp_warps_issue_stalled_wait",
        "pmsampling:smsp__warps_issue_stalled_wait.avg"]),
    ("stall: warpgroup_arrive (warps, sm90)", [
        "warpsampling:smsp__pcsamp_warps_issue_stalled_warpgroup_arrive"]),
    ("stall: math_pipe_throttle (warps)", [
        "warpsampling:smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
        "pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg"]),
    ("stall: mio_throttle (warps)", [
        "warpsampling:smsp__pcsamp_warps_issue_stalled_mio_throttle",
        "pmsampling:smsp__warps_issue_stalled_mio_throttle.avg"]),
    ("selected (issuing warps)", [
        "warpsampling:smsp__pcsamp_warps_issue_stalled_selected"]),
]


def ascii_plot(ts, vals, label, max_rows=12, max_cols=80):
    """Render one series as ASCII rows (top = max), time left to right."""
    if not vals:
        return [f"\n{label}: no data"]
    n = len(vals)
    ncols = min(max_cols, n)
    bucket = max(1, n // ncols)
    buckets = []
    for c in range(ncols):
        chunk = vals[c * bucket:min(n, (c + 1) * bucket)]
        buckets.append(sum(chunk) / len(chunk) if chunk else 0.0)
    mx = max(buckets) if buckets else 0.0
    span_us = (ts[-1] - ts[0]) / 1e3 if len(ts) > 1 else 0.0
    lines = [f"\n{label}",
             f"  (n={n} samples over {span_us:.1f} us, {(span_us / max(n - 1, 1)):.2f} us/sample, max={mx:.3g})"]
    if mx <= 0:
        lines.append("  all zero inside the workload window")
        return lines
    for r in range(max_rows, 0, -1):
        threshold = mx * r / max_rows
        lines.append(f"  {threshold:8.3g} | " + "".join("#" if b >= threshold else " " for b in buckets))
    lines.append("  " + " " * 10 + "-" * len(buckets))
    lines.append("  " + " " * 10 + " (time ->)")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", default=None)
    ap.add_argument("--action", type=int, action="append", default=None)
    ap.add_argument("--metric", type=str, action="append", default=None,
                    help="plot these series instead of the defaults (repeatable)")
    ap.add_argument("--no-trim", action="store_true", help="plot the whole sampling window")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=80)
    ap.add_argument("--list", action="store_true", help="list the timeline series in the report and exit")
    args = ap.parse_args()

    tags = args.tag or [p.stem for p in args.report]
    if len(tags) != len(args.report):
        ap.error("--report and --tag counts must match")
    actions = args.action or [0] * len(args.report)
    if len(actions) == 1 and len(args.report) > 1:
        actions = actions * len(args.report)

    out_lines = []
    for rep, tag, act in zip(args.report, tags, actions):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        report, action = load_action(rep, act)
        if args.list:
            print(f"{rep} action={act}: {describe(report, action)}")
            for name in timeline_metrics(action):
                ts, vals = timeline_series(action, name, trim=False)
                print(f"  {len(ts) if ts else 0:5d} samples  {name}")
            wins = workload_windows(action)
            print(f"  workload windows (ns): {wins}")
            continue
        out_lines.append(f"\n{'=' * 70}\n{tag}: {rep.name} action={act}\n{describe(report, action)}")
        wins = workload_windows(action)
        if wins:
            out_lines.append("workload windows: " + ", ".join(f"{(e - s) / 1e3:.1f} us" for s, e in wins)
                             + ("  (series trimmed to their pass group's window)" if not args.no_trim else ""))
        selections = ([(m, [m]) for m in args.metric] if args.metric else DEFAULT_SERIES)
        for label, candidates in selections:
            name = next((c for c in candidates if has_metric(action, c)), None)
            if name is None:
                out_lines.append(f"\n{label}: not in this report ({candidates[0]})")
                continue
            ts, vals = timeline_series(action, name, trim=not args.no_trim)
            if ts is None:
                out_lines.append(f"\n{label}: no instances ({name})")
                continue
            out_lines.extend(ascii_plot(ts, vals, f"{label}   [{name}]", args.rows, args.cols))

    if args.list:
        return
    text = "\n".join(out_lines) + "\n"
    if args.run_dir:
        analysis_dir = args.run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        out_path = analysis_dir / "pm_timeline_plots.txt"
        out_path.write_text(text)
        print(f"-> {out_path}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
