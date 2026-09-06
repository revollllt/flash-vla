#!/usr/bin/env python3
"""Aggregate per-PC warp-stall samples into per-source-line hotspots.

Requires a report with per-PC samples.  On this host's ncu 2025.4.1 a
`--set full` report already has them (no separate `--set source` pass); source
lines need a `-lineinfo` build, otherwise sites are PC addresses and `--sass`
prints the instruction at each hot PC instead.

Produces in `<run-dir>/analysis/`:
    stall_hotspots_<tag>.txt  top sites ranked by total samples with a per-reason
                              breakdown, then the top sites per stall reason.

Usage:
    python3 extract_stall_hotspots.py --run-dir artifacts/profile/myrun \\
            --report artifacts/profile/myrun/reports/full_<tag>.ncu-rep --tag <tag> [--action N] [--sass]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    PCSAMP_PREFIX, describe, load_action, pc_to_source_line, pcsamp_metrics, per_pc_values, safe, sass_at,
)


def short_reason(metric: str) -> str:
    return metric[len(PCSAMP_PREFIX):] if metric.startswith(PCSAMP_PREFIX) else metric


def collect_per_pc(action, include_not_issued: bool):
    """dict[pc] -> dict[reason -> samples]"""
    per_pc = defaultdict(lambda: defaultdict(int))
    for name in pcsamp_metrics(action, include_not_issued=include_not_issued):
        for pc, v in per_pc_values(action, name):
            if pc is not None and v:
                per_pc[pc][short_reason(name)] += int(v)
    return per_pc


def aggregate_by_site(action, per_pc, by_pc: bool):
    per_site = defaultdict(lambda: defaultdict(int))
    site_pcs = defaultdict(set)
    for pc, stalls in per_pc.items():
        f, ln = pc_to_source_line(action, pc)
        site = f"pc {pc:#x}" if (by_pc or f == "?") else f"{Path(f).name}:{ln}"
        site_pcs[site].add(pc)
        for reason, v in stalls.items():
            per_site[site][reason] += v
    return per_site, site_pcs


def write_report(header, per_site, site_pcs, out_path, tag, top_n, with_sass, total_samples, action):
    ranked = sorted(per_site.items(), key=lambda kv: -sum(kv[1].values()))
    reasons = sorted({r for s in per_site.values() for r in s})
    with open(out_path, "w") as f:
        f.write(f"===== Stall hotspots for {tag} =====\n{header}\n")
        f.write(f"total samples (smsp__pcsamp_sample_count): {total_samples}\n")
        f.write(f"distinct sites: {len(ranked)}   (site = file:line with -lineinfo, else PC)\n\n")
        f.write(f"{'Rank':>4} {'Samples':>8} {'%':>6}  {'Site':<48} Breakdown (reason: samples)\n")
        f.write("-" * 150 + "\n")
        for i, (site, stalls) in enumerate(ranked[:top_n]):
            total = sum(stalls.values())
            pct = 100.0 * total / total_samples if total_samples else 0.0
            breakdown = ", ".join(f"{r}: {v}" for r, v in sorted(stalls.items(), key=lambda x: -x[1]) if v)
            f.write(f"{i:>4} {total:>8} {pct:>5.1f}%  {site:<48} {breakdown}\n")
            if with_sass:
                for pc in sorted(site_pcs[site])[:4]:
                    sass = sass_at(action, pc).strip()
                    if sass:
                        f.write(f"{'':>22}{pc:#x}  {sass}\n")
        f.write("\n\n===== Per-reason top sites =====\n")
        for reason in reasons:
            items = [(site, s.get(reason, 0)) for site, s in per_site.items()]
            items = sorted((it for it in items if it[1] > 0), key=lambda x: -x[1])
            if not items:
                continue
            f.write(f"\n--- {reason} ---\n")
            for site, v in items[:10]:
                f.write(f"  {v:>8}  {site}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="report(s) with per-PC samples (repeatable)")
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--action", type=int, action="append", default=None,
                    help="launch index inside each report (repeatable; default 0)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--include-not-issued", action="store_true",
                    help="also count the *_not_issued twins (default: issued-state samples only)")
    ap.add_argument("--by-pc", action="store_true", help="never merge PCs into source lines")
    ap.add_argument("--sass", action="store_true", help="print the SASS at the hot PCs")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    actions = args.action or [0] * len(args.report)
    if len(actions) == 1 and len(args.report) > 1:
        actions = actions * len(args.report)

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for rep, tag, act in zip(args.report, args.tag, actions):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        report, action = load_action(rep, act)
        per_pc = collect_per_pc(action, args.include_not_issued)
        if not per_pc:
            print(f"[{tag}] no per-PC samples in this report (collect with --set full; "
                  f"-lineinfo for source lines)", file=sys.stderr)
            continue
        per_site, site_pcs = aggregate_by_site(action, per_pc, args.by_pc)
        total = safe(action, "smsp__pcsamp_sample_count", 0) or sum(sum(s.values()) for s in per_site.values())
        out = analysis_dir / f"stall_hotspots_{tag}.txt"
        write_report(describe(report, action), per_site, site_pcs, out, tag, args.top, args.sass, total, action)
        resolved = sum(1 for s in per_site if not s.startswith("pc "))
        print(f"[{tag}] -> {out} ({len(per_site)} sites, {resolved} resolved to source lines)")


if __name__ == "__main__":
    main()
