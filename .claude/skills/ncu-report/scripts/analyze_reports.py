#!/usr/bin/env python3
"""Extract and compare key metrics from .ncu-rep files (offline, no GPU).

Produces in `<run_dir>/analysis/`:
    metrics_all_<tag>.json       every metric, archival (re-analysis needs no .ncu-rep)
    metrics_key_<tag>.{txt,json} curated key metrics for the report's GPU generation
                                 (sm90: + wgmma / TMA / cluster / spill / bank-conflict families)
    rules_<tag>.txt              NCU rule engine, sorted by Est. Speedup
    compare_<tag1>_vs_<tag2>.txt side-by-side when >= 2 reports are given

Usage:
    # one report
    python3 analyze_reports.py --run-dir artifacts/profile/myrun \\
            --report artifacts/profile/myrun/reports/full_<tag>.ncu-rep --tag <tag>

    # two reports -> side-by-side (repeat --report/--tag; --action per report, default 0)
    python3 analyze_reports.py --run-dir artifacts/profile/myrun \\
            --report .../full_v1.ncu-rep --tag v1 --report .../full_v2.ncu-rep --tag v2

    # a multi-mode capture: pick the launch
    python3 analyze_reports.py --run-dir ... --report profiles/attn/ncu_555126.ncu-rep --tag attn --action 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    describe, dump_all_metrics, key_metrics_for, load_action, rule_speedups, safe,
)


def collect(report_path: Path, tag: str, action_idx: int, analysis_dir: Path) -> tuple[dict, list[str]]:
    report, action = load_action(report_path, action_idx)
    header = describe(report, action)
    print(f"[{tag}] {report_path.name} action={action_idx}: {header}")

    n = dump_all_metrics(action, analysis_dir / f"metrics_all_{tag}.json")
    print(f"  -> metrics_all_{tag}.json ({n} metrics)")

    metrics = key_metrics_for(action)
    key = {m: safe(action, m) for m in metrics}
    key["__kernel_name__"] = action.name()
    key["__header__"] = header
    key["__action__"] = action_idx
    (analysis_dir / f"metrics_key_{tag}.json").write_text(json.dumps(key, indent=2, default=str))
    with open(analysis_dir / f"metrics_key_{tag}.txt", "w") as f:
        f.write(f"===== {tag} =====\n{header}\naction: {action_idx}\n\n")
        for m in metrics:
            f.write(f"{m:95s} = {key[m]}\n")
    print(f"  -> metrics_key_{tag}.{{json,txt}} ({len(metrics)} metrics)")

    with open(analysis_dir / f"rules_{tag}.txt", "w") as f:
        f.write(f"===== NCU rules for {tag} (sorted by Est. Speedup; never sum them) =====\n")
        for est, ident, title, message, section in rule_speedups(action):
            head = f"est +{est:.1f}%" if est is not None else "info    "
            f.write(f"[{head}] {ident} ({section}): {title}\n    {message}\n")
    print(f"  -> rules_{tag}.txt")
    return key, metrics


def compare(collected: dict, metric_union: list[str], analysis_dir: Path):
    tags = list(collected.keys())
    if len(tags) < 2:
        return
    out_path = analysis_dir / f"compare_{'_vs_'.join(tags)}.txt"
    col_w = max(20, max(len(t) for t in tags) + 2)
    with open(out_path, "w") as f:
        f.write(f"{'Metric':<95}" + "".join(f"{t:>{col_w}}" for t in tags) + f"{'delta(last/first)':>20}\n")
        f.write("-" * (95 + col_w * len(tags) + 20) + "\n")
        for m in metric_union:
            vals = [collected[t].get(m) for t in tags]
            f.write(f"{m:<95}")
            for v in vals:
                s = f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
                f.write(f"{s:>{col_w}}")
            a, b = vals[0], vals[-1]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
                f.write(f"{100.0 * (b - a) / a:>+19.1f}%")
            f.write("\n")
    print(f"compare -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Extract key NCU metrics and compare.")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="profile run directory; outputs go to <run-dir>/analysis/")
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="a .ncu-rep file (repeatable)")
    ap.add_argument("--tag", type=str, action="append", required=True,
                    help="short label per --report")
    ap.add_argument("--action", type=int, action="append", default=None,
                    help="launch index inside each report (repeatable; default 0 for all)")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    actions = args.action or [0] * len(args.report)
    if len(actions) == 1 and len(args.report) > 1:
        actions = actions * len(args.report)
    if len(actions) != len(args.report):
        ap.error("--action must be given once, or once per --report")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected, union = {}, []
    for rep, tag, act in zip(args.report, args.tag, actions):
        if not rep.exists():
            print(f"[skip] {rep} does not exist", file=sys.stderr)
            continue
        key, metrics = collect(rep, tag, act, analysis_dir)
        collected[tag] = key
        for m in metrics:
            if m not in union:
                union.append(m)

    compare(collected, union, analysis_dir)


if __name__ == "__main__":
    main()
