"""The promotion gate: one candidate against the acceptance registry, one evidence record.

    python -m eval.promotion_gate --target h100/pi05
    python -m eval.promotion_gate --target h100/pi05 --candidate lab/plans/pi05-attn-cuda.json

A script, not a service. It reads the Target's entry of the acceptance
registry, runs the in-engine correctness checks and the same-process A/B/A
latency run through the generic harnesses, computes the floor model for the
candidate, applies the gate semantics of
`docs/architecture/30-acceptance-criteria.md`, and writes one evidence record.
Harnesses only report; this is the one consumer that produces a verdict.

Verdicts:
  pass     every gate passed and the candidate rule holds
  fail     a gate failed, or the candidate regresses beyond the calibrated noise
  blocked  a gate could not run (a baseline adapter that is not installed, a
           missing budget), so nothing is claimed either way

Baseline-tier scripts (official implementation adapters) are run as
subprocesses when `--baseline` is given and the Target declares the
capability; without it they are recorded as not run, which blocks the
verdict rather than passing it. Latency never overrides a correctness gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks import floor as floor_model
from benchmarks import latency
from benchmarks.targets import resolve
from eval import acceptance
from eval.correctness import in_engine

REPO = Path(__file__).resolve().parent.parent


def _registry_version() -> str:
    return hashlib.sha1((REPO / "eval" / "acceptance.py").read_bytes()).hexdigest()[:12]


def _depth(config: dict[str, Any] | None) -> dict[str, int | None]:
    """Registry depth config to `in_engine.run` arguments; 'full' is the Target's default."""
    out: dict[str, int | None] = {}
    for key in ("steps", "layers"):
        value = (config or {}).get(key, 1)
        out[key] = None if value == "full" else int(value)
    return out


def _run_in_engine_checks(target: str, candidate: str,
                          checks: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """The in-engine tier: one lockstep comparison per registry check."""
    results = []
    for check in checks:
        if check.get("oracle") not in (None, "in_engine_reference"):
            continue
        name = check["check"]
        if name in ("replay_determinism", "finiteness"):
            # Folded into every in-engine comparison; read from the shallow one.
            continue
        depth = _depth(check.get("config"))
        report = in_engine.run(target, candidate, seed=seed, **depth)
        threshold = report["threshold"] if check.get("threshold") else None
        passed = report["replay_identical"] and report["finite"] and (
            threshold is None or check["mode"] != "gate" or report["min_cosine"] > threshold)
        results.append({"check": name, "mode": check["mode"],
                        "config": {k: (v if v is not None else "full") for k, v in depth.items()},
                        "min_cosine": report["min_cosine"], "threshold": threshold,
                        "replay_identical": report["replay_identical"],
                        "finite": report["finite"], "status": "passed" if passed else "failed",
                        "report": report})
    # The two folded gates, read from the first (shallowest) comparison.
    if results:
        first = results[0]
        results.insert(0, {"check": "finiteness", "mode": "gate",
                           "status": "passed" if first["finite"] else "failed"})
        results.insert(0, {"check": "replay_determinism", "mode": "gate",
                           "status": "passed" if first["replay_identical"] else "failed"})
    return results


def _run_baseline_checks(scripts: tuple[str, ...], checks: list[dict[str, Any]],
                         run: bool) -> list[dict[str, Any]]:
    """The official-baseline tier: the Target's scripts as subprocesses, or not run.

    Each script runs once; a baseline check passes when every script passed,
    is unavailable when any script could not import its adapter, and fails
    otherwise. The scripts own their own gate/report split internally.
    """
    baseline_checks = [c for c in checks if c.get("oracle") == "official_baseline"]
    if not baseline_checks:
        return []
    if not run:
        return [{"check": c["check"], "mode": c["mode"], "status": "not_run",
                 "reason": "--baseline not given"} for c in baseline_checks]
    runs = []
    for script in scripts:
        proc = subprocess.run([sys.executable, "-m", script], capture_output=True, text=True)
        status = "passed" if proc.returncode == 0 else "failed"
        if "ModuleNotFoundError" in proc.stderr:
            status = "unavailable"
        runs.append({"script": script, "status": status, "returncode": proc.returncode,
                     "stderr_tail": proc.stderr[-2000:]})
    statuses = {r["status"] for r in runs}
    overall = ("unavailable" if "unavailable" in statuses
               else "failed" if "failed" in statuses else "passed")
    return [{"check": c["check"], "mode": c["mode"], "status": overall, "scripts": runs}
            for c in baseline_checks]


def _latency_verdict(report: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """Apply the candidate rule to an A/B/A report: improve on one statistic, regress on none."""
    deltas = report["deltas"]
    candidate_leg = next(leg for leg in deltas["legs"] if not leg["same_as_reference"])
    mde = deltas["minimum_detectable_effect"] or {}
    metric, stat = rule["improve"]
    key = f"{metric}.{stat}"
    improve = candidate_leg["deltas"].get(key, {})
    improved = improve.get("delta", 0.0) < 0 and improve.get("distinguishable", False)
    regressions = []
    for metric_r, stat_r in rule["no_regression"]:
        k = f"{metric_r}.{stat_r}"
        entry = candidate_leg["deltas"].get(k)
        if entry is None:
            continue
        if entry["delta"] > 0 and entry.get("distinguishable", False):
            regressions.append({k: entry["delta"], "mde": mde.get(k)})
    return {"improve": {key: improve, "mde": mde.get(key)}, "improved": improved,
            "regressions": regressions, "passed": improved and not regressions}


def run(target: str, candidate: str = "shipped", reference: str = "reference",
        reps: int | None = None, seed: int = 0, baseline: bool = False,
        out_dir: str | None = None) -> dict[str, Any]:
    """Run every check for one candidate plan and write the evidence record."""
    target = resolve(target)
    candidate = candidate or "shipped"
    reference = reference or "reference"
    spec = acceptance.for_target(target)
    record: dict[str, Any] = {
        "target": target, "candidate": {"plan": candidate},
        "reference": {"plan": reference}, "acceptance_version": _registry_version(),
        "budget": spec["budget"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": [], "latency": None, "floor": None, "verdict": None,
    }
    if spec["budget"] is None:
        record["verdict"] = "blocked"
        record["reason"] = f"no acceptance entry with a budget for {target}"
        return _finish(record, out_dir)

    checks = list(spec["correctness"]["checks"])
    record["checks"] += _run_in_engine_checks(target, candidate, checks, seed)
    baseline_scripts = tuple(spec["scripts"].get("official_baseline", ()))
    if "baseline_adapter" in spec["capabilities"]:
        record["checks"] += _run_baseline_checks(baseline_scripts, checks, baseline)

    lat = spec["latency"]
    plans = [reference, candidate, reference]
    latency_report = latency.run(target, plans, reps=reps or lat["reps"], warmup=lat["warmup"],
                                 seed=seed)
    record["latency"] = {"report": latency_report,
                         "rule": _latency_verdict(latency_report, lat["candidate_rule"])}
    try:
        record["floor"] = floor_model.run(target, candidate, seed=seed)
    except Exception as exc:  # the floor is context, never a gate
        record["floor"] = {"error": f"{type(exc).__name__}: {exc}"}

    gates = [c for c in record["checks"] if c["mode"] == "gate"]
    failed = [c["check"] for c in gates if c["status"] == "failed"]
    blocked = [c["check"] for c in gates if c["status"] in ("not_run", "unavailable")]
    if failed:
        record["verdict"], record["reason"] = "fail", f"gates failed: {failed}"
    elif not record["latency"]["rule"]["passed"]:
        record["verdict"] = "fail"
        record["reason"] = ("latency candidate rule not met: "
                            + json.dumps(record["latency"]["rule"], default=str)[:300])
    elif blocked:
        record["verdict"], record["reason"] = "blocked", f"gates not run: {blocked}"
    else:
        record["verdict"], record["reason"] = "pass", "every gate passed; candidate rule holds"
    return _finish(record, out_dir)


def _finish(record: dict[str, Any], out_dir: str | None) -> dict[str, Any]:
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    base = Path(out_dir or REPO / "artifacts" / "gate")
    slug = Path(str(record["candidate"]["plan"])).stem
    path = base / record["target"].replace("/", "_") / f"{slug}-{record['finished']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str))
    record["evidence_path"] = str(path)
    return record


def summary(record: dict[str, Any]) -> str:
    lines = [f"verdict: {record['verdict']}  ({record.get('reason', '')})",
             f"target: {record['target']}  candidate: {record['candidate']}",
             f"acceptance: {record['acceptance_version']}  evidence: {record.get('evidence_path')}"]
    for c in record["checks"]:
        extra = f" min_cos={c['min_cosine']:.7f} thr={c.get('threshold')}" if "min_cosine" in c else ""
        lines.append(f"  [{c['mode']:6s}] {c['check']:22s} {c['status']}{extra}")
    if record.get("latency"):
        rule = record["latency"]["rule"]
        lines.append(f"  latency: improved={rule['improved']} regressions={rule['regressions']} "
                     f"improve={rule['improve']}")
    if record.get("floor") and "totals" in record["floor"]:
        t = record["floor"]["totals"]
        lines.append(f"  floor: measured {t['measured_min_us']:.0f} us, structural "
                     f"{t['structural_us']:.0f} us, roofline {t['roofline_us']:.0f} us, "
                     f"valid={record['floor']['valid']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidate", default="shipped",
                        help="candidate plan: shipped, reference, JSON or a lab/plans/*.json path")
    parser.add_argument("--reference", default="reference",
                        help="reference plan (default: the Target's reference route)")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true", help="also run the baseline-tier scripts")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    record = run(args.target, args.candidate, reference=args.reference, reps=args.reps,
                 seed=args.seed, baseline=args.baseline, out_dir=args.out_dir)
    print(summary(record))
    return {"pass": 0, "fail": 1, "blocked": 2}[record["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
