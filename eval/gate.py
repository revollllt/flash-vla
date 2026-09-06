"""The promotion gate: one candidate plan against the acceptance registry, one evidence record.

    python -m eval.gate --target h100/pi05
    python -m eval.gate --target h100/pi05 --candidate lab/plans/pi05-attn-cuda.json
    python -m eval.gate --target h100/pi05 --mode no-regression --baseline

A script, not a service. It reads the Target's entry of the acceptance
registry (`eval/acceptance.py`), runs the in-engine correctness checks and the
same-process A/B/A latency run through the generic harnesses, applies the
deployment bound and the candidate rule, computes the floor model as context,
and writes one evidence record. Harnesses only report; this is the one
consumer that produces a verdict.

Verdicts, in the order they are decided:

  fail     a correctness gate failed
  blocked  the A/B/A run is invalid (control legs disagree beyond the
           registry's limit): rerun, nothing is claimed
  blocked  the tail bound is violated by the reference leg too: the node,
           not the candidate, is the cause
  fail     the candidate's tail sits above the deployment jitter bound while
           the reference's does not
  fail     the candidate rule of the chosen mode does not hold
  blocked  a baseline-tier gate was not run or its interpreter is missing
  pass     everything above held

`--mode improve` (default) is for a performance candidate: it must improve the
chunk `min` by more than the promotion bar and the control spread, and regress
neither `median` nor `p99` by more than the spread. `--mode no-regression` is
for a refactor or a correctness fix: nothing may regress beyond the spread.

Baseline-tier scripts (official implementation adapters) run under the
interpreter the registry names for the Target when `--baseline` is given;
otherwise they are recorded as not run, which blocks the verdict rather than
passing it. Latency never overrides a correctness gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmarks import floor as floor_model
from benchmarks import latency
from benchmarks.targets import resolve
from eval import acceptance
from eval import correctness as in_engine

REPO = Path(__file__).resolve().parent.parent


def _registry_version() -> str:
    return hashlib.sha1((REPO / "eval" / "acceptance.py").read_bytes()).hexdigest()[:12]


def _depth(config: dict[str, Any] | None) -> dict[str, int | None]:
    """Registry depth config to `correctness.run` arguments; 'full' is the Target's default."""
    out: dict[str, int | None] = {}
    for key in ("steps", "layers"):
        value = (config or {}).get(key, 1)
        out[key] = None if value == "full" else int(value)
    return out


def _run_in_engine_checks(target: str, candidate: str, checks: list[dict[str, Any]],
                          seed: int) -> list[dict[str, Any]]:
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
        threshold_key = check.get("threshold", "shallow_cosine")
        report = in_engine.run(target, candidate, seed=seed, threshold=threshold_key, **depth)
        gated = check["mode"] == "gate" and check.get("threshold") is not None
        passed = report["replay_identical"] and report["finite"] and (
            not gated or report["min_cosine"] > report["threshold"])
        results.append({"check": name, "mode": check["mode"],
                        "config": {k: (v if v is not None else "full") for k, v in depth.items()},
                        "min_cosine": report["min_cosine"],
                        "threshold": report["threshold"] if gated else None,
                        "threshold_key": threshold_key if gated else None,
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
                         run: bool, python: str | None) -> list[dict[str, Any]]:
    """The official-baseline tier: the Target's scripts as subprocesses, or not run.

    Each script runs once under the registry's interpreter; a baseline check
    passes when every script passed, is unavailable when the interpreter is
    missing or a script could not import its adapter, and fails otherwise.
    The scripts own their own gate/report split internally.
    """
    baseline_checks = [c for c in checks if c.get("oracle") == "official_baseline"]
    if not baseline_checks:
        return []
    if not run:
        return [{"check": c["check"], "mode": c["mode"], "status": "not_run",
                 "reason": "--baseline not given"} for c in baseline_checks]
    if not python or not Path(python).exists():
        return [{"check": c["check"], "mode": c["mode"], "status": "unavailable",
                 "reason": f"baseline interpreter not found: {python}"} for c in baseline_checks]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), str(REPO / "src"),
                                                    env.get("PYTHONPATH", "")) if p)
    runs = []
    for script in scripts:
        proc = subprocess.run([python, "-m", script], capture_output=True, text=True,
                              cwd=REPO, env=env)
        status = "passed" if proc.returncode == 0 else "failed"
        if "ModuleNotFoundError" in proc.stderr:
            status = "unavailable"
        runs.append({"script": script, "python": python, "status": status,
                     "returncode": proc.returncode, "stderr_tail": proc.stderr[-2000:]})
    statuses = {r["status"] for r in runs}
    overall = ("unavailable" if "unavailable" in statuses
               else "failed" if "failed" in statuses else "passed")
    return [{"check": c["check"], "mode": c["mode"], "status": overall, "scripts": runs}
            for c in baseline_checks]


def _candidate_leg(report: dict[str, Any]) -> dict[str, Any]:
    """The middle leg of the A/B/A, by position: a candidate may share the reference's plan
    (a no-regression self-test), so plan names cannot identify it."""
    return report["deltas"]["legs"][0]


def _latency_verdict(report: dict[str, Any], lat: dict[str, Any], mode: str) -> dict[str, Any]:
    """Apply the candidate rule of `mode` to an A/B/A report."""
    deltas = report["deltas"]
    spread = deltas.get("control_spread_ms")
    valid = bool(deltas.get("valid"))
    candidate = _candidate_leg(report)["deltas"]
    rule = lat["candidate_rule"]
    bar = lat["promotion_bar_ms"]
    out: dict[str, Any] = {"mode": mode, "run_valid": valid, "control_spread_ms": spread,
                           "control_spread_max_ms": deltas.get("control_spread_max_ms"),
                           "promotion_bar_ms": bar, "regressions": []}
    if spread is None:
        out.update({"passed": False, "reason": "no control leg"})
        return out
    # A statistic regresses when it moves by more than the larger of the bar
    # and its own control spread: the `min` spread is not the median's noise.
    own_spread = deltas.get("minimum_detectable_effect") or {}

    def regressed(k: str) -> dict[str, float] | None:
        entry = candidate.get(k)
        if entry is None:
            return None
        limit = max(bar, own_spread.get(k, spread))
        return {k: entry["delta"], "limit_ms": limit} if entry["delta"] > limit else None

    if mode == "improve":
        metric, stat = rule["improve"]
        key = f"{metric}.{stat}"
        delta = candidate.get(key, {}).get("delta")
        needed = max(bar, spread)
        improved = delta is not None and delta < -needed
        out["improve"] = {key: delta, "needed_ms": needed}
        for metric_r, stat_r in rule["no_regression"]:
            k = f"{metric_r}.{stat_r}"
            if k != key and regressed(k):
                out["regressions"].append(regressed(k))
        out["passed"] = bool(improved and not out["regressions"])
    elif mode == "no_regression":
        for metric_r, stat_r in rule["no_regression"]:
            k = f"{metric_r}.{stat_r}"
            if regressed(k):
                out["regressions"].append(regressed(k))
        out["passed"] = not out["regressions"]
    else:
        raise ValueError(f"unknown mode {mode!r}; registry modes: {rule['modes']}")
    return out


def _deployment_verdict(report: dict[str, Any], dep: dict[str, Any]) -> dict[str, Any]:
    """The tail bound: candidate p99 - min against the jitter budget, read beside the reference."""
    metric = dep["metric"]
    jitter = dep["jitter_ms"]

    def tail(leg: dict[str, Any]) -> float | None:
        stats = leg["metrics"][metric]
        if stats.get("p99") is None:
            return None
        return stats["p99"] - stats["min"]

    legs = report["legs"]
    reference = tail(legs[0])
    candidate = tail(legs[1])
    out = {"metric": metric, "jitter_ms": jitter, "candidate_p99_minus_min": candidate,
           "reference_p99_minus_min": reference}
    if candidate is None or reference is None:
        out["status"] = "blocked"
        out["reason"] = "p99 not reported (insufficient repetitions)"
    elif candidate <= jitter:
        out["status"] = "passed"
    elif reference > jitter:
        out["status"] = "blocked"
        out["reason"] = "the reference leg violates the bound too: node noise, rerun"
    else:
        out["status"] = "failed"
    return out


def run(target: str, candidate: str = "shipped", reference: str = "reference",
        mode: str | None = None, reps: int | None = None, seed: int = 0,
        baseline: bool = False, out_dir: str | None = None) -> dict[str, Any]:
    """Run every check for one candidate plan and write the evidence record."""
    target = resolve(target)
    candidate = candidate or "shipped"
    reference = reference or "reference"
    spec = acceptance.for_target(target)
    mode = mode or spec["latency"]["candidate_rule"]["default_mode"]
    record: dict[str, Any] = {
        "target": target, "candidate": {"plan": candidate}, "reference": {"plan": reference},
        "mode": mode, "acceptance_version": _registry_version(),
        "budget": spec["budget"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": [], "latency": None, "deployment": None, "floor": None, "verdict": None,
    }
    if spec["budget"] is None:
        record["verdict"] = "blocked"
        record["reason"] = f"no acceptance entry with a budget for {target}"
        return _finish(record, out_dir)

    checks = list(spec["correctness"]["checks"])
    record["checks"] += _run_in_engine_checks(target, candidate, checks, seed)
    baseline_scripts = tuple(spec["scripts"].get("official_baseline", ()))
    if "baseline_adapter" in spec["capabilities"]:
        record["checks"] += _run_baseline_checks(baseline_scripts, checks, baseline,
                                                 spec["baseline_python"])

    lat = spec["latency"]
    plans = [reference, candidate, reference]
    latency_report = latency.run(target, plans, reps=reps or lat["reps"], warmup=lat["warmup"],
                                 seed=seed)
    record["latency"] = {"report": latency_report,
                         "rule": _latency_verdict(latency_report, lat, mode)}
    record["deployment"] = _deployment_verdict(latency_report, spec["deployment"])
    try:
        record["floor"] = floor_model.run(target, candidate, seed=seed)
    except Exception as exc:  # the floor is context, never a gate
        record["floor"] = {"error": f"{type(exc).__name__}: {exc}"}

    gates = [c for c in record["checks"] if c["mode"] == "gate"]
    failed = [c["check"] for c in gates if c["status"] == "failed"]
    blocked = [c["check"] for c in gates if c["status"] in ("not_run", "unavailable")]
    rule = record["latency"]["rule"]
    dep = record["deployment"]
    if failed:
        record["verdict"], record["reason"] = "fail", f"gates failed: {failed}"
    elif not rule["run_valid"]:
        record["verdict"] = "blocked"
        record["reason"] = (f"A/B/A run invalid: control spread {rule['control_spread_ms']} ms "
                            f"> {rule['control_spread_max_ms']} ms; rerun")
    elif dep["status"] == "blocked":
        record["verdict"], record["reason"] = "blocked", f"deployment bound: {dep['reason']}"
    elif dep["status"] == "failed":
        record["verdict"] = "fail"
        record["reason"] = (f"candidate tail {dep['candidate_p99_minus_min']:.3f} ms above "
                            f"min exceeds the jitter bound {dep['jitter_ms']} ms")
    elif not rule["passed"]:
        record["verdict"] = "fail"
        record["reason"] = f"candidate rule ({mode}) not met: " + json.dumps(
            {k: rule[k] for k in ("improve", "regressions") if k in rule}, default=str)[:300]
    elif blocked:
        record["verdict"], record["reason"] = "blocked", f"gates not run: {blocked}"
    else:
        record["verdict"] = "pass"
        record["reason"] = f"every gate passed; candidate rule ({mode}) and tail bound hold"
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
             f"target: {record['target']}  candidate: {record['candidate']['plan']}  "
             f"reference: {record['reference']['plan']}  mode: {record.get('mode')}",
             f"acceptance: {record['acceptance_version']}  evidence: {record.get('evidence_path')}"]
    for c in record["checks"]:
        extra = (f" min_cos={c['min_cosine']:.7f} thr={c.get('threshold')}"
                 if "min_cosine" in c else "")
        lines.append(f"  [{c['mode']:6s}] {c['check']:22s} {c['status']}{extra}")
    if record.get("latency"):
        rule = record["latency"]["rule"]
        lines.append(f"  latency: run_valid={rule['run_valid']} spread={rule['control_spread_ms']} "
                     f"bar={rule['promotion_bar_ms']} improve={rule.get('improve')} "
                     f"regressions={rule['regressions']} passed={rule['passed']}")
    if record.get("deployment"):
        d = record["deployment"]
        lines.append(f"  deployment: {d['status']} candidate p99-min={d['candidate_p99_minus_min']} "
                     f"reference={d['reference_p99_minus_min']} jitter={d['jitter_ms']}")
    if record.get("floor") and "totals" in record["floor"]:
        t = record["floor"]["totals"]
        lines.append("  floor: " + ", ".join(f"{k} {v:.0f} us" for k, v in t.items()
                                              if k.endswith("_us")) + f", valid={record['floor']['valid']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidate", default="shipped",
                        help="candidate plan: shipped, reference, JSON or a lab/plans/*.json path")
    parser.add_argument("--reference", default="reference",
                        help="reference plan (default: the Target's reference route)")
    parser.add_argument("--mode", choices=["improve", "no-regression"], default=None,
                        help="candidate rule (default: the registry's default mode)")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true", help="also run the baseline-tier scripts")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    mode = args.mode.replace("-", "_") if args.mode else None
    record = run(args.target, args.candidate, reference=args.reference, mode=mode,
                 reps=args.reps, seed=args.seed, baseline=args.baseline, out_dir=args.out_dir)
    print(summary(record))
    return {"pass": 0, "fail": 1, "blocked": 2}[record["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
