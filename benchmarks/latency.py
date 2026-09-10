"""End-to-end latency of any Target through the engine protocol.

    python -m benchmarks latency --target h100/pi05
    python -m benchmarks latency --target h100/pi05 --plan reference --plan shipped
    python -m benchmarks latency --target h100/pi05 --plan shipped --calibrate   # shipped x3

One request, batch 1, the Target's fixed shapes, inherited device clock state.
The default measures only chunk_latency: wall time including input staging,
host work and graph replay to the available chunk. Use --breakdown for CUDA-event,
host-slot, segment and overhead diagnostics; use --attribution for telemetry.
Both are opt-in. Statistics retain min, median, p99 and ordered raw samples.
Median is the primary daily comparison; p99 from 100 samples is descriptive.

Each version runs in a fresh process and uses only its initial capture. One or
two versions are sufficient. A repeated control or --calibrate optionally
estimates drift; without it, noise is unknown, not zero. A/B comparisons require
the same device, workload and measurement context but do not require a third leg.

Time-ordered `samples_ms` are retained even without attribution. Device-state
observations before and after each leg remain separate from stable context
identity; they are evidence for drift, not a request to change clocks.

Beside each leg's `metrics` sits an additive `attribution` block
(`tools/profiling/attribution.py`): every timed loop's per-forward samples with
their timestamps, the process's per-forward context-switch and page-fault
deltas, the cyclic collector's collections, and a 10 Hz record of the device's
clocks and of the other compute processes on it. A tail is then attributable
from the record instead of argued about. `--breakdown` retains the complete metric
set expected by explicit legacy qualification.

The runner contains no model or stage names: it builds the engine through
`flash_vla.inference`, takes the program from the engine, and samples inputs
from it. This runner reports measurements; it does not promote code.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tools.profiling.attribution import Attribution, LoopTrace

import torch

from benchmarks.config import LATENCY_DEFAULTS
from flash_vla.runtime.identity import Identity, MeasurementContext
from flash_vla.runtime.engine import host_slots, segments

from flash_vla.environment import collect as _env, device_selector, require_cuda, report_context
from flash_vla.inference import PLAN_NAMES, build, resolve, parse_options

_LAT = LATENCY_DEFAULTS


def _stats(samples: list[float], p99_min_reps: int) -> dict[str, Any]:
    ordered = sorted(samples)
    n = len(ordered)
    out = {"min": ordered[0], "median": statistics.median(ordered), "n": n,
           "samples_ms": samples}
    if n >= p99_min_reps:
        out["p99"] = ordered[min(n - 1, int(round(0.99 * (n - 1))))]
    else:
        out["p99"] = None
        out["p99_note"] = f"insufficient: {n} < {p99_min_reps} repetitions"
    return out


def _time_wall(call: Callable[[], Any], reps: int, warmup: int,
               trace: LoopTrace | None = None) -> list[float]:
    """Wall clock of `call` plus a synchronize, `reps` times.

    A `trace` is filled outside the timed region only: the repetition's start
    is the same `perf_counter` reading the sample is computed from, and the
    process counters are read after the synchronize has returned.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    if trace is not None:
        trace.enter()
    for index in range(reps):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
        if trace is not None:
            trace.start(index, start)
            trace.mark(index)
    if trace is not None:
        trace.leave()
    return samples


def _time_event(call: Callable[[], Any], reps: int, warmup: int,
                trace: LoopTrace | None = None) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    if trace is not None:
        trace.enter()
    for index in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        if trace is not None:
            trace.start(index, host_start)
            trace.mark(index)
    if trace is not None:
        trace.leave()
    return samples


def measure(engine, inputs: dict[str, Any], reps: int, warmup: int,
            p99_min_reps: int, soak_s: float = 0.0,
            attribution: Attribution | None = None, *,
            breakdown: bool = False) -> dict[str, Any]:
    """Chunk latency, plus optional per-device/stage diagnostics on the same engine.

    Optional `soak_s` seconds of forwards precede warmup. Fixed-time load
    does not certify stable device clocks or latency. An `attribution` collector, if
    given, is filled with one record per timed loop, keyed by the same
    `metric.name` the flattened report uses; it is read by its owner after the
    collector's context closes and never appears in `metrics`.
    """
    def timed(key: str, timer: Callable[..., list[float]], call: Callable[[], Any]):
        trace: LoopTrace | None = attribution.trace(key, reps) if attribution else None
        samples = timer(call, reps, warmup, trace)
        if attribution is not None and trace is not None:
            attribution.record(trace, samples)
        return _stats(samples, p99_min_reps)

    engine.forward(**inputs)                     # settles any host-side state
    torch.cuda.synchronize()
    deadline = time.perf_counter() + soak_s
    while time.perf_counter() < deadline:
        engine.forward(**inputs)
    torch.cuda.synchronize()
    forward = lambda: engine.forward(**inputs)   # noqa: E731
    metrics = {"chunk_latency": timed("chunk_latency", _time_wall, forward)}
    if not breakdown:
        return metrics
    metrics.update(device_latency=timed("device_latency", _time_event, forward),
                   host_time={}, segment_latency={})
    for slot in host_slots(engine):
        metrics["host_time"][slot] = timed(
            f"host_time.{slot}", _time_wall, lambda slot=slot: engine.host(slot, **inputs))
    for name in segments(engine):
        metrics["segment_latency"][name] = timed(
            f"segment_latency.{name}", _time_event, lambda name=name: engine.replay(name))
    # Overhead is the chunk statistic minus the sum of the segments' same
    # statistic; for `p99` that is a difference of tails, not a tail of a
    # difference, and is reported as such.
    overhead = {}
    for stat in ("min", "median", "p99"):
        if metrics["chunk_latency"].get(stat) is None or any(
                s.get(stat) is None for s in metrics["segment_latency"].values()):
            continue
        overhead[stat] = metrics["chunk_latency"][stat] - sum(
            s[stat] for s in metrics["segment_latency"].values())
    metrics["overhead"] = overhead
    return metrics


def _flatten(metrics: dict[str, Any]) -> dict[str, float]:
    """`metric.stat` -> value, segments and host slots by name."""
    flat: dict[str, float] = {}
    for metric, value in metrics.items():
        if metric in ("host_time", "segment_latency"):
            for name, stats in value.items():
                for stat in ("min", "median", "p99"):
                    if stats.get(stat) is not None:
                        flat[f"{metric}.{name}.{stat}"] = stats[stat]
        elif metric == "overhead":
            for stat, v in value.items():
                flat[f"overhead.{stat}"] = v
        else:
            for stat in ("min", "median", "p99"):
                if value.get(stat) is not None:
                    flat[f"{metric}.{stat}"] = value[stat]
    return flat


def _deltas(legs: list[dict[str, Any]], control_spread_max_ms: float | None = None) -> dict[str, Any]:
    """Per-leg deltas against the first leg, and the control spread."""
    first = _flatten(legs[0]["metrics"])

    def same(leg):
        return (leg["plan"] == legs[0]["plan"]
                and leg.get("identity", {}) == legs[0].get("identity", {})
                and leg.get("options", {}) == legs[0].get("options", {}))

    control = [leg for leg in legs[1:] if same(leg)]
    spread: dict[str, float] = {}
    for leg in control:
        for key, value in _flatten(leg["metrics"]).items():
            if key in first:
                spread[key] = max(spread.get(key, 0.0), abs(value - first[key]))
    # Optional repeated controls estimate drift. Without controls, the
    # measured difference is still valid but its noise remains unknown.
    limit = control_spread_max_ms
    key = "chunk_latency.min"
    out = {"reference_leg": 0, "control_legs": len(control),
           "minimum_detectable_effect": spread or None,
           "control_spread_ms": spread.get(key) if spread else None,
           "control_spread_max_ms": limit,
           "valid": (True if not control else None if limit is None
                     else spread.get(key, float("inf")) <= limit),
           "legs": []}
    for index, leg in enumerate(legs[1:], start=1):
        flat = _flatten(leg["metrics"])
        deltas = {}
        for key, value in flat.items():
            if key not in first:
                continue
            delta = value - first[key]
            entry: dict[str, Any] = {"delta": delta}
            if key in spread:
                entry["distinguishable"] = abs(delta) > spread[key]
            deltas[key] = entry
        out["legs"].append({"leg": index, "plan": leg["plan"],
                            "same_as_reference": same(leg), "deltas": deltas})
    return out








def _measure_leg(target, plan, *, reps, warmup, seed, soak_s, attribution, options, breakdown=False):
    """Build once and measure the initial capture in a fresh worker process."""
    require_cuda()
    torch.cuda.init()
    engine = build(target, plan, seed=seed, **options)
    inputs = engine.sample_inputs(seed)
    environment = _env(engine.device)
    runtime_before = environment.get("runtime_observation")
    context = report_context(engine, environment)
    collector = None
    if attribution:
        from tools.profiling.attribution import Attribution, summary as attribution_summary
        collector = Attribution(device_index=device_selector(engine.device))
    with torch.cuda.device(engine.device):
        if collector is None:
            metrics = measure(engine, inputs, reps, warmup, _LAT["p99_min_reps"],
                              soak_s=soak_s, breakdown=breakdown)
            evidence = None
        else:
            with collector:
                metrics = measure(engine, inputs, reps, warmup, _LAT["p99_min_reps"],
                                  soak_s=soak_s, attribution=collector, breakdown=breakdown)
            evidence = collector.as_dict()
    environment = _env(engine.device)
    after = MeasurementContext.from_dict(report_context(engine, environment))
    if after.segment_key != MeasurementContext.from_dict(context).segment_key:
        raise ValueError("latency measurement context changed during a leg; repeat that measurement")
    leg = {"plan": plan, "identity": engine.identity.as_dict(),
           "measurement_context": context, "metrics": metrics, "attribution": evidence,
           "runtime_observation": {"before": runtime_before,
                                   "after": environment.get("runtime_observation")},
           "options": options, "process_id": os.getpid(),
           "gpu_uuid": environment.get("gpu_uuid"),
           "implementation_source": getattr(engine, "implementation_source", None)}
    if evidence is not None:
        print(attribution_summary(evidence), flush=True)
    return leg, environment


def _run_leg(target, plan, **kwargs):
    """Use exec, not fork, so no CUDA context or previous graph survives."""
    with tempfile.TemporaryDirectory(prefix="flash-vla-latency-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        request.write_text(json.dumps(dict(target=target, plan=plan, **kwargs)))
        subprocess.run([sys.executable, "-m", "benchmarks.latency", "--worker",
                        str(request), str(response)], check=True)
        return json.loads(response.read_text())


def run(target: str, plans: list[str | None], reps: int = _LAT["reps"],
        warmup: int = _LAT["warmup"], seed: int = 0, calibrate: bool = False,
        soak_s: float = _LAT["soak_s"],
        attribution: bool = False, breakdown: bool = False, leg_options: list[dict[str, Any]] | None = None,
        control_spread_max_ms: float | None = None, **overrides) -> dict[str, Any]:
    """Measure each leg's first capture in a fresh process, then compare.

    A leg whose plan equals the first leg's is a control leg. Worker startup,
    loading and capture are outside the measured latency.
    """
    target = resolve(target)
    plans = [plan or "shipped" for plan in plans]
    options = list(leg_options or [{} for _ in plans])
    if len(options) != len(plans):
        raise ValueError("leg_options must match the number of plans")
    if calibrate:
        plans = [plans[0]] * 3
        options = [options[0]] * 3
    legs = []
    for index, (plan, option) in enumerate(zip(plans, options)):
        print(f"== leg {index}: {target} plan={plan}", flush=True)
        leg, environment = _run_leg(target, plan, reps=reps, warmup=warmup, seed=seed,
                                    soak_s=soak_s, attribution=attribution, breakdown=breakdown,
                                    options={**overrides, **option})
        if legs:
            if leg["gpu_uuid"] != legs[0]["gpu_uuid"]:
                raise ValueError("latency comparison requires the same physical GPU")
            if not Identity.from_dict(legs[0]["identity"]).same_workload(
                    Identity.from_dict(leg["identity"])):
                raise ValueError("plans in one run may differ in implementation only")
            if (MeasurementContext.from_dict(legs[0]["measurement_context"]).segment_key
                    != MeasurementContext.from_dict(leg["measurement_context"]).segment_key):
                raise ValueError("latency measurement context changed; measure each environment separately")
        leg["leg"] = index
        legs.append(leg)

    report = {
        "identity": legs[0]["identity"],
        "protocol": "latency-v2",
        "instrumented": attribution,
        "measurement_context": legs[0]["measurement_context"],
        "env": environment,
        "config": {"reps": reps, "warmup": warmup, "seed": seed, "plans": plans,
                   "calibration": calibrate, "breakdown": breakdown,
                   "primary_statistic": _LAT["primary_statistic"], "capture_policy": "fresh-process-first-capture-per-leg",
                   "statistics": list(_LAT["statistics"]),
                   "p99_min_reps": _LAT["p99_min_reps"],
                   "soak_s": soak_s,
                   "control_spread_max_ms": control_spread_max_ms,
                   "attribution": attribution, "leg_options": options},
        "legs": legs,
        "deltas": _deltas(legs, control_spread_max_ms) if len(legs) > 1 else None,
    }
    if calibrate:
        report["calibration"] = report["deltas"]["minimum_detectable_effect"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, help="h100/pi05, h100/pi0, or a full name")
    parser.add_argument("--plan", action="append", default=None,
                        help=f"one of {PLAN_NAMES}, a JSON object or a lab/plans/*.json path; "
                             "repeat for A/B (default: shipped)")
    parser.add_argument("--reps", type=int, default=_LAT["reps"])
    parser.add_argument("--warmup", type=int, default=_LAT["warmup"])
    parser.add_argument("--soak-seconds", type=float, default=_LAT["soak_s"],
                        help="optional pre-warmup load period (default: no soak)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true",
                        help="run the first plan three times and report the spread")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--option", action="append", default=[],
                        help="target-local construction option as key=value, every leg")
    parser.add_argument("--breakdown", action="store_true",
                        help="also measure device, host, segment and overhead diagnostics")
    parser.add_argument("--attribution", action="store_true",
                        help="collect CPU/GPU telemetry (diagnostic timing)")
    parser.add_argument("--no-attribution", dest="attribution", action="store_false",
                        help="skip per-leg CPU/GPU attribution collection")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.set_defaults(attribution=False)
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in (("steps", args.steps), ("layers", args.layers))
                 if v is not None}
    overrides.update(parse_options(args.option))
    report = run(args.target, args.plan or [None], reps=args.reps, warmup=args.warmup,
                 seed=args.seed, calibrate=args.calibrate, attribution=args.attribution,
                 soak_s=args.soak_seconds, breakdown=args.breakdown,
                 **overrides)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        request, response = map(Path, sys.argv[2:])
        response.write_text(json.dumps(_measure_leg(**json.loads(request.read_text()))))
    else:
        raise SystemExit(main())
