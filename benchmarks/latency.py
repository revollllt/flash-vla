"""End-to-end latency of any Target through the engine protocol.

    python -m benchmarks latency --target h100/pi05
    python -m benchmarks latency --target h100/pi05 --plan reference --plan shipped --plan reference
    python -m benchmarks latency --target h100/pi05 --plan shipped --calibrate   # shipped x3

One request, batch 1, the Target's fixed shapes, clocks unlocked. Every
metric the acceptance registry names (`eval/acceptance.py`) is reported with
`min`, `median` and `p99`:

  chunk_latency     wall clock from inputs available to the chunk available:
                    staging, host slots, every segment launch and replay
  device_latency    CUDA-event time around the same forward
  host_time         wall clock of each declared host slot
  segment_latency   each segment replayed alone, back to back, no host gap
  overhead          chunk latency minus the sum of segment latencies

Deltas are read only within one process. Legs run in the order given; a leg
whose plan repeats the first leg's is a control leg, and the spread between
control legs is the run's minimum detectable effect per statistic. A delta
below it is reported as indistinguishable, and a chunk `min` spread above the
registry's `control_spread_max_ms` marks the whole run invalid. `--calibrate`
runs the first plan three times to measure that spread on its own.

Beside each leg's `metrics` sits an additive `attribution` block
(`benchmarks/attribution.py`): every timed loop's per-forward samples with
their timestamps, the process's per-forward context-switch and page-fault
deltas, the cyclic collector's collections, and a 10 Hz record of the device's
clocks and of the other compute processes on it. A tail is then attributable
from the record instead of argued about. `metrics` keeps its exact shape, so
the report stays readable by anything written against the older schema.

The runner contains no model or stage names: it builds the engine through
`benchmarks.targets`, takes the program from the engine, and samples inputs
from it. Verdicts are not produced here; the promotion gate reads this report
against the acceptance registry.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from typing import Any, Callable

import torch

from eval.acceptance import DEFAULTS
from flash_vla.runtime.identity import Identity, MeasurementContext
from flash_vla.runtime.engine import host_slots, segments

from .attribution import Attribution, LoopTrace
from .attribution import summary as attribution_summary
from .metrics import env_block, require_cuda, report_context
from .targets import PLAN_NAMES, build, resolve

_LAT = DEFAULTS["latency"]


def _stats(samples: list[float], p99_min_reps: int) -> dict[str, Any]:
    ordered = sorted(samples)
    n = len(ordered)
    out = {"min": ordered[0], "median": statistics.median(ordered), "n": n}
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
            attribution: Attribution | None = None) -> dict[str, Any]:
    """Every latency metric of one engine on `inputs`, min/median/p99 each.

    `soak_s` seconds of forwards run first so an unlocked GPU's clocks and
    temperature settle before anything is read. An `attribution` collector, if
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
    metrics: dict[str, Any] = {
        "chunk_latency": timed("chunk_latency", _time_wall, forward),
        "device_latency": timed("device_latency", _time_event, forward),
        "host_time": {},
        "segment_latency": {},
    }
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


def _deltas(legs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-leg deltas against the first leg, and the control spread."""
    first = _flatten(legs[0]["metrics"])

    def same(leg):
        return leg["plan"] == legs[0]["plan"]

    control = [leg for leg in legs[1:] if same(leg)]
    spread: dict[str, float] = {}
    for leg in control:
        for key, value in _flatten(leg["metrics"]).items():
            if key in first:
                spread[key] = max(spread.get(key, 0.0), abs(value - first[key]))
    # The control spread is the run's minimum detectable effect, and above the
    # registry's limit it invalidates the run: the gate blocks rather than
    # reading a delta against a node that did not hold still.
    limit = _LAT["control_spread_max_ms"]
    key = "chunk_latency.min"
    out = {"reference_leg": 0, "control_legs": len(control),
           "minimum_detectable_effect": spread or None,
           "control_spread_ms": spread.get(key) if spread else None,
           "control_spread_max_ms": limit,
           "valid": bool(control) and spread.get(key, float("inf")) <= limit,
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


def _env() -> dict[str, Any]:
    env = env_block()
    env.update({
        "node": platform.node(),
        "job": os.environ.get("SLURM_JOB_ID"),
        "driver": _driver_version(),
        "clocks": _LAT["clocks"],
    })
    return env


def parse_options(items: list[str]) -> dict[str, Any]:
    """`key=value` strings to a dict; true/false and integers are converted."""
    out: dict[str, Any] = {}
    for item in items:
        key, _, value = item.partition("=")
        low = value.lower()
        out[key] = (True if low == "true" else False if low == "false"
                    else int(value) if value.lstrip("-").isdigit() else value)
    return out


def device_selector() -> str | None:
    """What `nvidia-smi -i` must be given to sample the device this process runs on.

    `CUDA_VISIBLE_DEVICES` renumbers the process's view but not `nvidia-smi`'s,
    and on this partition it can hold a UUID rather than an index, so the UUID
    torch reports for device 0 is the only selector that means the same thing to
    both. Falls back to the first visible index, then to every device.
    """
    try:
        uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        if uuid is not None:
            return f"GPU-{uuid}"
    except (AssertionError, RuntimeError, AttributeError):
        pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return visible or None


def _driver_version() -> str | None:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def run(target: str, plans: list[str | None], reps: int = _LAT["reps"],
        warmup: int = _LAT["warmup"], seed: int = 0, calibrate: bool = False,
        attribution: bool = True, **overrides) -> dict[str, Any]:
    """Build one runner per leg, measure it, and report legs, deltas and calibration.

    A leg whose plan equals the first leg's is a control leg. Each leg also
    carries an `attribution` block unless `attribution=False`; the collector
    spans the leg's soak as well as its timed loops.
    """
    require_cuda()
    torch.cuda.init()
    target = resolve(target)
    plans = [plan or "shipped" for plan in plans]
    if calibrate:
        plans = [plans[0]] * 3
    selector = device_selector() if attribution else None
    legs = []
    built = []
    reference_context = None
    reference_identity: Identity | None = None
    for plan in plans:
        if any(item["plan"] == plan for item in built):
            continue
        engine = build(target, plan, seed=seed, **overrides)
        if reference_identity is None:
            reference_identity = engine.identity
        elif not reference_identity.same_workload(engine.identity):
            raise ValueError("plans in one run may differ in implementation only")
        inputs = engine.sample_inputs(seed)
        built.append({"plan": plan, "engine": engine, "inputs": inputs})
    try:
        for index, plan in enumerate(plans):
            print(f"== leg {index}: {target} plan={plan}", flush=True)
            item = next(item for item in built if item["plan"] == plan)
            engine, inputs = item["engine"], item["inputs"]
            context = report_context(engine, _env())
            current_context = MeasurementContext.from_dict(context)
            if reference_context is None:
                reference_context = current_context
            elif reference_context.segment_key != current_context.segment_key:
                raise ValueError("A/B/A measurement context changed; re-anchor in a new segment")
            collector = Attribution(device_index=selector) if attribution else None
            if collector is None:
                metrics = measure(engine, inputs, reps, warmup, _LAT["p99_min_reps"],
                                  soak_s=_LAT["soak_s"])
                evidence = None
            else:
                with collector:
                    metrics = measure(engine, inputs, reps, warmup, _LAT["p99_min_reps"],
                                      soak_s=_LAT["soak_s"], attribution=collector)
                evidence = collector.as_dict()
            legs.append({"leg": index, "plan": plan, "identity": engine.identity.as_dict(),
                         "measurement_context": context, "metrics": metrics, "attribution": evidence})
            print(json.dumps(legs[-1]["metrics"]), flush=True)
            if evidence is not None:
                print(attribution_summary(evidence), flush=True)
    finally:
        built.clear()
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "identity": legs[0]["identity"],
        "measurement_context": legs[0]["measurement_context"],
        "env": _env(),
        "config": {"reps": reps, "warmup": warmup, "seed": seed, "plans": plans,
                   "calibration": calibrate, "statistics": list(_LAT["statistics"]),
                   "p99_min_reps": _LAT["p99_min_reps"],
                   "promotion_bar_ms": _LAT["promotion_bar_ms"],
                   "control_spread_max_ms": _LAT["control_spread_max_ms"],
                   "attribution": attribution},
        "legs": legs,
        "deltas": _deltas(legs) if len(legs) > 1 else None,
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
                             "repeat for A/B/A (default: shipped)")
    parser.add_argument("--reps", type=int, default=_LAT["reps"])
    parser.add_argument("--warmup", type=int, default=_LAT["warmup"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true",
                        help="run the first plan three times and report the spread")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--option", action="append", default=[],
                        help="target-local construction option as key=value, every leg")
    parser.add_argument("--no-attribution", dest="attribution", action="store_false",
                        help="skip the per-leg attribution record (no nvidia-smi sampler, "
                             "no per-forward sample list)")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in (("steps", args.steps), ("layers", args.layers))
                 if v is not None}
    overrides.update(parse_options(args.option))
    report = run(args.target, args.plan or [None], reps=args.reps, warmup=args.warmup,
                 seed=args.seed, calibrate=args.calibrate, attribution=args.attribution,
                 **overrides)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
