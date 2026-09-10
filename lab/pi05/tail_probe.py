"""Where inside one forward does the host block, and where does a late forward lose its time.

    python -m lab.pi05.tail_probe --target h100/pi05 --reps 400
    python -m lab.pi05.tail_probe --target h100/pi0 --reps 400

The per-leg attribution record says a Pi0.5 forward costs exactly seven
voluntary context switches, that a late one costs thirteen to twenty-one, and
that neither the host slot nor any single stage replay accounts for more than
0.16 of them when it is timed alone (job 599723). Seven blocking waits
therefore appear only when the steps run as one forward, and the record cannot
say which step they belong to because it measures whole loops.

This probe walks the same steps `ModelRunner.forward` walks -- stage, then each
host slot and stage replay in program order, then the synchronize a caller
needs before it can read the actions -- and reads `perf_counter` and
`getrusage` between them. Per step it reports the wall time and the voluntary
and involuntary context switches, split by whether that repetition was late.
The step that both blocks and absorbs a late forward's excess is the one that
owns the tail.

It measures a different thing from `benchmarks latency`, and deliberately: the
counters between steps cost a syscall each, so the numbers here are for
attribution and are never a latency claim.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from typing import Any, Callable

import torch

from flash_vla.inference import build, resolve
from benchmarks.config import LATENCY_DEFAULTS as _LAT

_JITTER = 0.5


def _host_halves(engine, inputs: dict[str, Any], name: str):
    """Pi0.5's host slot as its two halves, or None when the Target is not built that way.

    The slot is host computation followed by staged copies
    (`hardware/nvidia/h100/pi05/prefix.py`), and the probe has to separate them:
    the first is CPU work and the second is CUDA API calls, and only one of them
    can be the step that goes late. Splitting here rather than in the Target
    keeps the deployment path's `host` the one thing the runner calls.
    """
    state = getattr(engine, "host_state", None)
    if state is None or not (hasattr(state, "build") and hasattr(state, "copy_into")):
        return None
    return [(f"host:{name}:build", lambda: state.build(inputs.get("state"))),
            (f"host:{name}:copy", lambda: state.copy_into(engine.buffers))]


def _steps(engine, inputs: dict[str, Any],
           split_host: bool = False) -> list[tuple[str, Callable[[], Any]]]:
    """The forward's steps, in the order the runner walks them, plus the caller's wait."""
    steps: list[tuple[str, Callable[[], Any]]] = [("stage", lambda: engine.stage(**inputs))]
    for step in engine.program:
        if step.kind != "host":
            steps.append((f"replay:{step.name}", lambda n=step.name: engine.replay(n)))
            continue
        halves = _host_halves(engine, inputs, step.name) if split_host else None
        steps.extend(halves or [(f"host:{step.name}",
                                 lambda n=step.name: engine.host(n, **inputs))])
    steps.append(("synchronize", torch.cuda.synchronize))
    return steps


def probe(engine, inputs: dict[str, Any], reps: int, warmup: int, soak_s: float,
          split_host: bool = False) -> dict[str, Any]:
    """Per step of `reps` forwards: wall time and context switches, preallocated."""
    steps = _steps(engine, inputs, split_host)
    names = [name for name, _ in steps]
    for _ in range(warmup):
        engine.forward(**inputs)
    torch.cuda.synchronize()
    deadline = time.perf_counter() + soak_s
    while time.perf_counter() < deadline:
        engine.forward(**inputs)
    torch.cuda.synchronize()

    times = [[0.0] * reps for _ in names]
    nvcsw = [[0] * reps for _ in names]
    nivcsw = [[0] * reps for _ in names]
    minflt = [[0] * reps for _ in names]
    total = [0.0] * reps
    for rep in range(reps):
        mark = time.perf_counter()
        start = mark
        usage = resource.getrusage(resource.RUSAGE_SELF)
        for index, (_, call) in enumerate(steps):
            call()
            now = time.perf_counter()
            after = resource.getrusage(resource.RUSAGE_SELF)
            times[index][rep] = (now - mark) * 1e3
            nvcsw[index][rep] = after.ru_nvcsw - usage.ru_nvcsw
            nivcsw[index][rep] = after.ru_nivcsw - usage.ru_nivcsw
            minflt[index][rep] = after.ru_minflt - usage.ru_minflt
            mark, usage = now, after
        total[rep] = (mark - start) * 1e3

    floor = min(total)
    late = [i for i, value in enumerate(total) if value - floor > _JITTER]
    prompt = [i for i in range(reps) if i not in set(late)]
    return {
        "steps": names, "reps": reps, "total_ms": [round(v, 6) for v in total],
        "floor_ms": floor, "late_indices": late,
        "per_step": {name: {
            "ms": [round(v, 6) for v in times[i]],
            "nvcsw": nvcsw[i], "nivcsw": nivcsw[i],
            "ms_median": statistics.median(times[i]),
            "ms_max": max(times[i]),
            "nvcsw_median": statistics.median(nvcsw[i]),
            "nvcsw_total": sum(nvcsw[i]),
            "nivcsw_total": sum(nivcsw[i]),
            "minflt_total": sum(minflt[i]),
            "minflt_when_late": sum(minflt[i][j] for j in late),
            "ms_median_when_late": statistics.median([times[i][j] for j in late]) if late else None,
            "nvcsw_median_when_late": (statistics.median([nvcsw[i][j] for j in late])
                                       if late else None),
            "ms_median_when_prompt": (statistics.median([times[i][j] for j in prompt])
                                      if prompt else None),
        } for i, name in enumerate(names)},
    }


def table(report: dict[str, Any]) -> str:
    late = len(report["late_indices"])
    head = (f"{'step':26s} {'ms med':>8s} {'ms max':>9s} {'nvcsw tot':>10s} "
            f"{'nivcsw tot':>11s} {'minflt tot':>11s} {'minflt late':>12s} "
            f"{'ms med (on time)':>17s} {'ms med (late)':>14s}")
    lines = [f"reps {report['reps']}, floor {report['floor_ms']:.3f} ms, "
             f"{late} repetitions above floor + {_JITTER} ms", head, "-" * len(head)]
    for name in report["steps"]:
        s = report["per_step"][name]
        lines.append(f"{name:26s} {s['ms_median']:8.4f} {s['ms_max']:9.4f} "
                     f"{s['nvcsw_total']:10d} {s['nivcsw_total']:11d} "
                     f"{s['minflt_total']:11d} {s['minflt_when_late']:12d} "
                     f"{s['ms_median_when_prompt']:17.4f} "
                     + (f"{s['ms_median_when_late']:14.4f}" if late else f"{'-':>14s}"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="h100/pi05")
    parser.add_argument("--plan", default="shipped")
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-host", action="store_true",
                        help="time the host slot as its two halves, the host computation and "
                             "the staged copies, where the Target is built that way")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    target = resolve(args.target)
    engine = build(target, args.plan, seed=args.seed)
    inputs = engine.sample_inputs(args.seed)
    report = probe(engine, inputs, args.reps, _LAT["warmup"], _LAT["soak_s"],
                   split_host=args.split_host)
    report.update(target=target, plan=args.plan,
                  env={"node": platform.node(), "job": os.environ.get("SLURM_JOB_ID"),
                       "cores": sorted(os.sched_getaffinity(0)), "torch": torch.__version__})
    report["split_host"] = args.split_host
    print(f"== {target} plan={args.plan} node={report['env']['node']} "
          f"split_host={args.split_host}")
    print(table(report))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[report] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
