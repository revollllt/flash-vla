"""The four decisive experiments on Pi0.5's chunk-latency tail, one job, one node.

    python -m lab.pi05.tail_experiments --plan shipped --plan reference
    python -m lab.pi05.tail_experiments --experiment order --legs 8

Pi0.5's chunk latency intermittently reads 2.5 to 17.3 ms above its own `min`,
against the acceptance registry's 0.5 ms deployment bound; Pi0, measured the
same way in the same jobs, never does. Four mechanisms are on the table
(`.agents/notes/proposed/performance/2026-09-06-pi05-chunk-tail-attribution.md`)
and each has a treatment that removes exactly one of them:

  order      run the `prompt` host slot before the `vision_encoder` replay, so
             the three replays follow back to back and the GPU never waits on
             the host mid-forward
  affinity   bind the measuring thread to one core of the job's cgroup
  d2d        write the four per-inference buffers by device-to-device copy from
             a snapshot of the pinned staging, taken once
  gc         disable Python's cyclic collector

Every treatment is applied to an **already-constructed engine** and undone
after the leg, so a control leg and a treated leg share one engine, one
process, one CUDA context, one node and one set of weights, and differ only in
the treatment. Legs alternate A/B/A/B..., which buys three independent p99
readings per arm instead of one: historically 22 of 57 legs carried a tail, so
a single clean treated leg proves nothing.

Each leg is measured by `benchmarks.latency.measure`, the same function the
deployment harness and the promotion gate use, and carries the same
`attribution` record, so a treated leg's tail is read against the control legs
of its own experiment on the same numbers the gate would read.

Nothing here is a deployment configuration. Whichever treatment the record
supports becomes either a change to the deployment path or a stated deployment
requirement, in its own change, with the gate's verdict as evidence.
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

from tools.profiling.attribution import Attribution
from tools.profiling.attribution import summary as attribution_summary
from benchmarks.latency import device_selector, measure
from flash_vla.inference import build, resolve
from lab.optimize.policy import DEFAULTS

_LAT = DEFAULTS["latency"]
#: The four buffers Pi0.5's host slot writes every inference, and the pinned
#: staging tensors they are copied from (`hardware/nvidia/h100/pi05/prefix.py`).
_STAGED = (("prompt_ids", "token_ids"), ("prompt_scale", "embed_scale"),
           ("mask_bias", "mask_bias"), ("action_expert_rope", "decoder_rope"))

Restore = Callable[[], None]


# -- treatments ------------------------------------------------------------

def _treat_order(engine, inputs: dict[str, Any]) -> Restore:
    """Move every host slot ahead of every replay for the duration of one leg.

    `ModelRunner.forward` walks `self.program`, so the order is a property of
    that tuple and not of the captured graphs, which are per stage. It is legal
    for Pi0.5: the slot writes `prompt_ids`, `prompt_scale`, `mask_bias` and
    `action_expert_rope`, none of which `vision_encoder` reads, and all of which
    are written before their first reader either way.
    """
    before = engine.program
    engine.program = tuple(sorted(before, key=lambda step: step.kind != "host"))
    return lambda: setattr(engine, "program", before)


def _treat_affinity(engine, inputs: dict[str, Any]) -> Restore:
    """Bind the measuring thread to the lowest core of the job's cgroup.

    `sched_setaffinity(0, ...)` binds the calling *thread*, so the driver's own
    threads keep the whole cgroup and only the thread that issues the forward is
    constrained. That is the deployment question: does the host thread being
    moved or preempted explain the tail.
    """
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(before)})
    return lambda: os.sched_setaffinity(0, before)


def _treat_d2d(engine, inputs: dict[str, Any]) -> Restore:
    """Serve the four per-inference copies from device memory instead of pinned host memory.

    The sampled `state` is constant across a run, so the staged values never
    change and a device snapshot writes exactly the bytes the host-to-device
    path would have written. Tokenization still runs; only the copies move.
    """
    state = engine.host_state
    engine.host("prompt", **inputs)          # fill the pinned staging once
    staged = [(buffer, getattr(state, source).to(engine.device, copy=True))
              for buffer, source in _STAGED]
    torch.cuda.synchronize()

    @torch.no_grad()
    def copy_into(buffers, non_blocking: bool = True) -> None:
        for name, source in staged:
            buffers[name].copy_(source, non_blocking=non_blocking)

    state.copy_into = copy_into
    return lambda: state.__dict__.pop("copy_into", None)


def _treat_gc(engine, inputs: dict[str, Any]) -> Restore:
    """Turn Python's cyclic collector off for one leg."""
    enabled = gc.isenabled()
    gc.disable()

    def restore() -> None:
        if enabled:
            gc.enable()
        gc.collect()

    return restore


TREATMENTS: dict[str, tuple[str, Callable[..., Restore]]] = {
    "order": ("the GPU waits on the host at the mid-forward dependency", _treat_order),
    "affinity": ("the host thread is moved or preempted on the shared node", _treat_affinity),
    "d2d": ("the four pinned host-to-device copies stall", _treat_d2d),
    "gc": ("a cyclic collection runs across a forward", _treat_gc),
}


# -- the run ---------------------------------------------------------------

def _tail(stats: dict[str, Any]) -> float | None:
    return None if stats.get("p99") is None else stats["p99"] - stats["min"]


def _late_total(block: dict[str, Any] | None) -> int:
    if not block:
        return 0
    return sum(len(loop.get("late", [])) for loop in block.get("loops", {}).values())


def _parity(engine, inputs: dict[str, Any], treat: Callable[..., Restore]) -> dict[str, Any]:
    """One forward before and after the treatment: the output must be bit-identical.

    None of these treatments may change a number. This is the cheap check that
    says so before three minutes of timing are spent on the leg.
    """
    reference = engine.forward(**inputs).clone()
    restore = treat(engine, inputs)
    try:
        treated = engine.forward(**inputs).clone()
    finally:
        restore()
    return {"bit_identical": bool(torch.equal(reference, treated)),
            "max_abs": float((reference.float() - treated.float()).abs().max())}


def run_leg(engine, inputs: dict[str, Any], *, reps: int, selector: str | None,
            treat: Callable[..., Restore] | None) -> dict[str, Any]:
    """One leg: apply the treatment if there is one, measure, undo it."""
    restore = treat(engine, inputs) if treat is not None else (lambda: None)
    try:
        collector = Attribution(device_index=selector)
        with collector:
            metrics = measure(engine, inputs, reps, _LAT["warmup"], _LAT["p99_min_reps"],
                              soak_s=_LAT["soak_s"], attribution=collector)
    finally:
        restore()
    return {"metrics": metrics, "attribution": collector.as_dict()}


def run(target: str, plans: list[str], experiments: list[str], *, reps: int, legs: int,
        seed: int) -> dict[str, Any]:
    """Every experiment on every plan, one engine per plan, legs alternating A/B."""
    target = resolve(target)
    selector = device_selector()
    out: dict[str, Any] = {
        "target": target, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": {"node": platform.node(), "job": os.environ.get("SLURM_JOB_ID"),
                "torch": torch.__version__, "cores": sorted(os.sched_getaffinity(0)),
                "clocks": _LAT["clocks"]},
        "config": {"plans": plans, "experiments": experiments, "reps": reps, "legs": legs,
                   "seed": seed, "warmup": _LAT["warmup"], "soak_s": _LAT["soak_s"],
                   "jitter_ms": DEFAULTS["deployment"]["jitter_ms"]},
        "identity": {}, "parity": [], "legs": [],
    }
    for plan in plans:
        print(f"== building {target} plan={plan}", flush=True)
        engine = build(target, plan, seed=seed)
        inputs = engine.sample_inputs(seed)
        out["identity"][plan] = engine.identity.as_dict()
        for name in experiments:
            thesis, treat = TREATMENTS[name]
            parity = _parity(engine, inputs, treat)
            parity.update(experiment=name, plan=plan)
            out["parity"].append(parity)
            print(f"-- {name} on {plan}: {thesis}; bit-identical={parity['bit_identical']}",
                  flush=True)
            for index in range(legs):
                treated = bool(index % 2)
                leg = run_leg(engine, inputs, reps=reps, selector=selector,
                              treat=treat if treated else None)
                leg.update(experiment=name, plan=plan, leg=index, treated=treated)
                out["legs"].append(leg)
                print(_leg_line(leg), flush=True)
                print(attribution_summary(leg["attribution"]), flush=True)
        del engine
        torch.cuda.empty_cache()
    out["summary"] = summarize(out["legs"], DEFAULTS["deployment"]["jitter_ms"])
    return out


def _leg_line(leg: dict[str, Any]) -> str:
    chunk = leg["metrics"]["chunk_latency"]
    device = leg["metrics"]["device_latency"]
    arm = "treated" if leg["treated"] else "control"
    return (f"  {leg['experiment']:9s} {leg['plan']:10s} leg {leg['leg']} {arm:8s} "
            f"chunk min {chunk['min']:7.3f} median {chunk['median']:7.3f} "
            f"tail {_tail(chunk):7.3f} | device tail {_tail(device):7.3f} | "
            f"late {_late_total(leg.get('attribution'))}")


def summarize(legs: list[dict[str, Any]], jitter_ms: float) -> list[dict[str, Any]]:
    """Per experiment, plan and arm: the tails of every leg and how many broke the bound."""
    rows: dict[tuple[str, str, bool], dict[str, Any]] = {}
    for leg in legs:
        key = (leg["experiment"], leg["plan"], leg["treated"])
        row = rows.setdefault(key, {"experiment": key[0], "plan": key[1],
                                    "arm": "treated" if key[2] else "control",
                                    "legs": 0, "chunk_min_ms": [], "chunk_tail_ms": [],
                                    "device_tail_ms": [], "late": 0, "over_bound": 0})
        chunk, device = leg["metrics"]["chunk_latency"], leg["metrics"]["device_latency"]
        row["legs"] += 1
        row["chunk_min_ms"].append(round(chunk["min"], 4))
        row["chunk_tail_ms"].append(round(_tail(chunk) or 0.0, 4))
        row["device_tail_ms"].append(round(_tail(device) or 0.0, 4))
        row["late"] += _late_total(leg.get("attribution"))
        row["over_bound"] += int((_tail(chunk) or 0.0) > jitter_ms
                                 or (_tail(device) or 0.0) > jitter_ms)
    for row in rows.values():
        row["chunk_min_median_ms"] = statistics.median(row["chunk_min_ms"])
        row["worst_chunk_tail_ms"] = max(row["chunk_tail_ms"])
        row["worst_device_tail_ms"] = max(row["device_tail_ms"])
    return list(rows.values())


def table(summary: list[dict[str, Any]]) -> str:
    head = (f"{'experiment':10s} {'plan':10s} {'arm':8s} {'legs':>4s} {'min med':>9s} "
            f"{'worst chunk tail':>17s} {'worst dev tail':>15s} {'over bound':>11s} {'late':>5s}")
    lines = [head, "-" * len(head)]
    for row in summary:
        lines.append(f"{row['experiment']:10s} {row['plan']:10s} {row['arm']:8s} "
                     f"{row['legs']:4d} {row['chunk_min_median_ms']:9.3f} "
                     f"{row['worst_chunk_tail_ms']:17.3f} {row['worst_device_tail_ms']:15.3f} "
                     f"{row['over_bound']:11d} {row['late']:5d}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="h100/pi05")
    parser.add_argument("--plan", action="append", default=None,
                        help="repeat for several plans (default: shipped and reference)")
    parser.add_argument("--experiment", action="append", default=None,
                        choices=sorted(TREATMENTS), help=f"default: all of {sorted(TREATMENTS)}")
    parser.add_argument("--reps", type=int, default=_LAT["reps"])
    parser.add_argument("--legs", type=int, default=6,
                        help="legs per experiment and plan, alternating control/treated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    report = run(args.target, args.plan or ["shipped", "reference"],
                 args.experiment or sorted(TREATMENTS), reps=args.reps, legs=args.legs,
                 seed=args.seed)
    print("\n" + table(report["summary"]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[report] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
