"""Per-call-site GPU time in isolation, for any Target: python -m benchmarks kernels.

    python -m benchmarks kernels --target h100/pi05
    python -m benchmarks kernels --target h100/pi05 --segment action_expert --site action_expert_attention
    python -m benchmarks kernels --target h100/pi0 --plan reference --timer cupti --csv out.csv

Where `profile` attributes time inside the captured graph, this launches one
call site at a time outside any graph, the way FlashInfer benchmarks its
kernels, against the engine's real weights and buffers. The invocations of
every call site are recorded from one instrumented eager run of the segment,
so the arguments are exactly what the pipeline passes, and the recorded
invocations are cycled while timing so a weight-heavy call site reads cold
(each layer's weight is a first touch, as in the graph).

Timing backends (--timer):
  cudagraph  the repository's amortised in-graph regime: `n_inner` recorded
             invocations captured into one graph, replayed, divided (default)
  cupti      hardware-level kernel time of the first recorded invocation,
             cold L2 by flush (needs cupti-python; auto-fallback to events)
  events     CUDA events around the first recorded invocation, cold L2 by flush

FLOPs and bytes come from the graph's derived costs, so the achieved TFLOP/s
and TB/s are against the same minimal-traffic model the floor uses.

Call sites a plan must invoke together (`engine.atomic_groups`: a producer
and the persistent consumer that waits on its counters) are one case, their
recorded invocations replayed in pipeline order, because one of them alone is
not a valid program. A case whose derived cost is zero at the Target's shape
issues no kernel and is skipped rather than timed.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any, Callable

import torch

from flash_vla.bench import KernelResult, bench_gpu_time, render_table, write_csv
from flash_vla.runtime.engine import segments

from .latency import parse_options
from .metrics import require_cuda
from .targets import PLAN_NAMES, build, resolve


def record_invocations(engine, segment: str) -> dict[str, list[tuple[tuple, dict]]]:
    """Every op-table call of one eager run of `segment`: call site -> [(args, kwargs)]."""
    calls: dict[str, list[tuple[tuple, dict]]] = {}

    def recording(name: str, fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return fn(*args, **kwargs)
        return wrapped

    with engine.instrument(recording):
        engine.run_eager(segment)
        torch.cuda.synchronize()
    return calls


def _graph_samples(invoke: Callable[[int], Any], n_inner: int, reps: int, warmup: int = 4) -> list[float]:
    """Per-call ms over `reps` replays of a graph holding `n_inner` invocations."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            invoke(i)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(n_inner):
            invoke(i)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / n_inner)
    return samples


def run(target: str, plan: str | None = None, seed: int = 0, only_segments: list[str] | None = None,
        only_sites: list[str] | None = None, timer: str = "cudagraph", reps: int = 40,
        n_inner: int = 48, repeat_time_ms: int = 100, dry_run_time_ms: int = 25,
        **overrides) -> list[KernelResult]:
    """Time every selected call site of every selected segment in isolation."""
    require_cuda()
    torch.cuda.init()
    engine = build(resolve(target), plan or "shipped", seed=seed, **overrides)
    inputs = engine.sample_inputs(seed)
    engine.forward(**inputs)
    torch.cuda.synchronize()
    costs = engine.costs
    results: list[KernelResult] = []
    for segment in segments(engine):
        if only_segments and segment not in only_segments:
            continue
        per_call = {inv.call_site: inv.cost for inv in costs.get(segment, ())}
        calls = record_invocations(engine, segment)
        order = list(calls)                       # pipeline order of first invocation
        grouped: set[str] = set()
        cases: list[tuple[str, ...]] = []
        for group in engine.atomic_groups:
            members = tuple(s for s in order if s in group)
            if len(members) > 1:
                cases.append(members)
                grouped |= set(members)
        cases += [(s,) for s in order if s not in grouped]
        for members in cases:
            if only_sites and not any(s in only_sites for s in members):
                continue
            if any(len(calls[s]) != len(calls[members[0]]) for s in members):
                raise RuntimeError(f"atomic group {members} recorded unequal invocation "
                                   f"counts: { {s: len(calls[s]) for s in members} }")
            fns = [getattr(engine.ops, s) for s in members]
            invocations = [calls[s] for s in members]
            count = len(invocations[0])

            def invoke(i: int) -> None:
                for fn, calls_of in zip(fns, invocations):
                    args, kwargs = calls_of[i % count]
                    fn(*args, **kwargs)

            flops = sum(per_call[s].flops for s in members if s in per_call) or None
            nbytes = sum(per_call[s].bytes for s in members if s in per_call) or None
            label = f"{segment}/" + "+".join(members)
            if not flops and not nbytes:
                # A site that moves no bytes and does no math at this shape
                # (Pi0's prompt embedding at prompt_len 0) issues no kernel:
                # the CUPTI timer raises on an empty iteration and the other
                # timers would time nothing. Skipped, as the floor model does.
                print(f"{label:24} :: skipped (no device work at this shape)", flush=True)
                continue
            if timer == "cudagraph":
                samples = _graph_samples(invoke, n_inner=min(n_inner, count), reps=reps)
            else:
                samples = bench_gpu_time(
                    invoke, input_args=(0,), enable_cupti=(timer == "cupti"),
                    repeat_time_ms=repeat_time_ms, dry_run_time_ms=dry_run_time_ms)
            results.append(KernelResult(label=label, samples=samples, flops=flops, bytes=nbytes,
                                        identity=engine.identity.as_dict()))
            print(results[-1].perf_line(), flush=True)
    del engine
    torch.cuda.empty_cache()
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks kernels",
                                     description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--plan", default=None,
                        help=f"one of {PLAN_NAMES}, a JSON object or a lab/plans/*.json path")
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--segment", action="append", default=None, help="restrict to a segment")
    parser.add_argument("--site", action="append", default=None, help="restrict to a call site")
    parser.add_argument("--timer", choices=["cudagraph", "cupti", "events"], default="cudagraph")
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--n-inner", type=int, default=48)
    parser.add_argument("--repeat-time-ms", type=int, default=100)
    parser.add_argument("--dry-run-time-ms", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=None, help="append results to a CSV file")
    args = parser.parse_args(argv)
    results = run(args.target, args.plan, seed=args.seed, only_segments=args.segment,
                  only_sites=args.site, timer=args.timer, reps=args.reps, n_inner=args.n_inner,
                  repeat_time_ms=args.repeat_time_ms, dry_run_time_ms=args.dry_run_time_ms,
                  **parse_options(args.option))
    if not results:
        print("no call sites selected", file=sys.stderr)
        return 1
    print()
    print(render_table(results))
    if args.csv:
        write_csv(args.csv, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
