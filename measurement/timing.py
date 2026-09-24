"""Timed loops and their statistics, shared by every measuring harness.

`wall_samples` times a call plus the synchronize that makes it observable,
the latency a caller sees; `event_samples` times the device work between two
CUDA events. Both warm up, then take one sample per repetition, and fill an
optional `LoopTrace` (`measurement.attribution`) outside the timed region.
`summarize` reduces samples to the statistics a report carries. In-graph
per-call timing of kernels is `flash_vla.runtime.cuda.graph_samples`, which
the backend autotuner also uses.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, TypedDict

import torch

from .attribution import LoopTrace


class RequiredStats(TypedDict):
    min: float
    median: float
    n: int
    samples_ms: list[float]
    p99: float | None


class SampleStats(RequiredStats, total=False):
    """A report's statistics of one timed loop; `p99_note` says why `p99` is None."""
    p99_note: str


def summarize(samples: list[float], p99_min_reps: int) -> SampleStats:
    """`min`, `median` and, from `p99_min_reps` samples on, `p99` of `samples`, which are kept."""
    ordered = sorted(samples)
    n = len(ordered)
    out: SampleStats = {"min": ordered[0], "median": statistics.median(ordered), "n": n,
                        "samples_ms": samples, "p99": None}
    if n >= p99_min_reps:
        out["p99"] = ordered[min(n - 1, int(round(0.99 * (n - 1))))]
        return out
    out["p99_note"] = f"insufficient: {n} < {p99_min_reps} repetitions"
    return out


def wall_samples(call: Callable[[], object], reps: int, warmup: int,
                 trace: LoopTrace | None = None) -> list[float]:
    """Wall-clock milliseconds of `call` plus a synchronize, `reps` times.

    A `trace` is filled outside the timed region only: the repetition's start
    is the same `perf_counter` reading the sample is computed from, and the
    process counters are read after the synchronize has returned.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
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


def event_samples(call: Callable[[], object], reps: int, warmup: int,
                  trace: LoopTrace | None = None) -> list[float]:
    """CUDA-event milliseconds of `call`, `reps` times: device time between the
    events recorded around it, synchronized after each. A `trace` is filled as
    in `wall_samples`."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
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


__all__ = ["SampleStats", "event_samples", "summarize", "wall_samples"]
