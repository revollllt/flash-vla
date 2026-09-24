"""CUDA-graph capture and in-graph timing.

These are graph mechanisms, not benchmark policy, which is why they sit in the
runtime rather than in the measurement layer: the backend autotuner needs
them to measure a kernel at all, and a production package must not import a
harness to do it. `benchmarks/kernels.py` and the lab probes time call sites
with the same `graph_samples`.

The distinction that makes them necessary: eager timing cannot resolve these
kernels. At Pi0's decoder shapes the ~15 us launch overhead is several times the
kernel itself, so an eager measurement reports launch cost and hides a 3 us
difference between two tile configs. Capturing `n_inner` calls into a graph and
dividing amortises that away, which is the only regime in which a config sweep
means anything -- three production configs were wrong in ways an eager benchmark
physically could not see.

`graph_samples` also cycles the caller's input sets so weight-heavy kernels
read cold HBM rather than a warm L2, matching how they run inside the per-layer
loop where every layer's weight is a first touch.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from .graph import StreamGraph


def capture(run: Callable[[], Any], warmup: int = 3) -> StreamGraph:
    """Warm up `run` on a side stream, then capture one call into a CUDA graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = StreamGraph()
    with graph.capture():
        run()
    torch.cuda.synchronize()
    return graph


def graph_samples(invoke: Callable[[int], object], *, n_inner: int, reps: int,
                  warmup: int = 4) -> list[float]:
    """Milliseconds per call over `reps` replays of one graph holding `n_inner`
    calls, launch overhead amortised away.

    `invoke(i)` issues the launches of the i-th call against the i-th input
    set, on the current device and stream, and must be capturable; cycling i
    over distinct weights keeps the reads cold. The warmup runs on a side
    stream and every replay is synchronized, so this must not be called while
    another capture is active.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            invoke(i)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = StreamGraph()
    with graph.capture():
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


__all__ = ["capture", "graph_samples"]
