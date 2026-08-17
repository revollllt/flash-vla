"""CUDA-graph capture and in-graph timing.

These are graph mechanisms, not benchmark policy, which is why they sit beside
`ScratchPool` rather than in the benchmark harness: the backend autotuner needs
them to measure a kernel at all, and a production package must not import a test
harness to do it. `benchmarks.metrics` re-exports both.

The distinction that makes them necessary: eager timing cannot resolve these
kernels. At Pi0's decoder shapes the ~15 us launch overhead is several times the
kernel itself, so an eager measurement reports launch cost and hides a 3 us
difference between two tile configs. Capturing `n_inner` calls into a graph and
dividing amortises that away, which is the only regime in which a config sweep
means anything -- three production configs were wrong in ways an eager benchmark
physically could not see.

`graph_time_cold` also cycles the caller's input sets so weight-heavy kernels
read cold HBM rather than a warm L2, matching how they run inside the per-layer
loop where every layer's weight is a first touch.
"""
from __future__ import annotations

import statistics
from typing import Any, Callable

import torch


def capture(run: Callable[[], Any], warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Warm up `run` on a side stream, then capture one call into a CUDA graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()
    return graph


def graph_time_cold(invoke: Callable[[int], Any], n_inner: int = 48, reps: int = 40,
                    warmup: int = 4) -> float:
    """Median us per call, launch overhead removed.

    `invoke(i)` must issue exactly one kernel launch against the i-th input set;
    cycling i over distinct weights keeps the reads cold.
    """
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
        samples.append(start.elapsed_time(end) / n_inner * 1000)
    return statistics.median(samples)


__all__ = ["capture", "graph_time_cold"]
