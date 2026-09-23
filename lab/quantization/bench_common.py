"""Timing and checking shared by the quantization lab scripts.

A GEMM is timed as a CUDA graph of calls that cycle through enough weight
copies to exceed 2x the 96 MiB L2 (weights cold, as in the model) with one warm
activation, after a cosine check against the BF16 matmul it approximates.
Comparisons between variants capture each variant once and replay them in turn
for several rounds, so clock drift hits every variant alike.
"""
import math
import statistics
from typing import Callable

import torch

L2_MIB = 96
Call = Callable[[], object]


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    approx_flat, exact_flat = approx.float().flatten(), exact.float().flatten()
    return float(torch.dot(approx_flat, exact_flat)
                 / (approx_flat.norm() * exact_flat.norm() + 1e-30))


def weight_copies(weight_bytes: int) -> int:
    """Copies of one weight whose total exceeds twice the L2, between 2 and 64."""
    return max(2, min(64, math.ceil(2 * L2_MIB * 2**20 / weight_bytes)))


def cycled(calls: list[Call], copies: int) -> list[Call]:
    """Repeat the per-copy calls so a graph holds about 256 calls."""
    return calls * max(1, 256 // copies)


def capture_graph(calls: list[Call]) -> torch.cuda.CUDAGraph:
    """Run every distinct call once on a side stream (FlashInfer's GEMMs stage
    host scalars on first use), then capture all calls in order."""
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for call in dict.fromkeys(calls):
            call()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def graph_time_us(calls: list[Call], replays: int = 5) -> float:
    """Median µs per call over a CUDA graph that runs every call once."""
    graph = capture_graph(calls)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    per_call_us = []
    for _ in range(replays):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        per_call_us.append(start.elapsed_time(end) * 1000 / len(calls))
    return statistics.median(per_call_us)


def interleaved_times_us(variants: dict[str, list[Call]], units: dict[str, int],
                         rounds: int = 21) -> dict[str, list[float]]:
    """{variant: calls} -> {variant: µs per unit in each round}. Each variant is one
    graph; a round replays every graph once, and `units[variant]` divides its time."""
    graphs = {name: capture_graph(calls) for name, calls in variants.items()}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    per_unit_us: dict[str, list[float]] = {name: [] for name in graphs}
    for _ in range(rounds):
        for name, graph in graphs.items():
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            per_unit_us[name].append(start.elapsed_time(end) * 1000 / units[name])
    return per_unit_us
