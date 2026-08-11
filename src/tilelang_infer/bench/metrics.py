"""Timing and comparison primitives shared by the benchmarks.

Two timing regimes, and the choice matters more than it looks:

`bench_ms` times one eager call with CUDA events. Below roughly 15 us/call the
~15 us launch overhead dominates and the measurement says nothing about the
kernel. Use it for whole-pipeline calls only.

`graph_time_cold` captures `n_inner` calls in a CUDA graph, replays, and divides.
Launch overhead is amortised away, so it resolves single-digit-microsecond
kernels -- the regime every decoder kernel lives in. It also cycles the caller's
inputs so weight-heavy kernels read cold HBM instead of a warm L2, which is how
they actually run inside the per-layer loop.
"""
from __future__ import annotations

import statistics
import sys
from typing import Any, Callable

import torch


def percentile(values: list[float], q: float) -> float:
    """q-quantile by nearest rank, q in [0, 1]."""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def bench_ms(fn: Callable[[], Any], warmup: int = 10, iterations: int = 100) -> dict[str, float]:
    """Eager per-call wall time in ms. Meaningless below ~15 us/call; use graph_time_cold there."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "p90": percentile(samples, 0.90),
        "p99": percentile(samples, 0.99),
        "max": max(samples),
    }


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


def diff_stats(ref: torch.Tensor, got: torch.Tensor) -> dict[str, Any]:
    """Absolute/relative deviation of `got` from `ref`, both cast to fp32."""
    ref32 = ref.float()
    got32 = got.float()
    abs_diff = (ref32 - got32).abs()
    rel_diff = abs_diff / ref32.abs().clamp_min(1e-6)
    return {
        "allclose_1e-1": bool(torch.allclose(ref32, got32, atol=1e-1, rtol=1e-1)),
        "allclose_5e-1": bool(torch.allclose(ref32, got32, atol=5e-1, rtol=5e-1)),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "max_rel": float(rel_diff.max().item()),
        "mean_rel": float(rel_diff.mean().item()),
        "num_gt_1e-1": int((abs_diff > 1e-1).sum().item()),
        "numel": int(ref32.numel()),
    }


def env_block() -> dict[str, Any]:
    """GPU / toolchain versions, for stamping result files."""
    import tilelang

    return {
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tilelang": getattr(tilelang, "__version__", None),
    }


def require_cuda() -> None:
    """Fail early and loudly off-GPU: on this cluster CUDA only exists inside sbatch."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit via sbatch/run_gpu.sbatch")
