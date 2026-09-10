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

That second regime and `capture` live in `flash_vla.runtime.cuda.timing`, not
here: the backend autotuner needs them and must not import a benchmark harness
to get them. They are re-exported below so existing call sites keep working.
"""
from __future__ import annotations

import statistics
import time
import sys
from importlib import metadata
from typing import Any, Callable

import torch

from flash_vla.runtime.cuda import capture, graph_time_cold


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


def env_block(device=None) -> dict[str, Any]:
    """GPU / toolchain versions, for stamping result files."""
    try:
        tilelang_version = metadata.version("tilelang")
    except metadata.PackageNotFoundError:
        tilelang_version = "unavailable"

    return {
        "gpu": torch.cuda.get_device_name(device),
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tilelang": tilelang_version,
    }


def require_cuda() -> None:
    """Fail early when CUDA is unavailable in the active runtime."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with a visible GPU and a compatible PyTorch/CUDA environment")


__all__ = ["bench_ms", "capture", "diff_stats", "env_block", "graph_time_cold", "percentile",
           "require_cuda"]


def report_context(engine, environment):
    """Stamp measurement provenance separately from architecture identity."""
    from flash_vla.runtime.identity import MeasurementContext
    return MeasurementContext(
        weights=engine.measurement_context["weights"],
        fixture=engine.measurement_context["fixture"],
        environment={
            "gpu_sku": environment.get("gpu"), "driver": environment.get("driver"),
            "cuda_runtime": environment.get("torch_cuda"), "pytorch": environment.get("torch"),
            "tilelang": environment.get("tilelang"), "clock_policy": environment.get("clock_policy"),
            "power_policy": environment.get("power_policy"), "capture_regime": "cuda_graph",
            "clock_observation": environment.get("clock_observation"),
        },
        hostname=environment.get("node"), slurm_job_id=environment.get("job"),
        timestamp=time.time(),
        reference_provenance=engine.measurement_context.get("reference_provenance", {}),
    ).as_dict()
