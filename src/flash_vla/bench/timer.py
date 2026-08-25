"""Portable GPU kernel timing with CUPTI / CUDA events / CUDA graphs.

This module ports the timing methodology of FlashInfer's ``bench_gpu_time``
(``flashinfer/testing/utils.py``) into flash-vla, so a single kernel can be
measured the way the production baselines are measured: hardware-level GPU time
when CUPTI is available, CUDA-graph-amortised or event timing otherwise, always
against a cold L2.

Why three backends instead of just reusing ``flash_vla.runtime.cuda.graph_time_cold``:

- CUPTI measures pure GPU kernel execution time from hardware timestamps,
  excluding the CPU-side launch overhead entirely. For a 4 us decoder kernel a
  CUDA event measurement is dominated by launch cost no matter how it is
  amortised; CUPTI sees the kernel alone.
- CUDA events measure launch + execution. That is the right quantity for
  whole-pipeline wall time but wrong for per-kernel attribution.
- CUDA graphs amortise launch overhead by replaying many calls per graph, which
  is what ``graph_time_cold`` already does for the tuning loop. It is the
  fallback when CUPTI is not installed (e.g. CUDA < 13).

Cold-L2 strategy is picked per backend, mirroring FlashInfer:

- CUPTI / CUDA events: allocate a ``2x L2`` buffer and ``zero_()`` it between
  iterations, so every measured call reads its inputs from HBM.
- CUDA graphs: rotating buffer copies are used (see ``rotating_copies``),
  because ``zero_()`` cannot run inside a captured graph cheaply.

Iteration counts are adaptive, not hard-coded: the kernel is timed 5 times to
estimate its cost, then warmup/measured counts are derived from target durations
(``dry_run_time_ms`` / ``repeat_time_ms``), so a 3 us kernel and a 300 us kernel
both get statistically meaningful samples without hand-tuning.
"""
from __future__ import annotations

import math
import statistics
import warnings
from functools import partial as _partial
from typing import Any, Callable, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# L2 helpers (ported from FlashInfer)
# ---------------------------------------------------------------------------


def get_l2_cache_size(device=None) -> int:
    """L2 cache size in bytes for the given CUDA device."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).L2_cache_size


def _extract_gpu_tensors(obj) -> list[torch.Tensor]:
    """Recursively collect all CUDA-resident tensors in a nested structure."""
    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_extract_gpu_tensors(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            tensors.extend(_extract_gpu_tensors(v))
    return tensors


def _calculate_tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            total += t.numel() * t.element_size()
    return total


def _clone_structure(obj):
    """Deep clone a nested structure, cloning GPU tensors, preserving others.

    Non-contiguous tensors are cloned via ``empty_strided`` so stride patterns
    are preserved.
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            if obj.is_contiguous():
                return obj.detach().clone()
            result = torch.empty_strided(
                obj.size(), obj.stride(), dtype=obj.dtype, device=obj.device
            )
            result.copy_(obj.detach())
            return result
        return obj
    if isinstance(obj, list):
        return [_clone_structure(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_structure(item) for item in obj)
    if isinstance(obj, dict):
        return {k: _clone_structure(v) for k, v in obj.items()}
    return obj


def calculate_rotation_count(
    tensors: Sequence[torch.Tensor], device=None, min_rotations: int = 2
) -> int:
    """Number of buffer copies needed to keep reads cold inside a CUDA graph.

    Conservative thresholds: skip rotation only when the working set is >= 5x
    L2 (cache effects truly negligible); otherwise ensure enough distinct
    buffers that any two uses of one buffer are separated by more than the
    safe-cache threshold of traffic.
    """
    l2_size = get_l2_cache_size(device)
    total_bytes = _calculate_tensor_bytes(tensors)
    if total_bytes == 0:
        return 1
    safe_cache_threshold = l2_size * 5
    if total_bytes >= safe_cache_threshold:
        return 1
    num_rotations = math.ceil(safe_cache_threshold / total_bytes) + 1
    return max(min_rotations, num_rotations)


def _infer_device_from_tensors(input_args, input_kwargs, default="cuda"):
    gpu_tensors = _extract_gpu_tensors(input_args) + _extract_gpu_tensors(input_kwargs)
    if gpu_tensors:
        return gpu_tensors[0].device
    return default


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _iter_counts(est_ms: float, dry_run_time_ms: int, repeat_time_ms: int) -> tuple[int, int]:
    """Derive warmup/measured iteration counts from an estimated per-call ms."""
    dry = max(1, int(dry_run_time_ms / est_ms))
    repeat = max(1, int(repeat_time_ms / est_ms))
    return dry, repeat


def _call_fn(fn: Callable, input_args, input_kwargs):
    if input_args or input_kwargs:
        fn(*input_args, **input_kwargs)
    else:
        fn()


def _estimate_ms(fn: Callable, input_args, input_kwargs, flush_buffer, aggregate=max):
    """Time ``fn`` 5 times to estimate per-call ms (used to size the run)."""
    torch.cuda.synchronize()
    _call_fn(fn, input_args, input_kwargs)  # exclude one-time init
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        if flush_buffer is not None:
            flush_buffer.zero_()
        _call_fn(fn, input_args, input_kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 5


def bench_gpu_time_with_cuda_event(
    fn,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    input_args: tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
) -> list[float]:
    """CUDA-event timing: measures launch + execution. No CUDA graphs.

    Each iteration is timed individually with CUDA events and, when
    ``cold_l2_cache`` is set, preceded by a ``2x L2`` buffer zero so the kernel
    reads cold HBM. Per-iteration ms returned.
    """
    if input_kwargs is None:
        input_kwargs = {}

    flush_buffer = None
    if cold_l2_cache:
        device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")
        l2 = get_l2_cache_size(device)
        flush_buffer = torch.empty((l2 * 2) // 1, device=device, dtype=torch.int8)

    est = _estimate_ms(fn, input_args, input_kwargs, flush_buffer)
    if dry_run_iters is None:
        dry_run_iters, repeat_iters = _iter_counts(est, dry_run_time_ms, repeat_time_ms)

    for _ in range(dry_run_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        _call_fn(fn, input_args, input_kwargs)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeat_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _call_fn(fn, input_args, input_kwargs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _rotating_copies(input_args, input_kwargs, device) -> list[tuple[tuple, dict]]:
    """Build rotating buffer copies for cold-L2 inside a CUDA graph."""
    tensors = _extract_gpu_tensors(input_args) + _extract_gpu_tensors(input_kwargs)
    n = calculate_rotation_count(tensors, device)
    copies = [(input_args, input_kwargs)]
    for _ in range(n - 1):
        copies.append((_clone_structure(input_args), _clone_structure(input_kwargs)))
    return copies


def bench_gpu_time_with_cudagraph(
    fn,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    num_iters_within_graph: int = 10,
    input_args: tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
) -> list[float]:
    """CUDA-graph timing: amortised launch overhead, cold L2 via rotating buffers.

    ``num_iters_within_graph`` kernel calls are captured into one graph, replay
    time is divided, so launch overhead per call is amortised away. When
    ``cold_l2_cache`` is set, GPU tensors in ``input_args``/``input_kwargs`` are
    cloned into enough buffers to exceed L2 (see ``calculate_rotation_count``)
    and the graph cycles through them, matching how kernels read cold weights
    inside the per-layer loop.
    """
    if input_kwargs is None:
        input_kwargs = {}
    device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")

    def call(i: int):
        args, kwargs = copies[i % len(copies)]
        fn(*args, **kwargs)

    copies = [(input_args, input_kwargs)]
    if cold_l2_cache:
        copies = _rotating_copies(input_args, input_kwargs, device)

    # Warm up on a side stream, then capture one graph of num_iters_within_graph calls.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(3):
            call(i)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(num_iters_within_graph):
            call(i)
    torch.cuda.synchronize()

    # Estimate per-call ms for iteration sizing.
    est_samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        est_samples.append(start.elapsed_time(end) / num_iters_within_graph)
    est = statistics.median(est_samples)

    if dry_run_iters is None:
        dry_run_iters, repeat_iters = _iter_counts(est, dry_run_time_ms, repeat_time_ms)

    for _ in range(dry_run_iters):
        graph.replay()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeat_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / num_iters_within_graph)
    return samples


def _cupti_available() -> bool:
    """True when cupti-python >= 13 is importable (CUDA 13+)."""
    try:
        from cupti import cupti  # noqa: F401

        from importlib.metadata import version as _version

        v = int(_version("cupti-python").split(".")[0])
        return v >= 13
    except Exception:
        return False


def bench_gpu_time_with_cupti(
    fn,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    use_cuda_graph: bool = False,
    input_args: tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
) -> list[float]:
    """CUPTI hardware-level kernel timing: measures pure GPU execution time.

    Requires ``cupti-python >= 13.0.0`` (CUDA 13+). Per-iteration GPU time is
    derived from CUPTI activity kernel start/end timestamps, excluding the
    CPU-side launch overhead entirely. Falls back to CUDA-event or graph timing
    when CUPTI is unavailable.
    """
    if input_kwargs is None:
        input_kwargs = {}
    if not _cupti_available():
        warnings.warn(
            "CUPTI is not installed. Try 'pip install -U cupti-python'. "
            "Falling back to CUDA events for benchmarking.",
            UserWarning,
            stacklevel=2,
        )
        if use_cuda_graph:
            return bench_gpu_time_with_cudagraph(
                fn=fn,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
                dry_run_time_ms=dry_run_time_ms,
                repeat_time_ms=repeat_time_ms,
                input_args=input_args,
                input_kwargs=input_kwargs,
                cold_l2_cache=cold_l2_cache,
            )
        return bench_gpu_time_with_cuda_event(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=cold_l2_cache,
        )

    from cupti import cupti

    # ---- CUPTI buffer callbacks (mirrors FlashInfer) -----------------------
    def func_buffer_requested():
        return 8 * 1024 * 1024, 0  # buffer_size, max_num_records

    def set_kernel_name(activity):
        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
            return activity.name
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return "MEMCPY"
        if activity.kind == cupti.ActivityKind.MEMSET:
            return "MEMSET"
        return "UNKNOWN"

    def get_copy_kind(activity):
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return activity.copy_kind
        return 0

    def get_bytes(activity):
        if activity.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
            return activity.bytes
        return 0

    def get_value(activity):
        if activity.kind == cupti.ActivityKind.MEMSET:
            return activity.value
        return 0

    def collect_kernel_info(activity):
        return (
            set_kernel_name(activity),
            activity.start,
            activity.end,
            activity.correlation_id,
            get_copy_kind(activity),
            get_bytes(activity),
            get_value(activity),
            activity.kind,
        )

    def func_buffer_completed(launches, kernels, activities):
        for activity in activities:
            if activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                kernels.append(collect_kernel_info(activity))
            elif activity.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append(
                    (
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                        activity.cbid,
                        activity.kind,
                    )
                )

    # ---- cold-L2 flush buffer ---------------------------------------------
    flush_buffer = None
    if cold_l2_cache:
        device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")
        l2 = get_l2_cache_size(device)
        flush_buffer = torch.empty((l2 * 2) // 1, device=device, dtype=torch.int8)

    # ---- optional CUDA graph capture --------------------------------------
    def call_fn():
        _call_fn(fn, input_args, input_kwargs)

    runner = call_fn
    if use_cuda_graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                call_fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call_fn()
        runner = graph.replay

    # ---- estimate + size the run ------------------------------------------
    est = _estimate_ms(runner, (), {}, flush_buffer)
    if dry_run_iters is None:
        dry_run_iters, repeat_iters = _iter_counts(est, dry_run_time_ms, repeat_time_ms)

    for _ in range(dry_run_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        runner()
    torch.cuda.synchronize()

    # ---- CUPTI activity measurement ---------------------------------------
    launches: list[tuple] = []
    kernels: list[tuple] = []
    iter_timestamps: list[tuple[int, int]] = []

    for kind in (
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    ):
        cupti.activity_enable(kind)

    cupti.activity_register_callbacks(
        func_buffer_requested,
        # partial() so the accumulated lists survive across buffer flushes.
        _partial(func_buffer_completed, launches, kernels),
    )

    for _ in range(repeat_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        torch.cuda.synchronize()
        start_cpu = cupti.get_timestamp()
        runner()
        end_cpu = cupti.get_timestamp()
        torch.cuda.synchronize()
        iter_timestamps.append((start_cpu, end_cpu))

    cupti.activity_flush_all(0)
    for kind in (
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    ):
        cupti.activity_disable(kind)
    cupti.finalize()

    # ---- correlate launches to per-iteration kernels ----------------------
    import bisect

    sorted_launches = sorted(launches, key=lambda l: l[0])
    launch_starts = [l[0] for l in sorted_launches]

    corr_id_to_kernels: dict[int, list[tuple]] = {}
    for k in kernels:
        corr_id_to_kernels.setdefault(k[3], []).append(k)

    samples: list[float] = []
    kernel_names: set[str] | None = None
    for start_cpu, end_cpu in iter_timestamps:
        left = bisect.bisect_left(launch_starts, start_cpu)
        right = bisect.bisect_right(launch_starts, end_cpu)
        corr_ids = {sorted_launches[i][2] for i in range(left, right)}
        iter_kernels = []
        for cid in corr_ids:
            iter_kernels.extend(corr_id_to_kernels.get(cid, []))
        if not iter_kernels:
            raise ValueError(f"No kernel activities recorded for an iteration")
        current_names = {f"{k[0]}_{k[4]}_{k[5]}_{k[6]}_{k[7]}" for k in iter_kernels}
        if kernel_names is None:
            kernel_names = current_names
        elif kernel_names != current_names:
            raise ValueError(f"Inconsistent kernel names: {kernel_names} != {current_names}")
        min_start = min(k[1] for k in iter_kernels)
        max_end = max(k[2] for k in iter_kernels)
        samples.append((max_end - min_start) / 1e6)  # ns -> ms
    return samples


def bench_gpu_time(
    fn,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    enable_cupti: bool = True,
    use_cuda_graph: bool = False,
    num_iters_within_graph: int = 10,
    input_args: tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
) -> list[float]:
    """Unified GPU kernel timing entry point.

    Timing backend precedence:
      1. CUPTI (``enable_cupti=True``): hardware-level pure GPU time.
      2. CUDA graphs (``use_cuda_graph=True``): amortised launch overhead,
         rotating-buffer cold L2.
      3. CUDA events (default): launch + execution, L2-flush cold L2.

    Returns per-iteration times in milliseconds.
    """
    if enable_cupti:
        return bench_gpu_time_with_cupti(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            use_cuda_graph=use_cuda_graph,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=cold_l2_cache,
        )
    if use_cuda_graph:
        return bench_gpu_time_with_cudagraph(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            num_iters_within_graph=num_iters_within_graph,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=cold_l2_cache,
        )
    return bench_gpu_time_with_cuda_event(
        fn=fn,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        input_args=input_args,
        input_kwargs=input_kwargs,
        cold_l2_cache=cold_l2_cache,
    )


__all__ = [
    "bench_gpu_time",
    "bench_gpu_time_with_cuda_event",
    "bench_gpu_time_with_cudagraph",
    "bench_gpu_time_with_cupti",
    "calculate_rotation_count",
    "get_l2_cache_size",
]
