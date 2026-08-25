"""Per-kernel GPU timing, ported from FlashInfer's ``bench_gpu_time``.

Model- and backend-agnostic on purpose: any callable that issues one kernel
launch can be measured, so a newly written kernel can be benchmarked with the
same methodology as the shipped ones, before it is part of anything.

Public surface:
- ``bench_gpu_time``            unified timing entry (CUPTI / CUDA graph / events)
- ``KernelResult`` / ``render_table`` / ``write_csv``   result reporting
- ``attention_flops`` / ``attention_tb_per_sec``        shape -> metric helpers

    samples = bench_gpu_time(my_kernel, input_args=(a, b, out))
    print(KernelResult("my_kernel", samples, flops=..., bytes=...).perf_line())

The built-in Pi0 cases and their CLI are not here -- they depend on the
benchmark buffers, so they live at ``benchmarks/kernels.py``
(``python -m benchmarks kernels``). See the ``benchmark-kernel`` skill.
"""
from .timer import (
    bench_gpu_time,
    bench_gpu_time_with_cuda_event,
    bench_gpu_time_with_cudagraph,
    bench_gpu_time_with_cupti,
    calculate_rotation_count,
    get_l2_cache_size,
)
from .metrics import (
    KernelResult,
    attention_flops,
    attention_tb_per_sec,
    render_table,
    tb_per_sec,
    tflops_per_sec,
    write_csv,
)

__all__ = [
    "bench_gpu_time",
    "bench_gpu_time_with_cuda_event",
    "bench_gpu_time_with_cudagraph",
    "bench_gpu_time_with_cupti",
    "calculate_rotation_count",
    "get_l2_cache_size",
    "KernelResult",
    "attention_flops",
    "attention_tb_per_sec",
    "render_table",
    "tb_per_sec",
    "tflops_per_sec",
    "write_csv",
]
