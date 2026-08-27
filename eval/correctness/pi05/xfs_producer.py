"""Small semantic and latency experiment for the Pi0.5 XFS producer.

This is deliberately independent of the persistent FFN implementation.  It
checks the producer contract against pure Torch, then times the producer and
the factor-only kernel it replaces.
"""
from __future__ import annotations

import argparse
import statistics

import torch

from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels.xfs import (
    tl_rms_xfs_kmajor,
)


M, M_PAD, K = 50, 64, 1024
EPS = 1e-6


def residual_rms_xfs_reference(
        residual: torch.Tensor,
        projected: torch.Tensor,
        gate: torch.Tensor,
        scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch algorithm from the real residual epilogue through XFS.

    ``projected`` denotes the FP32 output-projection accumulator, before the
    adaptive gate and residual add.  The TileLang XFS kernel starts at ``x``;
    the first line documents the immediately preceding kernel's rounding.
    """
    x = (residual.float()
         + projected.float() * gate.float()[None, :]).bfloat16()
    return x, rms_xfs_reference(x, scale)


def rms_xfs_reference(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exact producer boundary: BF16 [M,K], BF16 [K] -> BF16 [K,64]."""
    if x.dtype != torch.bfloat16 or tuple(x.shape) != (M, K):
        raise ValueError("x must be contiguous BF16 [50,1024]")
    if scale.dtype != torch.bfloat16 or tuple(scale.shape) != (K,):
        raise ValueError("scale must be contiguous BF16 [1024]")
    rstd = torch.rsqrt(x.float().square().mean(dim=1) + EPS).bfloat16()
    normalized = (x * rstd[:, None]).bfloat16()
    xfs_rows = (normalized * scale[None, :]).bfloat16()
    xfs = torch.zeros((K, M_PAD), dtype=torch.bfloat16, device=x.device)
    xfs[:, :M] = xfs_rows.T
    return xfs


def _median_graph_us(body, launches: int, reps: int) -> float:
    for _ in range(5):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            body()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / launches)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--launches", type=int, nargs="+", default=[1, 3, 20])
    args = parser.parse_args()

    gen = torch.Generator(device="cuda").manual_seed(31)
    x = (torch.randn((M, K), generator=gen, device="cuda") * 0.2).bfloat16()
    scale = (1.0 + torch.randn((K,), generator=gen, device="cuda") * 0.1).bfloat16()
    out = torch.empty((K, M_PAD), dtype=torch.bfloat16, device="cuda")
    factor = torch.empty((M,), dtype=torch.bfloat16, device="cuda")
    hidden_ready = torch.empty((32,), dtype=torch.int32, device="cuda")
    down_ready = torch.empty((32,), dtype=torch.int32, device="cuda")

    reference = rms_xfs_reference(x, scale)
    wrappers._rms_factor(x, factor)
    torch.cuda.synchronize()
    factor_reference = torch.zeros_like(reference)
    factor_reference[:, :M] = ((x * factor[:, None]).bfloat16()
                                * scale[None, :]).bfloat16().T
    configs = (
        ("bm8_ok32", dict(BLOCK_M=8, BLOCK_K=256, OUTPUT_K=32,
                          THREADS=128, M_PAD=M_PAD,
                          TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH=False,
                          RESET_READINESS_COUNTERS=True)),
        ("bm8_ok64", dict(BLOCK_M=8, BLOCK_K=256, OUTPUT_K=64,
                          THREADS=128, M_PAD=M_PAD,
                          TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH=False,
                          RESET_READINESS_COUNTERS=True)),
        ("bm4_ok32", dict(BLOCK_M=4, BLOCK_K=256, OUTPUT_K=32,
                          THREADS=128, M_PAD=M_PAD,
                          TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH=False,
                          RESET_READINESS_COUNTERS=True)),
        ("bm16_ok64", dict(BLOCK_M=16, BLOCK_K=256, OUTPUT_K=64,
                           THREADS=128, M_PAD=M_PAD,
                           TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH=False,
                           RESET_READINESS_COUNTERS=True)),
    )
    for name, config in configs:
        producer = wrappers._compiled(tl_rms_xfs_kmajor, M=M, K=K, **config)
        hidden_ready.fill_(7)
        down_ready.fill_(9)
        producer(x, scale, hidden_ready, down_ready, out)
        torch.cuda.synchronize()
        if hidden_ready.count_nonzero() or down_ready.count_nonzero():
            raise SystemExit(f"{name}: readiness counters were not reset")
        exact = torch.equal(out, reference)
        max_abs = (out.float() - reference.float()).abs().max().item()
        max_abs_old_path = (
            out.float() - factor_reference.float()).abs().max().item()
        pad_nonzero = torch.count_nonzero(out[:, M:]).item()
        print(f"[xfs] {name} exact={exact} max_abs={max_abs:.6g} "
              f"old_path_max_abs={max_abs_old_path:.6g} "
              f"pad_nonzero={pad_nonzero}")
        if pad_nonzero != 0:
            raise SystemExit(f"{name}: XFS padding rows are not zero")
        if max_abs > 1.0 / 64.0:
            raise SystemExit(f"{name}: XFS semantic mismatch: max_abs={max_abs}")
        if max_abs_old_path != 0.0:
            raise SystemExit(
                f"{name}: differs from factor path: max_abs={max_abs_old_path}")
        for launches in args.launches:
            producer_us = _median_graph_us(
                lambda: producer(x, scale, hidden_ready, down_ready, out),
                launches, args.reps)
            print(f"[xfs] {name} launches={launches} producer={producer_us:.3f} us")

    for launches in args.launches:
        factor_us = _median_graph_us(
            lambda: wrappers._rms_factor(x, factor), launches, args.reps)
        print(f"[xfs] factor_only launches={launches} time={factor_us:.3f} us")


if __name__ == "__main__":
    main()
