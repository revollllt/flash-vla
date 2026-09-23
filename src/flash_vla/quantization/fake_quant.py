"""Fake quantization: a block-scaled GEMM's arithmetic in PyTorch.

A block-scaled GEMM multiplies exact values: E4M3 or E2M1 codes times their
UE8M0 or UE4M3 block scales, accumulated in FP32, with NVFP4's two per-tensor
encode scales divided out of the result. Such a value needs at most six
significant bits, so the dequantized operands are exact in BF16, and a BF16
GEMM with FP32 output reproduces the kernel up to FP32 summation order. This is
the reference a recipe's kernels are held to, and what its quality is measured
with before they exist.

`fake_quantize` quantizes with `reference`, whose exact reciprocals move an NVFP4
value by one E2M1 step, on under 1% of values, relative to the kernels'
approximate ones. Where the kernels are available, quantize with them and pass
their bytes to `dequantized_operand`: the product then equals FlashInfer's b12x
GEMM up to summation order (lab/quantization/check_quant_ops.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from . import reference

BlockFormat = Literal["mxfp8", "nvfp4"]


@dataclass(frozen=True)
class FakeQuantized:
    """An operand as a block-scaled GEMM sees it: [rows, K], blocks along K."""
    values: torch.Tensor         # bf16: codes times block scales, exact
    decode_scale: torch.Tensor   # float32 scalar: 1 / NVFP4 encode scale, 1 for MXFP8


def dynamic_encode_scale(rows: torch.Tensor) -> torch.Tensor:
    """NVFP4's per-tensor encode scale 448 * 6 / amax(rows), float32 (1,)."""
    # An all-zero tensor would make the scale infinite.
    return ((448 * 6) / rows.float().abs().amax().clamp(min=torch.finfo(torch.float32).tiny)
            ).reshape(1)


def dequantized_operand(values: torch.Tensor, scale_codes: torch.Tensor, fmt: BlockFormat,
                        encode_scale: torch.Tensor | None) -> FakeQuantized:
    """Quantized bytes (E4M3 [M, K] or packed E2M1 [M, K/2], row-major scale codes
    [M, K/block]) as the GEMM operand; `encode_scale` is NVFP4's, None for MXFP8."""
    if fmt == "mxfp8":
        return FakeQuantized(reference.dequantize_mxfp8(values, scale_codes).to(torch.bfloat16),
                             torch.ones((), dtype=torch.float32, device=values.device))
    if fmt != "nvfp4":
        raise ValueError(f"unknown block format {fmt!r}")
    return FakeQuantized(reference.dequantize_nvfp4(values, scale_codes, 1.0).to(torch.bfloat16),
                         (1.0 / encode_scale).reshape(()).float())


def fake_quantize(rows: torch.Tensor, fmt: BlockFormat,
                  encode_scale: torch.Tensor | None = None) -> FakeQuantized:
    """bf16 [M, K] quantized by `reference`, blocks along K. NVFP4 without
    `encode_scale` uses the dynamic per-tensor scale."""
    if fmt == "mxfp8":
        return dequantized_operand(*reference.quantize_mxfp8(rows), fmt, None)
    scale = encode_scale if encode_scale is not None else dynamic_encode_scale(rows)
    return dequantized_operand(*reference.quantize_nvfp4(rows, scale), fmt, scale)


def fake_quant_matmul(activation: FakeQuantized, weight: FakeQuantized) -> torch.Tensor:
    """activation [M, K] times weight stored as [N, K]: FP32 [M, N], as the kernel
    accumulates it before its output rounding."""
    return (torch.mm(activation.values, weight.values.t(), out_dtype=torch.float32)
            * (activation.decode_scale * weight.decode_scale))
