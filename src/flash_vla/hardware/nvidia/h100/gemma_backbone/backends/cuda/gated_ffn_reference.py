"""T2 ABI mirror of the backbone's RMSNorm + gated feed-forward call site.

The oracle for `llm_backbone_norm_gated_ffn`, the single largest kernel in
either Target: 56.6 % of the Pi0.5 backbone's in-graph kernel time (job
599808). It defines what the kernel must compute over the same tensors, in the
same buffers, deliberately untiled.

Named dims (`models/<model>/spec.py`, `h100/<target>/pipeline.py`):

    ROWS    prefix tokens     968 (Pi0.5) | 768 (Pi0)
    D       backbone width    2048
    F       FFN width         16384

## The three roundings, and why each one is a contract

The shipped TileLang route is two kernels and this mirror reproduces both,
because a fused replacement must land on the same function, not a cleaner one.

1. **RMSNorm is `x * rsqrt(mean_D(x^2) + 1e-6)` with no learned weight** (it is
   folded into the projection offline), accumulated in fp32 and **rounded to
   bf16 before the projection**. That rounding is load-bearing: the owning
   Agent Note records that the backbone normalizes to bf16 ahead of the GEMM
   rather than folding a per-row factor into the A operand, and that the two
   orders are not interchangeable against the upstream implementation. A fused
   kernel that keeps the normalized activation in fp32 computes something
   else.
2. **Both projections accumulate in fp32 and are never rounded between the
   GEMM and the activation.** They are one fused epilogue on two live
   accumulators.
3. **GELU-tanh, then the product, then one rounding to bf16.** The activation
   is spelled in its sigmoid form, `v * sigmoid(C0 * v * (1 + C1 * v^2))` with
   `C0 = 2 * sqrt(2/pi)`; that is the algebraic identity of the tanh
   approximation and it is how the shipped kernel spells it, so the mirror
   spells it the same way rather than calling `F.gelu(..., approximate="tanh")`
   and inheriting a different rounding path.

The fp32 contractions require TF32 to be off, or the mirror's mantissa drops
below the bf16-with-fp32-accumulate it is judging. This module asserts that
rather than setting it, so a caller that changed the global setting is told.
"""
from __future__ import annotations

import torch

#: 2 * sqrt(2/pi): the tanh-GELU constant in the sigmoid form the kernel uses.
GELU_C0 = 1.5957691216057308
GELU_C1 = 0.044715
#: The backbone's RMSNorm epsilon; inside the mean, not outside the sqrt.
RMS_EPS = 1e-6


def _require_full_fp32_matmul() -> None:
    """TF32 would make an fp32 contraction here less exact than bf16 with fp32 accumulate."""
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "torch.backends.cuda.matmul.allow_tf32 is True; this reference contracts "
            "in fp32 and TF32 would truncate it to a 10-bit mantissa. Set it to False "
            "(or torch.set_float32_matmul_precision('highest')) in the harness.")


def rms_norm_reference(x: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
    """x_norm[:] = x * rsqrt(mean_D(x^2) + 1e-6), fp32 statistics, bf16 out.

    `x` and `x_norm` are (ROWS, D) bf16. Written in place; `x_norm` is the
    graph's own buffer, which the op spec declares as an auxiliary output.
    """
    rows, d = x.shape                                        # (ROWS, D)
    assert x_norm.shape == x.shape, (x_norm.shape, x.shape)
    assert x.dtype == x_norm.dtype == torch.bfloat16
    xf = x.float()                                           # (ROWS, D) fp32
    factor = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)   # (ROWS, 1)
    x_norm.copy_(xf * factor)                                # rounds to bf16
    return x_norm


def gelu_tanh(v: torch.Tensor) -> torch.Tensor:
    """The activation, in the sigmoid form the shipped kernel spells; fp32 in, fp32 out."""
    return v * torch.sigmoid(GELU_C0 * v * (1.0 + GELU_C1 * v * v))


def norm_gated_ffn_reference(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor,
                             out: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
    """out[:] = gelu_tanh(x_norm @ gate_w) * (x_norm @ up_w), with x_norm written too.

    `x` is (ROWS, D); `gate_w` and `up_w` are (D, F); `out` is (ROWS, F) and is
    fully written; `x_norm` is (ROWS, D) and receives the normalized
    activation. Every tensor is bf16 on one device. The two (ROWS, F) branch
    outputs are named locals here and are exactly what a fused kernel must not
    materialize -- 63 MB of round trip at the Pi0.5 shape.
    """
    _require_full_fp32_matmul()
    rows, d = x.shape                                        # (ROWS, D)
    d_g, f = gate_w.shape                                    # (D, F)
    assert gate_w.shape == up_w.shape, (gate_w.shape, up_w.shape)
    assert d == d_g, (x.shape, gate_w.shape)
    assert out.shape == (rows, f), (out.shape, (rows, f))
    assert x.dtype == gate_w.dtype == up_w.dtype == out.dtype == torch.bfloat16

    rms_norm_reference(x, x_norm)                            # (ROWS, D) bf16
    a = x_norm.float()                                       # the bf16 values, widened
    gate = a @ gate_w.float()                                # (ROWS, F) fp32, never rounded
    up = a @ up_w.float()                                    # (ROWS, F) fp32, never rounded
    out.copy_(gelu_tanh(gate) * up)                          # one rounding, on the store
    return out


__all__ = ["GELU_C0", "GELU_C1", "RMS_EPS", "gelu_tanh", "norm_gated_ffn_reference",
           "rms_norm_reference"]
