"""T2 ABI mirror of the two in-place residual projections.

The oracle for `llm_backbone_out_proj_residual` and
`llm_backbone_ffn_down_residual`, whose wrappers are cuBLAS `addmm_`. A
library call still gets a mirror: what the mirror pins is not the arithmetic
of a hand-written mainloop but the *contract* -- which operand is the
residual, that the accumulation is in place, and that the result is rounded
once. Those are the parts a wrong wiring gets wrong, and a route swap between
Targets is exactly a wiring change.

Named dims (`models/<model>/spec.py`, `h100/<target>/pipeline.py`):

    ROWS    prefix tokens        968 (Pi0.5) | 768 (Pi0)
    D       backbone width       2048
    F       FFN width            16384

`out_proj_residual` contracts over D, `ffn_down_residual` over F; both write D
columns into the residual stream.

Precision placement mirrors what `addmm_` on bf16 operands actually does: the
products are exact in fp32 (a bf16 product needs 16 mantissa bits), the
accumulation and the `+ C` epilogue are fp32, and the result is rounded to
bf16 **once**, on the store. Adding a rounded product to the residual would be
a different function, and so would rounding twice.

The contraction is written in fp32 for that reason, which requires TF32 to be
off; this module asserts that rather than setting it, so a caller that changed
the global setting is told instead of silently measured.
"""
from __future__ import annotations

import torch


def _require_full_fp32_matmul() -> None:
    """TF32 would make an fp32 contraction here less exact than bf16 with fp32 accumulate."""
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "torch.backends.cuda.matmul.allow_tf32 is True; this reference contracts "
            "in fp32 and TF32 would truncate it to a 10-bit mantissa. Set it to False "
            "(or torch.set_float32_matmul_precision('highest')) in the harness.")


def residual_gemm_reference(x: torch.Tensor, weight: torch.Tensor,
                            out: torch.Tensor) -> torch.Tensor:
    """out += x @ weight, in place, rounded once.

    `x` is (ROWS, K) with K = D for the output projection and K = F for the
    FFN down projection; `weight` is (K, D); `out` is (ROWS, D) and is both
    the residual read and the result written. All bf16.
    """
    _require_full_fp32_matmul()
    rows, k = x.shape                                        # (ROWS, K)
    k_w, d = weight.shape                                    # (K, D)
    assert k == k_w, (x.shape, weight.shape)
    assert out.shape == (rows, d), (out.shape, (rows, d))
    assert x.dtype == weight.dtype == out.dtype == torch.bfloat16

    # fp32 product, fp32 residual, one rounding into the bf16 buffer: the
    # in-place form is the contract, so the residual is read before the write.
    accumulated = out.float() + (x.float() @ weight.float())  # (ROWS, D) fp32
    out.copy_(accumulated)
    return out


__all__ = ["residual_gemm_reference"]
