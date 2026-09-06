"""T2 ABI mirror of the two pre-norm vision GEMMs.

The oracle for `vision_encoder_norm_qkv` and `vision_encoder_norm_ffn_up`:
same tensors, same buffers, same in-place write into `out`, deliberately
untiled and unfused so a reviewer can read the function it defines in a minute.
Never a performance baseline, and never run under CUDA-graph capture.

Numerics mirror openpi's placement rather than a cleaner one: the contraction
runs on bf16 operands (with the fp32 accumulation torch's matmul performs) and
rounds to bf16, then the tanh-GELU upcasts to fp32 and rounds back. A fused
cuBLASLt or wgmma epilogue rounds once where this rounds twice, so the mirror
is deliberately the looser of the two and the gap is one bf16 ulp -- three
orders below the shallow gate. It exists to catch a wrong layout, bias, norm
or activation, which it does at full size at one layer.
"""
from __future__ import annotations

import torch

from ... import geometry


def _layer_norm(x, norm_w, norm_b):
    """LayerNorm over the feature axis in fp32, rounded back to the input dtype.

    x (M, DIM) bf16, norm_w/norm_b (DIM,) bf16 -> (M, DIM) bf16. The statistics
    are the nonlinear part and upcast; the affine is applied in fp32 with them.
    """
    return torch.nn.functional.layer_norm(
        x.float(), (geometry.DIM,), norm_w.float(), norm_b.float(),
        eps=geometry.NORM_EPS).to(x.dtype)


def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out):
    """LayerNorm then the packed QKV projection, written into `out`.

    x        (VIEWS, TOKENS, DIM)      bf16, read
    norm_w   (DIM,)                    bf16, read
    norm_b   (DIM,)                    bf16, read
    qkv_w    (DIM, QKV_DIM)            bf16, read
    qkv_b    (QKV_DIM,)                bf16, read
    out      (VIEWS, TOKENS, QKV_DIM)  bf16, written in place

    `out` holds Q|K|V blocks of DIM, head-major inside a block.
    """
    views, tokens, dim = x.shape
    assert (tokens, dim) == (geometry.TOKENS, geometry.DIM), x.shape
    assert qkv_w.shape == (geometry.DIM, geometry.QKV_DIM), qkv_w.shape
    rows = views * tokens                                    # M
    x_norm = _layer_norm(x.reshape(rows, dim), norm_w, norm_b)   # (M, DIM)
    projected = torch.nn.functional.linear(x_norm, qkv_w.t(), qkv_b)  # (M, QKV_DIM)
    out.view(rows, geometry.QKV_DIM).copy_(projected)
    return out


def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out):
    """LayerNorm then the GELU feed-forward expansion, written into `out`.

    x        (VIEWS, TOKENS, DIM)  bf16, read
    norm_w   (DIM,)                bf16, read
    norm_b   (DIM,)                bf16, read
    weight   (DIM, FFN)            bf16, read
    bias     (FFN,)                bf16, read
    out      (VIEWS, TOKENS, FFN)  bf16, written in place

    The activation is the tanh approximation (upstream `gelu_pytorch_tanh`,
    which is also what cuBLASLt's GELU epilogue computes); the erf form is a
    different function here, not a rounding difference.
    """
    views, tokens, dim = x.shape
    assert (tokens, dim) == (geometry.TOKENS, geometry.DIM), x.shape
    assert weight.shape == (geometry.DIM, geometry.FFN), weight.shape
    rows = views * tokens                                    # M
    x_norm = _layer_norm(x.reshape(rows, dim), norm_w, norm_b)   # (M, DIM)
    hidden = torch.nn.functional.linear(x_norm, weight.t(), bias)  # (M, FFN)
    activated = torch.nn.functional.gelu(hidden.float(), approximate="tanh").to(hidden.dtype)
    out.view(rows, geometry.FFN).copy_(activated)
    return out


REFERENCES = {
    "vision_encoder_norm_qkv": vision_encoder_norm_qkv,
    "vision_encoder_norm_ffn_up": vision_encoder_norm_ffn_up,
}

__all__ = ["REFERENCES", "vision_encoder_norm_ffn_up", "vision_encoder_norm_qkv"]
