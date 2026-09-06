"""T2 ABI mirror of the fused SigLIP vision attention kernel.

Same tensors, same `out` buffer, same in-place write, deliberately untiled and
unfused. Never a performance baseline, and never run under CUDA-graph capture.

Two things this file exists to catch, neither of which a whole-stage check
would localize:

- the packed-QKV unpacking. `qkv` is (VIEWS, TOKENS, 3*DIM) with Q|K|V blocks
  of DIM and head-major inside a block, so a transposed head axis or a swapped
  K/V block is an easy and silent error; the mirror spells the view out.
- the 72 -> 80 pad. `zero_extend` builds the reference from heads explicitly
  extended with zeros, exactly as the kernel's shared tiles are, so a kernel
  whose pad columns are not zero disagrees here rather than hiding behind a
  tolerance. `zero_extend=False` computes the same function without the pad
  and must agree to the same tolerance -- if the two disagree, the pad is the
  bug.

Numerics mirror the production route: `scaled_dot_product_attention` on bf16
inputs with no explicit scale, which defaults to DH**-0.5, no mask and no
causality. The view axis is a batch axis.
"""
from __future__ import annotations

import torch

from ... import geometry


def _heads(qkv, block: int, views: int):
    """One of Q, K, V as (VIEWS, HEADS, TOKENS, HEAD_DIM).

    qkv (VIEWS, TOKENS, 3*DIM) -> block `block` of three, head-major inside.
    """
    packed = qkv.view(views, geometry.TOKENS, 3, geometry.HEADS, geometry.HEAD_DIM)
    return packed[:, :, block].permute(0, 2, 1, 3)      # (VIEWS, HEADS, TOKENS, DH)


def _zero_extended(x, width: int):
    """(..., HEAD_DIM) -> (..., width), the tail exactly zero."""
    pad = torch.zeros(*x.shape[:-1], width - x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat((x, pad), dim=-1)


def vision_encoder_attention(qkv, out, *, zero_extend: bool = True):
    """Multi-head self-attention over the packed QKV buffer, into `out`.

    qkv  (VIEWS, TOKENS, 3*DIM) bf16, read
    out  (VIEWS, TOKENS, DIM)   bf16, written in place

    With `zero_extend`, Q and K are extended to the kernel's padded width
    before the score contraction. The scale stays DH**-0.5: the pad columns
    contribute nothing to the dot product, so extending them must not change
    the function, and that is the property being checked.
    """
    views = qkv.shape[0]
    assert qkv.shape == (views, geometry.TOKENS, geometry.QKV_DIM), qkv.shape
    assert out.shape == (views, geometry.TOKENS, geometry.DIM), out.shape
    q = _heads(qkv, 0, views)                            # (VIEWS, HEADS, TOKENS, DH)
    k = _heads(qkv, 1, views)
    v = _heads(qkv, 2, views)
    if zero_extend:
        # The kernel contracts over DH_PAD = 80 with the tail held at zero.
        q = _zero_extended(q, 80)
        k = _zero_extended(k, 80)
    attended = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=geometry.HEAD_DIM ** -0.5)        # (VIEWS, HEADS, TOKENS, DH)
    merged = attended.transpose(1, 2).reshape(views, geometry.TOKENS, geometry.DIM)
    out.copy_(merged)
    return out


REFERENCES = {"vision_encoder_attention": vision_encoder_attention}

__all__ = ["REFERENCES", "vision_encoder_attention"]
