"""T2 ABI mirror of the fused prefix attention kernel.

The oracle, not a baseline: it defines what `kernels/enc_attn.cu` must
compute, over the same tensors, in the same buffers, with the same in-place
mutation, deliberately untiled. One contraction per call, no cleverness. It
never runs under CUDA-graph capture, so the shape asserts cost nothing.

Named dims, from the Target's configuration and the Gemma backbone's fixed
head geometry (`models/<model>/spec.py`):

    ROWS    prefix tokens                        968 (Pi0.5) | 768 (Pi0)
    H       query heads                          8
    DH      head dim                             256
    KEYS    keys attended                        ROWS
    M_Q     flattened query rows = ROWS * H      7744 | 6144

Multi-query: one K/V head serves all eight query heads, so the op is
single-head attention over M_Q flattened query rows against KEYS keys.

## Where the roundings are, and why they are not the obvious ones

The general rule for a reference is that a linear contraction runs on
storage-dtype inputs and lets torch accumulate in fp32, while a nonlinearity
upcasts and rounds back at its output. Inside a fused attention kernel the
score matrix is not an output, so applying that rule literally would describe
a different function from the one the kernel computes. The kernel's roundings
are:

- `S = Q K^T` lands in a **wgmma fp32 accumulator and is never rounded**; the
  scale and the additive mask are applied to it in fp32, and the online
  softmax runs there too. So this mirror contracts in fp32.
- `P` **is** rounded to bf16, because it is the A operand of the P.V wgmma.
  That rounding is real and this mirror performs it.
- `O` accumulates in fp32 and is rounded once on the store into `out`.

The reference *route* (the torch chain the in-engine check compares against)
differs here: `torch.matmul` on bf16 operands returns bf16, so it rounds the
scores before the softmax. That is a legitimate difference between two
implementations of the same op and it is why the in-engine comparison sits
looser than this mirror's.

fp32 contractions require TF32 to be off, or the mantissa drops to 10 bits and
the oracle becomes less accurate than the kernel it judges. This module
asserts that rather than setting it, so a caller that changed the global
setting is told instead of silently measured.
"""
from __future__ import annotations

import torch


def _require_full_fp32_matmul() -> None:
    """TF32 would make an fp32 contraction here less exact than bf16 with fp32 accumulate."""
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "torch.backends.cuda.matmul.allow_tf32 is True; the attention reference "
            "contracts in fp32 and TF32 would truncate it to a 10-bit mantissa. "
            "Set it to False (or torch.set_float32_matmul_precision('highest')) "
            "in the harness before comparing.")


def attention_reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        scale: float, mask: torch.Tensor | None,
                        out: torch.Tensor) -> torch.Tensor:
    """out[:] = softmax(Q @ K.T * scale + mask) @ V, written in place.

    Q is (M_Q, DH) with row = token * H + head; K and V are (KEYS, DH) leading
    -row views of the layer's KV cache; `mask` is (KEYS,) additive bf16 or
    `None` (a Target whose prefix has no padding); `out` is (M_Q, DH) and is
    fully written. Every tensor is bf16 and on the same device.
    """
    _require_full_fp32_matmul()
    m_q, dh = Q.shape                                        # (M_Q, DH)
    keys, dh_k = K.shape                                     # (KEYS, DH)
    assert dh == dh_k == V.shape[1], (Q.shape, K.shape, V.shape)
    assert V.shape[0] == keys, (K.shape, V.shape)
    assert out.shape == Q.shape, (out.shape, Q.shape)
    assert Q.dtype == K.dtype == V.dtype == out.dtype == torch.bfloat16
    assert mask is None or mask.shape == (keys,), None if mask is None else mask.shape

    # fp32 accumulator, never rounded: the scale and the mask meet it there.
    scores = (Q.float() @ K.float().transpose(0, 1)) * scale  # (M_Q, KEYS) fp32
    if mask is not None:
        # Additive and broadcast over query rows: the mask is a property of
        # the key, which is what lets one vector serve every query row.
        scores = scores + mask.float()[None, :]              # (M_Q, KEYS)
    # P is rounded because it is the A operand of the second wgmma.
    probs = torch.softmax(scores, dim=-1).to(torch.bfloat16)  # (M_Q, KEYS) bf16
    out.copy_((probs.float() @ V.float()))                   # (M_Q, DH) -> bf16
    return out


def attention_fp32_reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                             scale: float, mask: torch.Tensor | None) -> torch.Tensor:
    """The same function with no intermediate rounding at all, returned as fp32.

    Reported beside the bf16 mirror, never gated against: it bounds how much
    of a measured difference is the kernel and how much is bf16 rounding that
    both implementations are entitled to. Rounding the result would throw away
    exactly the information it exists to carry.
    """
    _require_full_fp32_matmul()
    scores = (Q.float() @ K.float().transpose(0, 1)) * scale
    if mask is not None:
        scores = scores + mask.float()[None, :]
    return torch.softmax(scores, dim=-1) @ V.float()


__all__ = ["attention_fp32_reference", "attention_reference"]
