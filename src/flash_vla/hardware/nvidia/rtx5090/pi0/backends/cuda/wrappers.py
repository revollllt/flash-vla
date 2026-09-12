"""Call sites of Pi0's action expert on hand-written CUDA pointwise stages.

A partial backend: it implements the three call sites the floor model put
furthest above their ceiling and nothing else, so a plan routes those here and
leaves the rest on torch.

What changes is launch count, not arithmetic. `norm_qkv_rope` is an RMSNorm, a
GEMM and a RoPE scatter; torch spells that as roughly twenty kernels and this
spells it as three. `norm_gated_ffn` goes from about ten to four. The GEMMs stay
on cuBLAS -- at 51 rows they are skinny, and whether a hand-written GEMM beats
cuBLAS there is a separate question with its own measurement.

Attributed in-graph time before this backend existed, against the ceiling built
from this machine's measured constants:

    action_expert_norm_qkv_rope    6.714 ms   529% of ceiling
    action_expert_norm_gated_ffn   6.316 ms   239%
"""
from __future__ import annotations

from functools import partial

import torch

from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang.wrappers import (
    OPS, ROUTE_CONSTRAINTS)

from . import pointwise as cu

#: Only the call sites this backend implements. A plan naming any other site
#: for this backend is a routing error and `make_wrappers` says so.
NAMES = frozenset({
    "action_expert_norm_qkv_rope",
    "action_expert_norm_gated_ffn",
    "action_expert_attention",
})


def action_expert_norm_qkv_rope(x, scale, weight_qkv, bias, rope, Q, K, V,
                                norm_factor, *, scratch):
    """RMS-scale x, project to QKV, rotate and scatter -- three launches.

    `x` is (tokens, dim) bf16; `weight_qkv` is (dim, heads*head_dim + 2*head_dim).
    Q is (tokens*heads, head_dim), K and V are (tokens, head_dim), all written in
    place. `scale` and `bias` are Pi0.5's AdaRMS terms and are None here.
    Safe during CUDA-graph capture.
    """
    assert scale is None and bias is None, "Pi0 has no AdaRMS scale or shift"
    m, kdim = x.shape
    n = weight_qkv.shape[1]
    normed = scratch("expert_norm", (m, kdim), x.dtype, x.device)
    packed = scratch("expert_qkv", (m, n), x.dtype, x.device)
    cu.rms_norm(x, normed)
    torch.mm(normed, weight_qkv, out=packed)
    cu.rope_scatter(packed, rope, Q, K, V)


def action_expert_norm_gated_ffn(x, scale, gate_w, up_w, gate_b, up_b, out,
                                 norm_factor, *, scratch):
    """out = gelu(rms(x) @ gate_w) * (rms(x) @ up_w) -- four launches.

    `x` is (tokens, dim) bf16, the weights (dim, ffn); `out` is (tokens, ffn),
    written in place. The AdaRMS terms are Pi0.5's and are None here. Safe
    during CUDA-graph capture.
    """
    assert scale is None and gate_b is None and up_b is None, "Pi0 has no AdaRMS terms"
    m, kdim = x.shape
    ffn = gate_w.shape[1]
    normed = scratch("expert_norm", (m, kdim), x.dtype, x.device)
    gate = scratch("expert_gate", (m, ffn), x.dtype, x.device)
    cu.rms_norm(x, normed)
    torch.mm(normed, gate_w, out=gate)
    torch.mm(normed, up_w, out=out)
    cu.gelu_mul(gate, out, out)
    return out


def action_expert_attention(Q, K, V, mask, out, prefix_len, *, scratch):
    """out = softmax(mask(Q @ K^T * scale)) @ V, multi-query, three launches.

    Both GEMMs stay on cuBLAS, which already reaches the tensor core; only the
    glue between them is hand-written. The torch chain spells that glue as two
    `arange`s, three comparisons, a `masked_fill`, a softmax and a cast, and
    rebuilds the mask on every one of the 180 calls per forward.

    `Q` and `out` are (queries, head_dim) bf16 and MAY ALIAS -- the scores
    workspace breaks the dependence, so the final GEMM does not read Q. `K` and
    `V` are (keys, head_dim). Safe during CUDA-graph capture.

    A fully fused single-kernel form was written and measured first
    (`kernels/expert_attention.cu`): 201 us against the torch chain's 70, because
    a CUDA-core dot product costs two shared loads and an FFMA per two FLOP. It
    is not routed here.
    """
    assert mask is None, "Pi0 has no key mask; the prefix length is the integer"
    queries, head_dim = Q.shape
    keys = K.shape[0]
    scores = scratch("expert_scores", (queries, keys), Q.dtype, Q.device)
    torch.mm(Q, K.t(), out=scores)
    cu.expert_masked_softmax(scores, scores, heads=DECODER_HEADS,
                             prefix=prefix_len, scale=float(head_dim ** -0.5))
    torch.mm(scores, V, out=out)
    return out


#: Pi0's action expert is multi-query: eight query heads over one KV head, so
#: the flat query axis is (token, head) and the first `DECODER_HEADS` rows are
#: the state token's.
DECODER_HEADS = 8

ALL_WRAPPERS = {
    "action_expert_norm_qkv_rope": action_expert_norm_qkv_rope,
    "action_expert_norm_gated_ffn": action_expert_norm_gated_ffn,
    "action_expert_attention": action_expert_attention,
}
#: Both need workspace for the normalized activation and the packed projection.
_TAKES_SCRATCH = tuple(ALL_WRAPPERS)


def make_wrappers(scratch, selected_names=None) -> dict:
    """The wrappers of `selected_names` (default: all), bound to `scratch`."""
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - NAMES
    if unknown:
        raise KeyError(f"the rtx5090 cuda backend does not implement {sorted(unknown)}")
    return {name: partial(ALL_WRAPPERS[name], scratch=scratch) for name in names}


__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "make_wrappers"]
