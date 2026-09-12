"""Every Pi0 call site in plain torch, for the RTX 5090's first route.

This is the bring-up backend: it exists to make the model run end to end on
sm_120, not to be fast. Nothing here is tuned, fused or hand-written, and the
optimization workflow starts from whatever it measures.

Why torch rather than the H100 TileLang route. Pi0's TileLang tile shapes were
tuned against 227 KB of shared memory per block; ten of nineteen exceed this
part's 99 KB [smem.bytes.cta.max] and fail at launch. Re-tiling them by hand is
guesswork against the wrong machine's tuning, and the project's architecture
asks for a route customised per hardware rather than an inherited one -- so the
first version drops the inheritance entirely and the kernels get written fresh,
against this machine's measured constants, in the optimization loop.

Semantics are taken from the TileLang wrappers this replaces, which are in turn
a 1:1 port of upstream:

* RMSNorm is ``rsqrt(mean(x^2) + 1e-6)``, reduced in fp32.
* LayerNorm is eps 1e-5.
* GELU is the tanh approximation (the kernels spell it as a sigmoid with
  ``1.5957691216057308`` and ``0.044715``, which is exactly that).
* RoPE rotates ADJACENT COLUMN PAIRS ``(2p, 2p+1)``, with ``rope[:, j % head_dim]``
  as cosine and the next column as sine -- an interleaved layout, not the
  split-halves one.
* The decoder mask keeps key ``j`` for flat query row ``gi`` when
  ``gi >= num_heads or j <= prefix_len``: the state token's heads see only the
  prefix, the action tokens see everything.

Every function writes into its caller's buffer. The runtime captures a CUDA
graph over these, so a function that rebinds instead of writing in place would
capture a dead pointer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang.kernels import attention as _attention
from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang.wrappers import NAMES, OPS, ROUTE_CONSTRAINTS

#: Pi0 shape constants, mirrored from the TileLang wrappers rather than imported
#: so this module reads on its own.
VISION_TOKENS = 256
VISION_DIM = 1152
VISION_FFN = 4304
PATCH_FEATURES = 14 * 14 * 3
ENCODER_DIM = 2048
DECODER_HEADS = 8
RMS_EPS = 1e-6
LN_EPS = 1e-5


def _rms(x: torch.Tensor) -> torch.Tensor:
    """x scaled by rsqrt(mean(x^2) + eps), reduced in fp32 and returned in x's dtype."""
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(x.dtype)


def _ln(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), LN_EPS).to(x.dtype)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float(), approximate="tanh").to(x.dtype)


def _rope_pairs(c: torch.Tensor, rope: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Rotate adjacent column pairs of `c` by the interleaved cos/sin in `rope`.

    `rope` is (M, head_dim) holding [cos0, sin0, cos1, sin1, ...]; column j uses
    `rope[:, j % head_dim]`. Computed in fp32 and cast back, matching the
    kernels, which accumulate the rotation in fp32 whatever the operand dtype.
    """
    m, n = c.shape
    x = c.float().view(m, n // 2, 2)
    r = rope.float()[:, : head_dim].view(m, head_dim // 2, 2)
    r = r.repeat(1, n // head_dim, 1)[:, : n // 2]
    cos, sin = r[..., 0], r[..., 1]
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1).view(m, n).to(c.dtype)


def _scatter_qkv(projected, rope, Q, K, V, head_dim, num_heads):
    """Split a packed (M, q+k+v) projection into Q/K/V, rotating Q and K only."""
    m = projected.shape[0]
    q_dim = num_heads * head_dim
    rotated = _rope_pairs(projected[:, : q_dim + head_dim].contiguous(), rope, head_dim)
    Q.view(m, q_dim).copy_(rotated[:, :q_dim])
    K.copy_(rotated[:, q_dim:])
    V.copy_(projected[:, q_dim + head_dim:])


# --------------------------------------------------------------------------
# Vision encoder
# --------------------------------------------------------------------------
def vision_encoder_patch_embed(images, patch_w, patch_b, pos_emb, out):
    """Patchify, project, and add bias plus the per-token positional embedding."""
    views = images.shape[0]
    m = VISION_TOKENS * views
    patches = (images.view(views, 16, 14, 16, 14, 3).permute(0, 1, 3, 2, 4, 5)
               .contiguous().view(m, PATCH_FEATURES))
    y = patches @ patch_w.reshape(PATCH_FEATURES, VISION_DIM) + patch_b
    # The kernel adds pos_emb[i % VISION_TOKENS]; every view shares the 256 rows.
    out.view(m, VISION_DIM).copy_(y + pos_emb.repeat(views, 1))
    return out


def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out, *, scratch=None):
    m = x.shape[0] * VISION_TOKENS
    x2 = x.view(m, VISION_DIM)
    out.view(m, qkv_w.shape[1]).copy_(_ln(x2, norm_w, norm_b) @ qkv_w + qkv_b)
    return out


def vision_encoder_attention(QKV, out):
    """Reused from the H100 route: it was already torch, and already SDPA."""
    return _attention.vision_encoder_attention(QKV, out)


def vision_encoder_out_proj_residual(x, weight, bias, res, out):
    out.copy_((x.view(-1, weight.shape[0]) @ weight + bias).view_as(res) + res)
    return out


def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out, *, scratch=None):
    m = x.shape[0] * VISION_TOKENS
    x2 = x.view(m, VISION_DIM)
    out.view(m, VISION_FFN).copy_(_gelu(_ln(x2, norm_w, norm_b) @ weight + bias))
    return out


def vision_encoder_ffn_down_residual(x, weight, bias, res, out):
    out.copy_((x.view(-1, weight.shape[0]) @ weight + bias).view_as(res) + res)
    return out


# --------------------------------------------------------------------------
# LLM backbone
# --------------------------------------------------------------------------
def llm_backbone_projector(x, norm_w, norm_b, proj_w, proj_b, out, x_norm):
    m = x.shape[0] * VISION_TOKENS
    x2, x_norm2 = x.view(m, VISION_DIM), x_norm.view(m, VISION_DIM)
    x_norm2.copy_(_ln(x2, norm_w, norm_b))
    out[:m].copy_(x_norm2 @ proj_w + proj_b)
    return out


def llm_backbone_embed_prompt(token_ids, table, scale, out):
    assert token_ids is None and scale is None, "Pi0's prompt embeddings are precomputed"
    out.copy_(table)
    return out


def llm_backbone_norm_qkv_rope(x, weight_qkv, rope, Q, K, V, x_norm, *, scratch=None):
    """RMSNorm, QKV projection, RoPE, scatter.

    The encoder normalises x BEFORE the GEMM and rounds to bf16 before rotating,
    where the decoder folds a scale factor into the GEMM instead. The two
    orderings both match upstream and are not interchangeable.
    """
    m = x.shape[0]
    head_dim, num_heads = V.shape[1], Q.shape[0] // x.shape[0]
    x_norm[:m].copy_(_rms(x))
    _scatter_qkv(x_norm[:m] @ weight_qkv, rope, Q, K, V, head_dim, num_heads)


def llm_backbone_attention(Q, K, V, scale, mask, out):
    assert mask is None, "Pi0 has no key mask"
    _attention.llm_backbone_attention(Q, K, V, scale, out)
    return out


def llm_backbone_out_proj_residual(x, weight, out):
    out.add_((x @ weight).view_as(out))
    return out


def llm_backbone_norm_gated_ffn(x, gate_w, up_w, out, x_norm):
    m = x.shape[0]
    x_norm[:m].copy_(_rms(x))
    n = x_norm[:m]
    out[:m].copy_(_gelu(n @ gate_w) * (n @ up_w))
    return out


def llm_backbone_ffn_down_residual(x, weight, out):
    out.add_((x @ weight).view_as(out))
    return out


# --------------------------------------------------------------------------
# Action expert
# --------------------------------------------------------------------------
def action_expert_state_proj(x, weight, bias, out):
    out.copy_((x.view(1, -1) @ weight + bias).view_as(out))
    return out


def action_expert_action_in_proj(x, weight, bias, out):
    out.copy_(F.silu((x @ weight + bias).float()).to(out.dtype).view_as(out))
    return out


def action_expert_action_mlp(x, weight, bias, out):
    out.copy_((x @ weight + bias).view_as(out))
    return out


def action_expert_norm_qkv_rope(x, scale, weight_qkv, bias, rope, Q, K, V, norm_factor):
    assert scale is None and bias is None, "Pi0 has no AdaRMS scale or shift"
    m = x.shape[0]
    head_dim, num_heads = V.shape[1], Q.shape[0] // m
    _scatter_qkv(_rms(x) @ weight_qkv, rope, Q, K, V, head_dim, num_heads)


def action_expert_attention(Q, K, V, mask, out, prefix_len, *, scratch=None):
    """out = softmax(mask(Q @ K^T * scale)) @ V, multi-query over a flat query axis.

    The mask keeps key j for flat row gi when `gi >= DECODER_HEADS or
    j <= prefix_len`: the state token's heads attend only to the prefix.
    `out` may alias Q, so the result is materialised before the copy.
    """
    assert mask is None, "Pi0 has no key mask; the prefix length is the integer"
    queries, head_dim = Q.shape
    keys = K.shape[0]
    logits = (Q.float() @ K.float().T) * float(head_dim ** -0.5)
    row = torch.arange(queries, device=Q.device).unsqueeze(1)
    col = torch.arange(keys, device=Q.device).unsqueeze(0)
    keep = (row >= DECODER_HEADS) | (col <= prefix_len)
    logits = logits.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(logits, dim=-1).to(V.dtype)
    out.copy_(probs @ V)
    return out


def action_expert_norm_gated_ffn(x, scale, gate_w, up_w, gate_b, up_b, out, norm_factor):
    assert scale is None and gate_b is None and up_b is None, "Pi0 has no AdaRMS terms"
    n = _rms(x)
    out.copy_(_gelu(n @ gate_w) * (n @ up_w))
    return out


def action_expert_out_proj_residual(x, weight, gate, out):
    assert gate is None, "the gate is Pi0.5's"
    out.add_((x @ weight).view_as(out))
    return out


def action_expert_ffn_down_residual(x, weight, gate, out):
    assert gate is None, "the gate is Pi0.5's"
    out.add_((x @ weight).view_as(out))
    return out


def action_expert_action_out_proj(x, weight, bias, out, norm_factor):
    out.add_((_rms(x) @ weight + bias).view_as(out))
    return out


ALL_WRAPPERS = {
    name: obj for name, obj in list(globals().items())
    if callable(obj) and not name.startswith("_") and name in NAMES
}

#: Wrappers whose H100 counterparts request workspace. None of these do -- torch
#: allocates its own temporaries -- but the runner still passes the allocator, so
#: they accept and ignore it.
_TAKES_SCRATCH = ("action_expert_attention", "llm_backbone_norm_qkv_rope",
                  "vision_encoder_norm_qkv", "vision_encoder_norm_ffn_up")


def make_wrappers(scratch, selected_names=None) -> dict:
    """The wrappers of `selected_names` (default: all), bound to `scratch`."""
    from functools import partial

    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"torch backend does not implement {sorted(unknown)}")
    missing = names - set(ALL_WRAPPERS)
    if missing:
        raise KeyError(f"torch backend is missing wrappers for {sorted(missing)}")
    return {name: (partial(ALL_WRAPPERS[name], scratch=scratch) if name in _TAKES_SCRATCH
                   else ALL_WRAPPERS[name])
            for name in names}


__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "make_wrappers"]
