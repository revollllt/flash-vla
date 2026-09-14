"""Every Pi0.5 call site in plain torch, for the RTX 5090's first route."""
from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F

from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels import attention as _attention
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.wrappers import (
    NAMES,
    OPS,
    ROUTE_CONSTRAINTS,
)

#: Mirrored from the TileLang wrappers so this module reads on its own.
VISION_TOKENS = 256
VISION_DIM = 1152
VISION_FFN = 4304
PATCH_FEATURES = 14 * 14 * 3
RMS_EPS = 1e-6
LN_EPS = 1e-5


def _rms(x: torch.Tensor) -> torch.Tensor:
    """The encoder's RMSNorm: the normalized activation, reduced in fp32 (`tl_rms_norm`)."""
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(x.dtype)


def _rms_factor(x: torch.Tensor) -> torch.Tensor:
    """The decoder's RMSNorm: the per-row factor alone, in bf16 (`tl_rms_factor`).

    The kernel reduces in fp32 and stores `F` as bf16, which the consuming GEMM
    reads back; the rounding is part of the contract, not an accident.
    """
    f = x.float()
    return torch.rsqrt(f.pow(2).mean(-1) + RMS_EPS).to(x.dtype)


def _ln(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), LN_EPS).to(x.dtype)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float(), approximate="tanh")


def _rope_pairs(c: torch.Tensor, rope: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Rotate adjacent column pairs of `c` by the interleaved cos/sin in `rope`.

    `rope` is (M, head_dim) holding [cos0, sin0, cos1, sin1, ...]; column j uses
    `rope[:, j % head_dim]`. The rotation is fp32 whatever `c`'s dtype, as the
    kernels' is.
    """
    m, n = c.shape
    x = c.float().view(m, n // 2, 2)
    r = rope.float()[:, :head_dim].view(m, head_dim // 2, 2)
    r = r.repeat(1, n // head_dim, 1)[:, : n // 2]
    cos, sin = r[..., 0], r[..., 1]
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1).view(m, n)


def _scatter_qkv(projected, rope, Q, K, V, head_dim, num_heads):
    """Split a packed (M, q+k+v) projection into Q/K/V, rotating Q and K only."""
    m = projected.shape[0]
    q_dim = num_heads * head_dim
    rotated = _rope_pairs(projected[:, : q_dim + head_dim].contiguous(), rope, head_dim)
    Q.view(m, q_dim).copy_(rotated[:, :q_dim])
    K.copy_(rotated[:, q_dim:])
    V.copy_(projected[:, q_dim + head_dim:])


# --------------------------------------------------------------------------
# Vision encoder -- 27 layers, LayerNorm, M = num_views * 256
# --------------------------------------------------------------------------
def vision_encoder_patch_embed(images, patch_w, patch_b, pos_emb, out):
    """Patchify, project, and add bias plus the per-token positional embedding."""
    views = images.shape[0]
    m = VISION_TOKENS * views
    patches = (images.view(views, 16, 14, 16, 14, 3).permute(0, 1, 3, 2, 4, 5)
               .contiguous().view(m, PATCH_FEATURES))
    y = torch.addmm(patch_b, patches, patch_w.reshape(PATCH_FEATURES, VISION_DIM))
    # The kernel adds pos_emb[i % VISION_TOKENS]; every view shares the 256 rows.
    out.view(m, VISION_DIM).copy_(y + pos_emb.repeat(views, 1))
    return out


def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out, *, scratch=None):
    m = x.shape[0] * VISION_TOKENS
    x2 = x.view(m, VISION_DIM)
    torch.addmm(qkv_b, _ln(x2, norm_w, norm_b), qkv_w, out=out.view(m, qkv_w.shape[1]))
    return out


def vision_encoder_attention(QKV, out):
    """Reused from the H100 route: it was already torch, and already SDPA."""
    return _attention.vision_encoder_attention(QKV, out)


def vision_encoder_out_proj_residual(x, weight, bias, res, out):
    # `res` and `out` are one buffer. Two roundings, as upstream has; spelling
    # it `(x @ w + bias) + res` would round a third time.
    out.add_(torch.addmm(bias, x.view(-1, weight.shape[0]), weight).view_as(out))
    return out


def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out, *, scratch=None):
    m = x.shape[0] * VISION_TOKENS
    x2 = x.view(m, VISION_DIM)
    # GELU sees the BF16-rounded pre-activation, as upstream's
    # `gelu_pytorch_tanh` does. cuBLASLt's fused epilogue would apply it to the
    # fp32 accumulator: one rounding fewer, a different function at bf16.
    out.view(m, VISION_FFN).copy_(
        _gelu(torch.addmm(bias, _ln(x2, norm_w, norm_b), weight)).to(out.dtype))
    return out


def vision_encoder_ffn_down_residual(x, weight, bias, res, out):
    out.add_(torch.addmm(bias, x.view(-1, weight.shape[0]), weight).view_as(out))
    return out


# --------------------------------------------------------------------------
# LLM backbone -- 18 layers, RMSNorm, M = prefix_len = 768 + 200
# --------------------------------------------------------------------------
def llm_backbone_projector(x, norm_w, norm_b, proj_w, proj_b, out, x_norm):
    m = x.shape[0] * VISION_TOKENS
    x2, x_norm2 = x.view(m, VISION_DIM), x_norm.view(m, VISION_DIM)
    x_norm2.copy_(_ln(x2, norm_w, norm_b))
    torch.addmm(proj_b, x_norm2, proj_w, out=out[:m])
    return out


def llm_backbone_embed_prompt(token_ids, table, scale, out):
    """out = table[token_ids] * scale, into the language rows of `llm_backbone_x`.

    `scale` carries sqrt(width) on valid rows and zero on padding, so the one
    multiply both applies the embedder's scale and zeroes the padded rows.
    """
    torch.index_select(table, 0, token_ids.to(torch.int32), out=out)
    out.mul_(scale)
    return out


def llm_backbone_norm_qkv_rope(x, weight_qkv, rope, Q, K, V, x_norm):
    """RMSNorm, QKV projection, RoPE, scatter.

    The encoder normalizes x BEFORE the GEMM and rounds the projection to bf16
    before rotating, where the decoder folds a per-row factor into the GEMM
    instead. Both orderings match upstream and are not interchangeable.
    """
    m = x.shape[0]
    head_dim, num_heads = V.shape[1], Q.shape[0] // m
    x_norm[:m].copy_(_rms(x))
    _scatter_qkv((x_norm[:m] @ weight_qkv).to(x.dtype), rope, Q, K, V, head_dim, num_heads)


def llm_backbone_attention(Q, K, V, scale, mask, out):
    """Reused from the H100 route: a torch multi-query chain with the key mask."""
    _attention.llm_backbone_attention(Q, K, V, scale, mask, out)
    return out


def llm_backbone_out_proj_residual(x, weight, out):
    out.addmm_(x, weight)
    return out


def llm_backbone_norm_gated_ffn(x, gate_w, up_w, out, x_norm):
    m = x.shape[0]
    x_norm[:m].copy_(_rms(x))
    n = x_norm[:m]
    out[:m].copy_((_gelu(n @ gate_w) * (n @ up_w).float()).to(out.dtype))
    return out


def llm_backbone_ffn_down_residual(x, weight, out):
    out.addmm_(x, weight)
    return out


# --------------------------------------------------------------------------
# Action expert -- 18 layers x 10 flow steps, M = chunk = 50, AdaRMSNorm
# --------------------------------------------------------------------------
def action_expert_action_in_proj(x, weight, bias, out):
    """out = x @ weight + bias: a bare linear, the timestep arriving through AdaRMSNorm."""
    out.copy_((x @ weight + bias).view_as(out))
    return out


def action_expert_norm_qkv_rope(x, scale, weight_qkv, bias, rope, Q, K, V, norm_factor):
    """AdaRMS-scale x, project, add the shift bias, rotate, scatter.

    `scale` meets the A operand in bf16 (the kernel's mainloop); the per-row
    factor and the bias land in the fp32 epilogue, the factor first, and only
    then does the rotation run.
    """
    m = x.shape[0]
    head_dim, num_heads = V.shape[1], Q.shape[0] // m
    factor = _rms_factor(x)
    norm_factor[:m].copy_(factor)
    projected = ((x * scale) @ weight_qkv).float() * factor[:, None].float() + bias.float()
    _scatter_qkv(projected, rope, Q, K, V, head_dim, num_heads)


def action_expert_attention(Q, K, V, mask, out, prefix_len=None, *, scratch=None):
    """out = softmax(Q @ K^T * scale + mask) @ V, multi-query over a flat query axis.

    `prefix_len` is accepted for signature parity with the vocabulary and unused:
    the additive `mask` carries the prefix, exactly as it does for the
    FlashDecoding kernel this replaces. `out` aliases Q, so the result is
    materialized before the copy.
    """
    head_dim = Q.shape[1]
    logits = (Q.float() @ K.float().T) * float(head_dim ** -0.5) + mask.float()[None, :]
    probs = torch.softmax(logits, dim=-1).to(V.dtype)
    out.copy_(probs @ V)
    return out


def action_expert_out_proj_residual(x, weight, gate, out):
    """out = out + (x @ weight) * gate, the AdaRMSNorm gated residual."""
    out.copy_(((x @ weight).float() * gate.float() + out.float()).to(out.dtype))
    return out


def action_expert_norm_gated_ffn(x, scale, gate_w, up_w, gate_b, up_b, out, norm_factor):
    """out = gelu(a @ gate_w + gate_b) * (a @ up_w + up_b), a = x * factor * scale.

    Both the row factor and the AdaRMS scale meet A in bf16 here, which is where
    `tl_ada_scaled_gate` applies them; each branch takes its own bias in fp32
    before the activation.
    """
    m = x.shape[0]
    factor = _rms_factor(x)
    norm_factor[:m].copy_(factor)
    a = x * factor[:, None] * scale
    out.copy_((_gelu((a @ gate_w).float() + gate_b.float())
               * ((a @ up_w).float() + up_b.float())).to(out.dtype))
    return out


def action_expert_ffn_down_residual(x, weight, gate, out):
    """Same contract as the out-projection, larger K."""
    out.copy_(((x @ weight).float() * gate.float() + out.float()).to(out.dtype))
    return out


def action_expert_action_out_proj(x, weight, bias, out, norm_factor):
    """out += bias + rms(x) @ weight, with the final AdaRMSNorm folded into both.

    The final norm's scale, its shift and the Euler dt all fold into the
    per-step `decoder_action_out_proj_w` and `_b` at checkpoint load, which
    leaves exactly this signature. `norm_factor` is accepted for signature
    parity and never written -- the factor exists only inside the fused kernel,
    and only inside this call here.
    """
    factor = _rms_factor(x)
    out.copy_((((x @ weight).float() * factor[:, None].float()
                + bias.float() + out.float())).to(out.dtype))
    return out


ALL_WRAPPERS = {
    name: obj for name, obj in list(globals().items())
    if callable(obj) and not name.startswith("_") and name in NAMES
}

#: Wrappers whose H100 counterparts take the runner's allocator. These ignore it.
_TAKES_SCRATCH = ("action_expert_attention", "vision_encoder_norm_qkv",
                  "vision_encoder_norm_ffn_up")


def make_wrappers(scratch, selected_names=None) -> dict:
    """The wrappers of `selected_names` (default: all), bound to `scratch`."""
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
