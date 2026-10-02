"""Device-independent Pi0 call sites in torch, shared by NVIDIA Targets.

This bring-up backend runs the model end to end without device-specific kernels.

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

from flash_vla.models.pi0 import attention as attention_kernels
from flash_vla.models.pi0.ops import CALL_SITES as NAMES
from flash_vla.models.pi0.spec import DECODER_HEADS, VISION_DIM, VISION_FFN, VISION_TOKENS
from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch

#: Features of one 14x14 RGB patch.
PATCH_FEATURES = 14 * 14 * 3
RMS_EPS = 1e-6
LN_EPS = 1e-5


def rms(x: torch.Tensor) -> torch.Tensor:
    """x scaled by rsqrt(mean(x^2) + eps), reduced in fp32 and returned in x's dtype."""
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(x.dtype)


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), LN_EPS).to(x.dtype)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float(), approximate="tanh").to(x.dtype)


def rope_pairs(c: torch.Tensor, rope: torch.Tensor, head_dim: int) -> torch.Tensor:
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


def scatter_qkv(projected: torch.Tensor, rope: torch.Tensor, Q: torch.Tensor,
                K: torch.Tensor, V: torch.Tensor, head_dim: int, num_heads: int) -> None:
    """Split a packed (M, q+k+v) projection into Q/K/V, rotating Q and K only."""
    m = projected.shape[0]
    q_dim = num_heads * head_dim
    rotated = rope_pairs(projected[:, : q_dim + head_dim].contiguous(), rope, head_dim)
    Q.view(m, q_dim).copy_(rotated[:, :q_dim])
    K.copy_(rotated[:, q_dim:])
    V.copy_(projected[:, q_dim + head_dim:])


# --------------------------------------------------------------------------
# Vision encoder
# --------------------------------------------------------------------------
def vision_encoder_patch_embed(images: torch.Tensor, patch_w: torch.Tensor, patch_b: torch.Tensor,
                               pos_emb: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Patchify, project, and add bias plus the per-token positional embedding."""
    views = images.shape[0]
    m = VISION_TOKENS * views
    patches = (images.view(views, 16, 14, 16, 14, 3).permute(0, 1, 3, 2, 4, 5)
               .contiguous().view(m, PATCH_FEATURES))
    y = patches @ patch_w.reshape(PATCH_FEATURES, VISION_DIM) + patch_b
    # The kernel adds pos_emb[i % VISION_TOKENS]; every view shares the 256 rows.
    out.view(m, VISION_DIM).copy_(y + pos_emb.repeat(views, 1))
    return out


def vision_encoder_norm_qkv(x: torch.Tensor, norm_w: torch.Tensor, norm_b: torch.Tensor,
                            qkv_w: torch.Tensor, qkv_b: torch.Tensor, out: torch.Tensor,
                            *, scratch: Scratch | None = None) -> torch.Tensor:
    m = x.shape[0] * VISION_TOKENS
    x2 = x.view(m, VISION_DIM)
    out.view(m, qkv_w.shape[1]).copy_(layer_norm(x2, norm_w, norm_b) @ qkv_w + qkv_b)
    return out


def vision_encoder_attention(QKV: torch.Tensor, out: torch.Tensor) -> None:
    """Reused from the H100 route: it was already torch, and already SDPA."""
    return attention_kernels.vision_encoder_attention(QKV, out)


def vision_encoder_out_proj_residual(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                     res: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.copy_((x.view(-1, weight.shape[0]) @ weight + bias).view_as(res) + res)
    return out


def vision_encoder_norm_ffn_up(x: torch.Tensor, norm_w: torch.Tensor, norm_b: torch.Tensor,
                               weight: torch.Tensor, bias: torch.Tensor, out: torch.Tensor,
                               *, scratch: Scratch | None = None) -> torch.Tensor:
    m = x.shape[0] * VISION_TOKENS
    x2 = x.view(m, VISION_DIM)
    out.view(m, VISION_FFN).copy_(gelu(layer_norm(x2, norm_w, norm_b) @ weight + bias))
    return out


def vision_encoder_ffn_down_residual(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                    res: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.copy_((x.view(-1, weight.shape[0]) @ weight + bias).view_as(res) + res)
    return out


# --------------------------------------------------------------------------
# LLM backbone
# --------------------------------------------------------------------------
def llm_backbone_projector(x: torch.Tensor, norm_w: torch.Tensor, norm_b: torch.Tensor,
                           proj_w: torch.Tensor, proj_b: torch.Tensor, out: torch.Tensor,
                           x_norm: torch.Tensor) -> torch.Tensor:
    m = x.shape[0] * VISION_TOKENS
    x2, x_norm2 = x.view(m, VISION_DIM), x_norm.view(m, VISION_DIM)
    x_norm2.copy_(layer_norm(x2, norm_w, norm_b))
    out[:m].copy_(x_norm2 @ proj_w + proj_b)
    return out


def llm_backbone_embed_prompt(token_ids: None, table: torch.Tensor, scale: None,
                              out: torch.Tensor) -> torch.Tensor:
    assert token_ids is None and scale is None, "Pi0's prompt embeddings are precomputed"
    out.copy_(table)
    return out


def llm_backbone_norm_qkv_rope(x: torch.Tensor, weight_qkv: torch.Tensor, rope: torch.Tensor,
                              Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                              x_norm: torch.Tensor, *, scratch: Scratch | None = None) -> None:
    """RMSNorm, QKV projection, RoPE, scatter.

    The encoder normalises x BEFORE the GEMM and rounds to bf16 before rotating,
    where the decoder folds a scale factor into the GEMM instead. The two
    orderings both match upstream and are not interchangeable.
    """
    m = x.shape[0]
    head_dim, num_heads = V.shape[1], Q.shape[0] // x.shape[0]
    x_norm[:m].copy_(rms(x))
    scatter_qkv(x_norm[:m] @ weight_qkv, rope, Q, K, V, head_dim, num_heads)


def llm_backbone_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           scale: float, mask: None, out: torch.Tensor) -> torch.Tensor:
    assert mask is None, "Pi0 has no key mask"
    attention_kernels.llm_backbone_attention(Q, K, V, scale, out)
    return out


def llm_backbone_out_proj_residual(x: torch.Tensor, weight: torch.Tensor,
                                  out: torch.Tensor) -> torch.Tensor:
    out.add_((x @ weight).view_as(out))
    return out


def llm_backbone_norm_gated_ffn(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor,
                               out: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    x_norm[:m].copy_(rms(x))
    n = x_norm[:m]
    out[:m].copy_(gelu(n @ gate_w) * (n @ up_w))
    return out


def llm_backbone_ffn_down_residual(x: torch.Tensor, weight: torch.Tensor,
                                  out: torch.Tensor) -> torch.Tensor:
    out.add_((x @ weight).view_as(out))
    return out


# --------------------------------------------------------------------------
# Action expert
# --------------------------------------------------------------------------
def action_expert_state_proj(x: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.copy_((x.view(1, -1) @ weight + bias).view_as(out))
    return out


def action_expert_action_in_proj(x: torch.Tensor, weight: torch.Tensor,
                                 bias: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.copy_(F.silu((x @ weight + bias).float()).to(out.dtype).view_as(out))
    return out


def action_expert_action_mlp(x: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    out.copy_((x @ weight + bias).view_as(out))
    return out


def action_expert_norm_qkv_rope(x: torch.Tensor, scale: None, weight_qkv: torch.Tensor,
                               bias: None, rope: torch.Tensor, Q: torch.Tensor, K: torch.Tensor,
                               V: torch.Tensor, norm_factor: torch.Tensor) -> None:
    assert scale is None and bias is None, "Pi0 has no AdaRMS scale or shift"
    m = x.shape[0]
    head_dim, num_heads = V.shape[1], Q.shape[0] // m
    scatter_qkv(rms(x) @ weight_qkv, rope, Q, K, V, head_dim, num_heads)


def action_expert_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: None,
                            out: torch.Tensor, prefix_len: int,
                            *, scratch: Scratch | None = None) -> torch.Tensor:
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


def action_expert_norm_gated_ffn(x: torch.Tensor, scale: None, gate_w: torch.Tensor,
                                up_w: torch.Tensor, gate_b: None, up_b: None,
                                out: torch.Tensor, norm_factor: torch.Tensor) -> torch.Tensor:
    assert scale is None and gate_b is None and up_b is None, "Pi0 has no AdaRMS terms"
    n = rms(x)
    out.copy_(gelu(n @ gate_w) * (n @ up_w))
    return out


def action_expert_out_proj_residual(x: torch.Tensor, weight: torch.Tensor, gate: None,
                                    out: torch.Tensor) -> torch.Tensor:
    assert gate is None, "the gate is Pi0.5's"
    out.add_((x @ weight).view_as(out))
    return out


def action_expert_ffn_down_residual(x: torch.Tensor, weight: torch.Tensor, gate: None,
                                   out: torch.Tensor) -> torch.Tensor:
    assert gate is None, "the gate is Pi0.5's"
    out.add_((x @ weight).view_as(out))
    return out


def action_expert_action_out_proj(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                 out: torch.Tensor, norm_factor: torch.Tensor) -> torch.Tensor:
    out.add_((rms(x) @ weight + bias).view_as(out))
    return out


ALL_WRAPPERS = {
    name: obj for name, obj in list(globals().items())
    if callable(obj) and not name.startswith("_") and name in NAMES
}

#: Wrappers whose H100 counterparts request workspace. None of these do -- torch
#: allocates its own temporaries -- but the runner still passes the allocator, so
#: they accept and ignore it.
TAKES_SCRATCH = ("action_expert_attention", "llm_backbone_norm_qkv_rope",
                  "vision_encoder_norm_qkv", "vision_encoder_norm_ffn_up")


def make_wrappers(scratch: Scratch,
                  selected_names: frozenset[str] | None = None) -> dict[str, Wrapper]:
    """The wrappers of `selected_names` (default: all), bound to `scratch`."""
    from functools import partial

    names = NAMES if selected_names is None else selected_names
    return {name: (partial(ALL_WRAPPERS[name], scratch=scratch) if name in TAKES_SCRATCH
                   else ALL_WRAPPERS[name])
            for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
