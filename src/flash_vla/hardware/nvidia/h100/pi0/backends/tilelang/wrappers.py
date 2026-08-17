"""Per-call-site wrappers: one function per operation in the Pi0 forward pass.

Each wrapper owns the tile configuration for its call site, compiles the kernel
once per shape, and calls it. Kernels are shared between call sites; configs are
not -- the same GEMM at M=51 and M=768 wants different tiles and a different
warp-specialization setting, which is why some kernels appear here twice through
two variants.

Every config below came from a sweep at the call site's real shape, measured
inside a CUDA graph with cold weights (`benchmarks.autotune`). They are not defaults
and not guesses: three of them were wrong in ways an eager benchmark could not
see, costing 2.4x on one kernel. Re-tune with
`python -m benchmarks` after any shape change, and re-check
correctness, not just time -- `kernels.tl_scaled_gate` in particular is
numerically sensitive to its tiling.

Shapes are passed through unpadded. TileLang masks out-of-bounds rows, reduction
columns and output columns exactly as Triton's masks do, so nothing here pads a
buffer; wrappers hand the ragged tensors straight to the kernel and let it write
its destination in place.
"""
from __future__ import annotations

import contextlib

import torch

from flash_vla.runtime.cuda import ScratchPool

from .kernels import base as kernels

_CACHE: dict = {}
_POOL = ScratchPool()


def set_pool(pool: ScratchPool) -> None:
    """Install the scratch pool used by every wrapper (see `use_pool` for scoped swaps)."""
    global _POOL
    _POOL = pool


@contextlib.contextmanager
def use_pool(pool: ScratchPool):
    """Temporarily route scratch allocation through `pool`, e.g. during graph capture."""
    global _POOL
    previous = _POOL
    _POOL = pool
    try:
        yield
    finally:
        _POOL = previous


def scratch(role: str, shape, dtype, device) -> torch.Tensor:
    """Graph-safe temporary from the active pool.

    Always go through this rather than capturing `_POOL` at import time: the
    active pool is swapped per engine by `use_pool`, and a module that bound the
    object once would keep writing into the wrong one.
    """
    return _POOL.get(role, shape, dtype, device)


def _compiled(kernel, **const):
    """Compile `kernel` for one shape and config, memoized on both.

    Keyed on `kernel.tl_name` rather than the kernel object, which TileLang
    leaves unhashable. The name distinguishes the two variants of a shared body,
    so a WS-on and a WS-off call site never collide.
    """
    key = (kernel.tl_name, tuple(sorted(const.items())))
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = kernel.compile(**const)
        _CACHE[key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# Decoder (18 layers x 10 diffusion steps, M = chunk + 1 = 51)
#
# Every shape here is under one wave, so these are latency-bound rather than
# compute-bound: the configs favour occupancy and pipeline depth over big tiles,
# and warp specialization is off wherever the kernel allows it.
# ---------------------------------------------------------------------------
_DEC_STATE_PROJ = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, NUM_STAGES=2, THREADS=128)
_DEC_ACTION_IN = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, NUM_STAGES=2, THREADS=128)
_DEC_ACTION_MLP = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, NUM_STAGES=3, THREADS=128)
_DEC_QKV = dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, NUM_STAGES=4, THREADS=128)
_DEC_SCORES = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, NUM_STAGES=5, THREADS=128)
_DEC_SOFTMAX = dict(BLOCK_M=1, THREADS=256)
_DEC_ATTN_V = dict(BLOCK_M=32, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=4, THREADS=128)
_DEC_RESIDUAL = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=4, THREADS=128)
_DEC_GATE = dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=3, THREADS=128)
_DEC_OUT_PROJ = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=3, THREADS=128)
_DEC_RMS = dict(BLOCK_M=2, BLOCK_K=256, THREADS=128)

DECODER_HEADS = 8


def _rms_factor(x, out, cfg=_DEC_RMS):
    """Write rsqrt(mean(x^2)+eps) into `out`, which the consuming GEMM then scales by."""
    M, K = x.shape
    _compiled(kernels.tl_rms_factor, M=M, K=K, **cfg)(x, out)
    return out


def decoder_state_proj(x, weight, bias, out):
    """out = x @ weight + bias, for the single state token (upstream matmul_1_32_1024_bias)."""
    _compiled(kernels.tl_matmul_bias, M=1, N=weight.shape[1], K=weight.shape[0],
              **_DEC_STATE_PROJ)(x.view(1, -1), weight, bias, out)
    return out


def decoder_action_in_proj(x, weight, bias, out):
    """out = silu(x @ weight + bias) (upstream matmul_k_32_1024_bias_silu)."""
    M, K = x.shape
    kfn = _compiled(kernels.tl_matmul_bias_silu, M=M, N=weight.shape[1], K=K, **_DEC_ACTION_IN)
    out.copy_(kfn(x, weight, bias))
    return out


def decoder_action_mlp(x, weight, bias, out):
    """out = x @ weight + bias (upstream matmul_k_1024_1024_bias)."""
    M, K = x.shape
    _compiled(kernels.tl_matmul_bias, M=M, N=weight.shape[1], K=K,
              **_DEC_ACTION_MLP)(x, weight, bias, out)
    return out


def decoder_norm_qkv_rope(x, weight_qkv, rope, Q, K, V, norm_factor):
    """RMS-scale x, project to QKV, apply RoPE, scatter into Q/K/V in place.

    Upstream rms_matmul_k_1024_2560_qkv_rope. The deep pipeline is what matters
    here: with 18 distinct per-layer weights the 5 MB projection weight is always
    a cold read, and overlapping it with compute is the whole cost of the kernel.
    """
    M, Kdim = x.shape
    head_dim = V.shape[1]
    num_heads = Q.shape[0] // M
    factor = _rms_factor(x, norm_factor[:M])
    kfn = _compiled(kernels.tl_qkv_gemm_rope, M=M, N=weight_qkv.shape[1], K=Kdim,
                    HEAD_DIM=head_dim, NUM_HEADS=num_heads, **_DEC_QKV)
    kfn(x, factor, weight_qkv, rope, Q.view(M, num_heads * head_dim), K, V)


def decoder_attention(Q, K, V, scores, out, encoder_seq_len):
    """out = softmax_mask0(Q @ K^T * scale) @ V, materializing the score matrix.

    Upstream matmul_k8_256_n_softmax_mask0 + matmul_k8_n_256. The fused path
    replaces this with FlashDecoding (`fused_wrappers.decoder_attention`), which
    keeps the scores in SRAM; this version stays as the readable reference and
    the fallback when the fusion is disabled.

    `out` aliases Q (both are the query buffer) -- safe, since the final GEMM
    reads its input through `scores`, not Q.
    """
    queries, head_dim = Q.shape
    keys = K.shape[0]
    frag_w = max(1024, 1 << (keys - 1).bit_length())

    scores_fn = _compiled(kernels.tl_matmul_abT_scale, M=queries, N=keys, K=head_dim,
                          SCALE=float(head_dim ** -0.5), **_DEC_SCORES)
    softmax_fn = _compiled(kernels.tl_softmax_mask0, Q=queries, N=keys, KEYS=keys,
                           NUM_HEADS=DECODER_HEADS, ENC_LEN=encoder_seq_len, FRAG_W=frag_w,
                           **_DEC_SOFTMAX)
    attn_v_fn = _compiled(kernels.tl_matmul, M=queries, N=head_dim, K=keys, **_DEC_ATTN_V)

    raw = scratch("attn_scores", (queries, keys), Q.dtype, Q.device)
    scores_fn(Q, K, raw)
    softmax_fn(raw, scores)
    attn_v_fn(scores, V, out)
    return out


def decoder_out_proj_residual(x, weight, out):
    """out += x @ weight, in place (upstream matmul_k_2048_1024_res)."""
    M, K = x.shape
    _compiled(kernels.tl_matmul_res, M=M, N=weight.shape[1], K=K,
              **_DEC_RESIDUAL)(x, weight, out, out)
    return out


def decoder_ffn_down_residual(x, weight, out):
    """out += x @ weight, in place (upstream matmul_k_4096_1024_res)."""
    M, K = x.shape
    _compiled(kernels.tl_matmul_res, M=M, N=weight.shape[1], K=K,
              **_DEC_RESIDUAL)(x, weight, out, out)
    return out


def decoder_norm_gated_ffn(x, gate_w, up_w, out, norm_factor):
    """out = gelu(rms(x) @ gate_w) * (rms(x) @ up_w) (upstream rms_matmul_k_1024_4096_gate)."""
    M, K = x.shape
    factor = _rms_factor(x, norm_factor[:M])
    _compiled(kernels.tl_scaled_gate, M=M, N=gate_w.shape[1], K=K,
              **_DEC_GATE)(x, factor, gate_w, up_w, out)
    return out


def decoder_action_out_proj(x, weight, bias, out, norm_factor):
    """out += bias + rms(x) @ weight (upstream rms_matmul_k_1024_32_bias_res)."""
    M, K = x.shape
    factor = _rms_factor(x, norm_factor[:M])
    kfn = _compiled(kernels.tl_scaled_matmul_bias_res, M=M, N=weight.shape[1], K=K,
                    **_DEC_OUT_PROJ)
    out.copy_(kfn(x, factor, weight, bias, out))
    return out


DECODER_WRAPPERS = {
    "decoder_state_proj": decoder_state_proj,
    "decoder_action_in_proj": decoder_action_in_proj,
    "decoder_action_mlp": decoder_action_mlp,
    "decoder_norm_qkv_rope": decoder_norm_qkv_rope,
    "decoder_attention": decoder_attention,
    "decoder_out_proj_residual": decoder_out_proj_residual,
    "decoder_ffn_down_residual": decoder_ffn_down_residual,
    "decoder_norm_gated_ffn": decoder_norm_gated_ffn,
    "decoder_action_out_proj": decoder_action_out_proj,
}


# ---------------------------------------------------------------------------
# Vision (27 layers, LayerNorm, M = num_views * 256 = 768, hidden 1152, FFN 4304)
#
# High occupancy, so warp specialization pays off everywhere except the patch
# embedding and QKV projection. Two axes here divide no practical block size
# (4304 = 16 * 269 with 269 prime, and K=588); they run unpadded on the kernel's
# masking, same as Triton.
# ---------------------------------------------------------------------------
_VIS_PATCH = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, NUM_STAGES=3, THREADS=128)
_VIS_NORM = dict(BLOCK_M=1, BLOCK_K=1152, THREADS=128)
_VIS_QKV = dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, NUM_STAGES=2, THREADS=128)
_VIS_OUT_PROJ = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, NUM_STAGES=4, THREADS=256)
_VIS_FFN_UP = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, NUM_STAGES=4, THREADS=256)
_VIS_FFN_DOWN = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, NUM_STAGES=3, THREADS=128)

VISION_TOKENS = 256
VISION_DIM = 1152
VISION_FFN = 4304
PATCH_FEATURES = 3 * 14 * 14


def _vision_layer_norm(x, norm_w, norm_b, out):
    """LayerNorm over the feature axis, written into `out`."""
    M, K = x.shape
    _compiled(kernels.tl_layer_norm, M=M, K=K, EPS=1e-5, **_VIS_NORM)(x, norm_w, norm_b, out)
    return out


def vision_patch_embed(images, patch_w, patch_b, pos_emb, out):
    """Patchify, project, add bias and the positional embedding (upstream conv2d_embed_n256_1152_res).

    The 14x14x3 convolution is a GEMM over flattened patches; the positional
    embedding is a 256-row residual broadcast across views by the kernel's modulo.
    """
    views = images.shape[0]
    M = VISION_TOKENS * views
    patches = (images.view(views, 16, 14, 16, 14, 3).permute(0, 1, 3, 2, 4, 5)
               .contiguous().view(M, PATCH_FEATURES))
    kfn = _compiled(kernels.tl_matmul_bias_res_mod, M=M, N=VISION_DIM, K=PATCH_FEATURES,
                    I_MOD=VISION_TOKENS, **_VIS_PATCH)
    kfn(patches, patch_w.reshape(PATCH_FEATURES, VISION_DIM), patch_b, pos_emb, out.view(M, VISION_DIM))
    return out


def vision_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out, x_norm):
    """LayerNorm then the packed QKV projection (upstream layer_norm_QKV_matmul_n256_1152_3456_bias)."""
    M = x.shape[0] * VISION_TOKENS
    hidden = qkv_w.shape[1]
    x2, x_norm2 = x.view(M, VISION_DIM), x_norm.view(M, VISION_DIM)
    _vision_layer_norm(x2, norm_w, norm_b, x_norm2)
    _compiled(kernels.tl_matmul_bias_nows, M=M, N=hidden, K=VISION_DIM,
              **_VIS_QKV)(x_norm2, qkv_w, qkv_b, out.view(M, hidden))
    return out


def vision_out_proj_residual(x, weight, bias, res, out):
    """out = attn @ weight + bias + res (upstream matmul_n256_1152_1152_bias_res).

    Collapses upstream's masked/unmasked and split-K branches into one
    fp32-accumulating GEMM. `res` and `out` alias the same buffer.
    """
    M = x.shape[0] * VISION_TOKENS
    kfn = _compiled(kernels.tl_matmul_bias_res, M=M, N=VISION_DIM, K=VISION_DIM, **_VIS_OUT_PROJ)
    kfn(x.reshape(M, VISION_DIM), weight, bias, res.view(M, VISION_DIM), out.view(M, VISION_DIM))
    return out


def vision_norm_ffn_up(x, norm_w, norm_b, weight, bias, out, x_norm):
    """LayerNorm then the GELU feed-forward expansion (upstream layer_norm_matmul_..._bias_gelu)."""
    M = x.shape[0] * VISION_TOKENS
    x2, x_norm2 = x.view(M, VISION_DIM), x_norm.view(M, VISION_DIM)
    _vision_layer_norm(x2, norm_w, norm_b, x_norm2)
    _compiled(kernels.tl_matmul_bias_gelu, M=M, N=VISION_FFN, K=VISION_DIM,
              **_VIS_FFN_UP)(x_norm2, weight, bias, out.view(M, VISION_FFN))
    return out


def vision_ffn_down_residual(x, weight, bias, res, out):
    """out = hidden @ weight + bias + res (upstream matmul_n256_4304_1152_bias_res)."""
    M = x.shape[0] * VISION_TOKENS
    kfn = _compiled(kernels.tl_matmul_bias_res, M=M, N=VISION_DIM, K=VISION_FFN, **_VIS_FFN_DOWN)
    kfn(x.reshape(M, VISION_FFN), weight, bias, res.view(M, VISION_DIM), out.view(M, VISION_DIM))
    return out


VISION_WRAPPERS = {
    "vision_patch_embed": vision_patch_embed,
    "vision_norm_qkv": vision_norm_qkv,
    "vision_out_proj_residual": vision_out_proj_residual,
    "vision_norm_ffn_up": vision_norm_ffn_up,
    "vision_ffn_down_residual": vision_ffn_down_residual,
}


# ---------------------------------------------------------------------------
# Encoder (18 layers, RMSNorm, M = encoder_seq_len, hidden 2048, FFN 16384)
#
# Same high-occupancy regime as vision: warp specialization on throughout. The
# gated FFN here is the single largest kernel in the model.
# ---------------------------------------------------------------------------
_ENC_PROJ_NORM = dict(BLOCK_M=1, BLOCK_K=1152, THREADS=128)
_ENC_PROJ = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_STAGES=3, THREADS=256)
_ENC_RMS = dict(BLOCK_M=1, BLOCK_K=128, THREADS=128)
_ENC_QKV = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, NUM_STAGES=4, THREADS=256)
_ENC_ROPE = dict(BLOCK_M=64, BLOCK_PAIR=128, THREADS=256)
_ENC_OUT_PROJ = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, NUM_STAGES=4, THREADS=256, SWIZZLE=0)
_ENC_GATE = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_STAGES=2, THREADS=256, SWIZZLE=8)
_ENC_FFN_DOWN = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_STAGES=3, THREADS=256, SWIZZLE=8)

ENCODER_DIM = 2048


def encoder_projector(x, norm_w, norm_b, proj_w, proj_b, out, x_norm):
    """Final vision LayerNorm then the projection into encoder width (upstream layer_norm_matmul_n256_1152_2048_bias)."""
    M = x.shape[0] * VISION_TOKENS
    x2, x_norm2 = x.view(M, VISION_DIM), x_norm.view(M, VISION_DIM)
    _compiled(kernels.tl_layer_norm, M=M, K=VISION_DIM, EPS=1e-5,
              **_ENC_PROJ_NORM)(x2, norm_w, norm_b, x_norm2)
    _compiled(kernels.tl_matmul_bias, M=M, N=ENCODER_DIM, K=VISION_DIM,
              **_ENC_PROJ)(x_norm2, proj_w, proj_b, out[:M])
    return out


def encoder_norm_qkv_rope(x, weight_qkv, rope, Q, K, V, x_norm):
    """RMSNorm, QKV projection, then RoPE scattered into Q/K/V (upstream rms_matmul_n_2048_2560_qkv_rope).

    Three kernels rather than the decoder's one: the encoder normalizes x before
    the GEMM instead of folding a factor into it, and rounds to bf16 before
    rotating. Both orderings match upstream and are not interchangeable.
    """
    M, Kdim = x.shape
    N = weight_qkv.shape[1]
    head_dim = V.shape[1]
    num_heads = Q.shape[0] // M

    _compiled(kernels.tl_rms_norm, M=M, K=Kdim, **_ENC_RMS)(x, x_norm[:M])
    projected = scratch("encoder_qkv", (M, N), x.dtype, x.device)
    _compiled(kernels.tl_matmul_ws, M=M, N=N, K=Kdim, **_ENC_QKV)(x_norm[:M], weight_qkv, projected)
    _compiled(kernels.tl_rope_scatter_bf16, M=M, N=N, HEAD_DIM=head_dim, NUM_HEADS=num_heads,
              **_ENC_ROPE)(projected, rope, Q.view(M, num_heads * head_dim), K, V)


def encoder_out_proj_residual(x, weight, out):
    """out += attn @ weight, in place (upstream matmul_n_2048_2048_res)."""
    M, K = x.shape
    _compiled(kernels.tl_matmul_res_ws, M=M, N=weight.shape[1], K=K,
              **_ENC_OUT_PROJ)(x, weight, out, out)
    return out


def encoder_norm_gated_ffn(x, gate_w, up_w, out, x_norm):
    """RMSNorm then the gated feed-forward (upstream rms_matmul_n_2048_16384_gate)."""
    M, K = x.shape
    _compiled(kernels.tl_rms_norm, M=M, K=K, **_ENC_RMS)(x, x_norm[:M])
    _compiled(kernels.tl_matmul_gate, M=M, N=gate_w.shape[1], K=K,
              **_ENC_GATE)(x_norm[:M], gate_w, up_w, out[:M])
    return out


def encoder_ffn_down_residual(x, weight, out):
    """out += hidden @ weight, in place (upstream matmul_n_16384_2048_res).

    K=16384 makes the weight far larger than L2, so this one runs with the L2
    swizzle enabled.
    """
    M, K = x.shape
    _compiled(kernels.tl_matmul_res_ws, M=M, N=weight.shape[1], K=K,
              **_ENC_FFN_DOWN)(x, weight, out, out)
    return out


ENCODER_WRAPPERS = {
    "encoder_projector": encoder_projector,
    "encoder_norm_qkv_rope": encoder_norm_qkv_rope,
    "encoder_out_proj_residual": encoder_out_proj_residual,
    "encoder_norm_gated_ffn": encoder_norm_gated_ffn,
    "encoder_ffn_down_residual": encoder_ffn_down_residual,
}

ALL_WRAPPERS = {**VISION_WRAPPERS, **ENCODER_WRAPPERS, **DECODER_WRAPPERS}
