"""Per-call-site wrappers: one function per operation in the Pi0.5 forward pass.

Each wrapper owns the tile configuration for its call site, compiles the kernel
once per shape, and calls it. Kernels are shared between call sites; configs are
not -- the same GEMM at M=50 and M=968 wants different tiles and a different
warp-specialization setting, which is why some kernels appear here twice through
two variants.

The configs below are Pi0's, carried over unchanged and **not yet re-tuned**
for this target. Every encoder call site moved from M=768 to M=968 when the
state joined the prompt, which is exactly the kind of shape change the Pi0
wrappers warn about: re-tune through `autotune.sweep_kernel` and pass
`correct=` -- `kernels.tl_scaled_gate` is numerically sensitive to its tiling
and some tilings produce garbage rather than failing, so timing alone will
happily rank a wrong config first. Vision is untouched at M=768 and its configs
are still the tuned ones.

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

# ---------------------------------------------------------------------------
# Prompt embedding
#
# Pi0's prompt is fixed at load time, so its embeddings are a checkpoint tensor
# copied into `encoder_x` once. Pi0.5's prompt carries the discretized state and
# changes every call, so the table stays resident (257152 x 2048, 1.05 GB bf16)
# and the target gathers 200 rows out of it -- 0.8 MB read, well under a
# microsecond of traffic. The cost that matters is node count, not bandwidth,
# which is why this is two ops rather than three: `prompt_embed_scale` carries
# sqrt(width) on valid rows and zero on padding, so one multiply both applies
# the embedder's scale and zeroes the padded rows.
#
# Torch rather than TileLang on purpose: it is a gather and a broadcast
# multiply, both already at the roofline, and keeping the ids on the host side
# of a readable op keeps them reachable from the parity gate. It stays an
# op-table entry so a fused TileLang version can replace it per call site later.
# ---------------------------------------------------------------------------


def encoder_embed_prompt(token_ids, table, scale, out):
    """out = table[token_ids] * scale, into the language rows of `encoder_x`."""
    torch.index_select(table, 0, token_ids.to(torch.int32), out=out)
    out.mul_(scale)
    return out


PROMPT_WRAPPERS = {
    "encoder_embed_prompt": encoder_embed_prompt,
}


ALL_WRAPPERS = {**VISION_WRAPPERS, **ENCODER_WRAPPERS, **PROMPT_WRAPPERS}
