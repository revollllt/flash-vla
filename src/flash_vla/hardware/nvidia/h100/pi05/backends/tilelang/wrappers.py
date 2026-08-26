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

from .kernels import adarms as ada_kernels
from .kernels import base as kernels
from .kernels import fused_norm as fused_norm_kernels
from .kernels import xfs as xfs_kernels

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
DECODER_HEADS = 8


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


# ---------------------------------------------------------------------------
# Decoder (18 layers x 10 flow steps, M = chunk = 50 -- Pi0 had 51, the extra
# row being the state token Pi0.5 moved into the prompt)
#
# Every config is Pi0's, carried
# over unchanged and NOT re-tuned: changing the tiling and the AdaRMSNorm maths
# in one step would make a numerical failure un-bisectable, and
# `kernels.tl_scaled_gate` is documented as tiling-dependent for correctness,
# not just speed. Re-tune after the variants are proven, with `correct=`.
#
# Every shape here is under one wave and far below the H100 ridge point of
# 295 FLOP/B (21.3, 32.0 and 10.7 for the three GEMM variants), so these are
# weight-bandwidth bound: the configs favour occupancy and pipeline depth over
# big tiles.
# ---------------------------------------------------------------------------
_DEC_ACTION_IN = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, NUM_STAGES=2, THREADS=128)
# Swept at M=50, N=2560, K=1024. BLOCK_K 128 -> 256 is worth 3%; BLOCK_N below
# 32 does not compile in TileLang 0.1.11 ("unsupported shared swizzle layout"),
# which is what closes the only lever that would raise this kernel's 61% SM
# coverage. warp_spec must stay on: the AdaRMS A-tile scale is a second write to
# A_shared, which the no-WS pipeline planner rejects -- a constraint the
# unscaled tl_qkv_gemm_rope did not have.
# BLOCK_N is stuck at 32. Halving it to 16 would double the CTA count (80 -> 160)
# and fill the machine, but TileLang 0.1.11 rejects it: a 16-wide bf16 W tile is
# 32 B per row and TMA's swizzle needs >= 64 B, so the W_shared copy fails to
# lower ("unsupported shared swizzle layout"). Measured, not assumed -- a sweep
# of BLOCK_N in (8, 16, 32) compiled only 32. BLOCK_K went 128 -> 256, a 3% edge
# over the Pi0 inheritance. Getting past 80 CTAs needs split-K or a TMA-disabled
# W load, both v2.
_DEC_QKV = dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=4, THREADS=128)
_DEC_RESIDUAL = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=4, THREADS=128)
_DEC_GATE = dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=3, THREADS=128)
_DEC_OUT_PROJ = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=3, THREADS=128, PRO_K=128)
_DEC_RMS = dict(BLOCK_M=2, BLOCK_K=256, THREADS=128)
_DEC_XFS = dict(BLOCK_M=8, BLOCK_K=256, OUTPUT_K=32, THREADS=128, M_PAD=64)

# NUM_SPLIT is a request, not the realized count -- `_num_splits` shrinks it.
# Pi0's 7 realizes as 6 at Pi0.5's 1018 keys; 8 realizes as 8 and measured
# fastest (10.25 us against 11.04 us) in a sweep of the split/combine pair, which
# has to be swept together because more splits speeds up one and slows the other.
_FD_SPLIT = dict(BLOCK_M=64, BLOCK_N=64, NUM_SPLIT=8, NUM_STAGES=1, THREADS=128)
_FD_COMBINE_BLOCK_M = 2
_LOG2E = 1.4426950408889634


def _rms_factor(x, out, cfg=_DEC_RMS):
    """Write rsqrt(mean(x^2)+eps) into `out`, which the consuming GEMM then scales by.

    Pi0's fused path folded this into `tl_fused_rms_gate`. Pi0.5 cannot: that
    kernel accumulates the row sum of squares from the same shared tile the GEMM
    consumes, and AdaRMSNorm needs the tile unscaled for the norm and scaled for
    the GEMM. v1 pays the extra launch; see the spec for what v2 would need.
    """
    M, K = x.shape
    _compiled(kernels.tl_rms_factor, M=M, K=K, **cfg)(x, out)
    return out


def decoder_rms_xfs(x, scale, out):
    """Write the next FFN's exact BF16 input as contiguous ``[1024,64]``.

    ``x`` is the BF16 ``decoder_x`` *after* ``decoder_out_proj_residual`` has
    applied its gated residual update.  This replaces ``_rms_factor`` for the
    persistent GatedProjection path; neither the row factor nor a row-major
    normalized activation is materialized.
    """
    M, K = x.shape
    if (M != 50 or K != 1024 or tuple(scale.shape) != (1024,)
            or tuple(out.shape) != (1024, 64)):
        raise ValueError(
            "decoder_rms_xfs requires x[50,1024], scale[1024], out[1024,64]")
    if (x.dtype != torch.bfloat16 or scale.dtype != torch.bfloat16
            or out.dtype != torch.bfloat16):
        raise ValueError("decoder_rms_xfs tensors must be BF16")
    if not x.is_contiguous() or not scale.is_contiguous() or not out.is_contiguous():
        raise ValueError("decoder_rms_xfs tensors must be contiguous")
    _compiled(xfs_kernels.tl_rms_xfs_kmajor, M=M, K=K, **_DEC_XFS)(x, scale, out)
    return out


def decoder_action_in_proj(x, weight, bias, out):
    """out = x @ weight + bias.

    Pi0 fused the timestep into this projection and followed it with a second
    MLP layer; Pi0.5's `action_in_proj` is a bare linear with no activation,
    because the timestep now arrives through AdaRMSNorm instead.
    """
    M, K = x.shape
    _compiled(kernels.tl_matmul_bias, M=M, N=weight.shape[1], K=K,
              **_DEC_ACTION_IN)(x, weight, bias, out)
    return out


def decoder_norm_qkv_rope(x, scale, weight_qkv, bias, rope, Q, K, V, norm_factor):
    """AdaRMS-scale x, project to QKV, add the shift bias, apply RoPE, scatter in place."""
    M, Kdim = x.shape
    head_dim = V.shape[1]
    num_heads = Q.shape[0] // M
    factor = _rms_factor(x, norm_factor[:M])
    kfn = _compiled(ada_kernels.tl_ada_qkv_gemm_rope, M=M, N=weight_qkv.shape[1], K=Kdim,
                    HEAD_DIM=head_dim, NUM_HEADS=num_heads, **_DEC_QKV)
    kfn(x, factor, scale, weight_qkv, bias, rope, Q.view(M, num_heads * head_dim), K, V)


def decoder_out_proj_residual(x, weight, gate, out):
    """out = out + (x @ weight) * gate."""
    M, K = x.shape
    _compiled(ada_kernels.tl_matmul_gated_res, M=M, N=weight.shape[1], K=K,
              **_DEC_RESIDUAL)(x, weight, gate, out, out)
    return out


def decoder_ffn_down_residual(x, weight, gate, out):
    """out = out + (x @ weight) * gate. Same kernel as the out-projection, larger K."""
    M, K = x.shape
    _compiled(ada_kernels.tl_matmul_gated_res, M=M, N=weight.shape[1], K=K,
              **_DEC_RESIDUAL)(x, weight, gate, out, out)
    return out


def decoder_norm_gated_ffn(x, scale, gate_w, up_w, gate_b, up_b, out, norm_factor):
    """out = gelu(ada(x) @ gate_w + gate_b) * (ada(x) @ up_w + up_b)."""
    M, K = x.shape
    factor = _rms_factor(x, norm_factor[:M])
    _compiled(ada_kernels.tl_ada_scaled_gate, M=M, N=gate_w.shape[1], K=K,
              **_DEC_GATE)(x, factor, scale, gate_w, up_w, gate_b, up_b, out)
    return out


def decoder_action_out_proj(x, weight, bias, out, norm_factor):
    """out += bias + rms(x) @ weight, with the final AdaRMSNorm folded into both.

    Unchanged from Pi0's fused kernel: the final norm's scale, its shift and the
    Euler dt all fold into per-step `decoder_action_out_proj_w` and `_b` at
    checkpoint load, which leaves exactly this signature. `norm_factor` is
    accepted for signature parity and never written -- the factor exists only
    inside the kernel.
    """
    M, K = x.shape
    _compiled(fused_norm_kernels.tl_fused_rms_matmul_bias_res, M=M, N=weight.shape[1], K=K,
              **_DEC_OUT_PROJ)(x, weight, bias, out, out)
    return out


def _num_splits(keys: int, block_n: int, requested: int) -> tuple[int, int]:
    """Largest split count <= `requested` for which no split starts past `keys`.

    An empty split is not merely wasted work. A TMA box whose first row is out of
    bounds reads garbage rather than zeroes, and that garbage becomes inf, then
    NaN, in the score initialization; it then survives the running-max update,
    because fmax(NaN, x) hides it. At Pi0.5's keys=1018 this selects 6.
    """
    def chunk_blocks(n: int) -> int:
        return ((keys + n - 1) // n + block_n - 1) // block_n

    while requested > 1 and (requested - 1) * chunk_blocks(requested) * block_n >= keys:
        requested -= 1
    return requested, chunk_blocks(requested)


def decoder_attention(Q, K, V, mask, out):
    """FlashDecoding attention with an additive per-key mask.

    One implementation, not two. Pi0 kept a three-kernel scores/softmax/attn@V
    chain as the readable reference beside the fused pair; Pi0.5 v1 has only the
    fused path, so a second masked softmax does not have to be written and kept
    in agreement with this one.

    `out` aliases Q; the split kernel only reads Q and the combine's write is
    ordered after it on the stream.
    """
    M, head_dim = Q.shape
    keys = K.shape[0]
    block_m, block_n = _FD_SPLIT["BLOCK_M"], _FD_SPLIT["BLOCK_N"]
    assert M % _FD_COMBINE_BLOCK_M == 0, "combine tile must divide the flat query rows"

    num_split, chunk_blocks = _num_splits(keys, block_n, _FD_SPLIT["NUM_SPLIT"])
    q_pad = (M + block_m - 1) // block_m * block_m
    config = dict(_FD_SPLIT, NUM_SPLIT=num_split)

    partial = scratch("fd_partial", (num_split, q_pad, head_dim), Q.dtype, Q.device)
    glse = scratch("fd_glse", (num_split, q_pad), torch.float32, Q.device)

    split_fn = _compiled(ada_kernels.tl_fd_flat_split_mask, M=M, HD=head_dim, KEYS=keys,
                         QPAD=q_pad,
                         CHUNK=chunk_blocks * block_n, CHUNK_BLOCKS=chunk_blocks,
                         SCALE_L2=float(head_dim ** -0.5) * _LOG2E, **config)
    combine_fn = _compiled(kernels.tl_fd_flat_combine, M=M, HD=head_dim, QPAD=q_pad,
                           NUM_SPLIT=num_split, BLOCK_M=_FD_COMBINE_BLOCK_M, THREADS=128)
    split_fn(Q, K, V, mask, partial, glse)
    combine_fn(partial, glse, out)
    return out


DECODER_WRAPPERS = {
    "decoder_action_in_proj": decoder_action_in_proj,
    "decoder_norm_qkv_rope": decoder_norm_qkv_rope,
    "decoder_attention": decoder_attention,
    "decoder_out_proj_residual": decoder_out_proj_residual,
    "decoder_rms_xfs": decoder_rms_xfs,
    "decoder_norm_gated_ffn": decoder_norm_gated_ffn,
    "decoder_ffn_down_residual": decoder_ffn_down_residual,
    "decoder_action_out_proj": decoder_action_out_proj,
}


ALL_WRAPPERS = {**VISION_WRAPPERS, **ENCODER_WRAPPERS, **PROMPT_WRAPPERS,
                **DECODER_WRAPPERS}
