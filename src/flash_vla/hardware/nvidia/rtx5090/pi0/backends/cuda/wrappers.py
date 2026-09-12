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

import os
from functools import partial

import torch

from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang.wrappers import (
    OPS, ROUTE_CONSTRAINTS)

from . import cutlass_gemm as cg
from . import pointwise as cu

#: Gate and up projections concatenated into one (dim, 2*ffn) weight, keyed by
#: the pair of source tensors. A skinny GEMM on this part costs about 8 us
#: before it reads a byte, and the expert's 51 x 1024 x 8192 packed form
#: measured 10.31 us against 10.34 for the 4096-wide half -- so the second half
#: of the weights rides along and the fixed cost is paid once.
#:
#: Built on first call, which the runner makes during warmup, before capture.
#: `torch.cat` allocates, so a cache miss inside a capture would be illegal;
#: the wrapper falls back to two GEMMs there rather than risking it.
_PACKED_GATE_UP: dict[tuple[int, int], torch.Tensor] = {}


def _packed_gate_up(gate_w, up_w):
    """The two weights as one contiguous (dim, 2*ffn), or None inside a capture."""
    key = (gate_w.data_ptr(), up_w.data_ptr())
    packed = _PACKED_GATE_UP.get(key)
    if packed is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        packed = torch.cat((gate_w, up_w), dim=1).contiguous()
        _PACKED_GATE_UP[key] = packed
    return packed

# --------------------------------------------------------------------------
# The route's GEMMs run on CUTLASS stream-K, not cuBLAS.
#
# Speed is half the reason -- 10 of the 11 shapes are at or above cuBLAS and
# four are 1.2x to 1.5x, because stream-K fills a 170-SM part that no power-of-
# two tile grid lands on. The other half is control: programmatic dependent
# launch needs the producer AND the consumer to carry `griddepcontrol`, and a
# cuBLAS call between two hand-written kernels breaks the chain.
#
# The tile per site is measured, not derived: stream-K removes the wave
# quantization but not the mainloop's own trade, where a big tile is more
# efficient per SM and a small one puts more CTAs on the part. The sweep is
# `lab/sm120/pi0_cutlass_sweep.py` and these are its picks.
# --------------------------------------------------------------------------
GEMM_CONFIG = {
    "vision_qkv": 5,
    "vision_out_proj": 10,
    "vision_ffn_up": 5,
    "vision_ffn_down": 10,
    "backbone_qkv": 10,
    "backbone_gate": 5,
    "backbone_up": 5,
    "backbone_out_proj": 10,
    "backbone_ffn_down": 5,
    "expert_gate_up": 3,
    "expert_ffn_down": 8,
    "expert_out_proj": 9,
}


#: Only the call sites this backend implements. A plan naming any other site
#: for this backend is a routing error and `make_wrappers` says so.
NAMES = frozenset({
    "action_expert_norm_qkv_rope",
    "action_expert_norm_gated_ffn",
    "action_expert_attention",
    "action_expert_action_out_proj",
    "llm_backbone_norm_qkv_rope",
    "llm_backbone_norm_gated_ffn",
    "action_expert_out_proj_residual",
    "action_expert_ffn_down_residual",
    "llm_backbone_out_proj_residual",
    "llm_backbone_ffn_down_residual",
    "llm_backbone_projector",
    "vision_encoder_norm_qkv",
    "vision_encoder_norm_ffn_up",
    "vision_encoder_out_proj_residual",
    "vision_encoder_ffn_down_residual",
})

#: Pi0's vision tower: 3 views x 256 patches, 1152 wide, 4304 feed-forward.
VISION_TOKENS = 256
VISION_DIM = 1152
VISION_FFN = 4304


def action_expert_norm_qkv_rope(x, scale, weight_qkv, bias, rope, Q, K, V,
                                norm_factor):
    """RMS-scale x, project to QKV, rotate and scatter -- one launch.

    `x` is (tokens, dim) bf16; `weight_qkv` is (dim, heads*head_dim + 2*head_dim).
    Q is (tokens*heads, head_dim), K and V are (tokens, head_dim), all written in
    place. `scale` and `bias` are Pi0.5's AdaRMS terms and are None here.
    Safe during CUDA-graph capture.

    The normalized activations are never materialized. Pi0's expert RMSNorm has
    no learnable gain, so it is a per-row scalar that commutes with the
    projection, and the kernel folds it into its epilogue -- which is also one
    fewer bf16 rounding than the three-launch form.
    """
    assert scale is None and bias is None, "Pi0 has no AdaRMS scale or shift"
    cu.expert_qkv(x, weight_qkv, rope, Q, K, V)


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
    cu.rms_norm(x, normed)
    packed_w = _packed_gate_up(gate_w, up_w)
    if packed_w is None:
        gate = scratch("expert_gate", (m, ffn), x.dtype, x.device)
        torch.mm(normed, gate_w, out=gate)
        torch.mm(normed, up_w, out=out)
        cu.gelu_mul(gate, out, out)
        return out
    both = scratch("expert_gate_up", (m, 2 * ffn), x.dtype, x.device)
    if not cg.gemm(normed, packed_w, both, config=GEMM_CONFIG["expert_gate_up"]):
        torch.mm(normed, packed_w, out=both)
    cu.gelu_mul_packed(both, out)
    return out


def action_expert_attention(Q, K, V, mask, out, prefix_len, *, scratch):
    """out = softmax(mask(Q @ K^T * scale)) @ V, multi-query, one fused kernel.

    `Q` and `out` are (queries, head_dim) bf16 and MAY ALIAS: a CTA reads its
    own sixteen query rows into shared memory before anything is written, and
    writes only those same rows. `K` and `V` are (keys, head_dim). Safe during
    CUDA-graph capture.

    The kernel splits the KEY axis eight ways. 408 flat queries is 26 mma M
    tiles and therefore 26 CTAs on a 170-SM part; the key axis is the only
    other source of parallelism, and the fp32 partials it costs stay in L2.
    Measured against the cuBLAS-plus-softmax form it replaces, at Pi0's shape:

        splits   CTAs   fused    vs split form
             1     26   33.34 us      0.76x
             4    104   21.12 us      1.19x
             8    208   19.04 us      1.32x
            12    312   20.90 us      1.21x

    A CUDA-core version of this fusion is in `kernels/expert_attention.cu` and
    is 8x slower than the split form; the instruction count is the wall there
    and only `mma.sync` clears it.
    """
    assert mask is None, "Pi0 has no key mask; the prefix length is the integer"
    if cu.expert_attention_mma(Q, K, V, out, heads=DECODER_HEADS,
                               prefix=prefix_len, splits=ATTENTION_SPLITS):
        return out
    # The split-KV workspace is built on the first call, which the runner makes
    # during warmup; a miss inside a capture falls back rather than allocating.
    queries, head_dim = Q.shape
    keys = K.shape[0]
    scores = scratch("expert_scores", (queries, keys), Q.dtype, Q.device)
    torch.mm(Q, K.t(), out=scores)
    cu.expert_masked_softmax(scores, scores, heads=DECODER_HEADS,
                             prefix=prefix_len, scale=float(head_dim ** -0.5))
    torch.mm(scores, V, out=out)
    return out


#: Slices of the key axis, swept at Pi0's shape; see the wrapper above.
ATTENTION_SPLITS = 8


#: Pi0's action expert is multi-query: eight query heads over one KV head, so
#: the flat query axis is (token, head) and the first `DECODER_HEADS` rows are
#: the state token's.
DECODER_HEADS = 8

def llm_backbone_norm_qkv_rope(x, weight_qkv, rope, Q, K, V, x_norm, *, scratch):
    """RMSNorm, QKV projection, rotate and scatter -- the backbone's three launches.

    Same three kernels as the expert's form on a different shape: 768 prefix
    rows of 2048 rather than 51 of 1024. The backbone normalises x BEFORE the
    GEMM where the expert folds a scale into it; both match upstream and are not
    interchangeable, which is why this is a separate wrapper rather than a
    shared one.

    `x` is (tokens, dim) bf16, `x_norm` a caller buffer at least that size, `Q`
    is (tokens*heads, head_dim), `K` and `V` are (tokens, head_dim). Written in
    place; safe during CUDA-graph capture.
    """
    m = x.shape[0]
    n = weight_qkv.shape[1]
    normed = x_norm[:m]
    packed = scratch("backbone_qkv", (m, n), x.dtype, x.device)
    cu.rms_norm(x, normed)
    if not cg.gemm(normed, weight_qkv, packed,
                   config=GEMM_CONFIG["backbone_qkv"]):
        torch.mm(normed, weight_qkv, out=packed)
    cu.rope_scatter(packed, rope, Q, K, V)


def llm_backbone_norm_gated_ffn(x, gate_w, up_w, out, x_norm, *, scratch):
    """RMSNorm then the gated feed-forward -- four launches.

    `x` is (tokens, dim) bf16, the weights (dim, ffn), `out` at least
    (tokens, ffn) and `x_norm` at least (tokens, dim). Written in place; safe
    during CUDA-graph capture.
    """
    # NOT packed, unlike the expert's form. Packing pays where a skinny GEMM's
    # fixed cost dominates; the backbone's 768 x 2048 x 16384 is compute-bound
    # at 82-85% of this machine's measured tensor ceiling, and packing measured
    # 1.01x there against 2.45x on the expert. It would also cost 2.4 GB of
    # duplicated weights for nothing.
    m, kdim = x.shape
    ffn = gate_w.shape[1]
    normed = x_norm[:m]
    gate = scratch("backbone_gate", (m, ffn), x.dtype, x.device)
    cu.rms_norm(x, normed)
    up = out[:m]
    if not (cg.gemm(normed, gate_w, gate, config=GEMM_CONFIG["backbone_gate"])
            and cg.gemm(normed, up_w, up, config=GEMM_CONFIG["backbone_up"])):
        torch.mm(normed, gate_w, out=gate)
        torch.mm(normed, up_w, out=up)
    cu.gelu_mul(gate, up, up)
    return out


# --------------------------------------------------------------------------
# Residual projections: fold the add into the GEMM's beta.
#
# `out += x @ w` is two kernels in torch and one cuBLAS call at beta = 1. No
# hand-written kernel is involved and the arithmetic is the same sum in a
# different order, so these ride along with the rest of the route rather than
# being their own iteration.
# --------------------------------------------------------------------------
def action_expert_out_proj_residual(x, weight, gate, out):
    """out += x @ weight, in place. `gate` is Pi0.5's and is None here."""
    assert gate is None, "the gate is Pi0.5's"
    flat = out.view(x.shape[0], -1)
    _residual_gemm(x, weight, flat, "expert_out_proj")
    return out


def action_expert_ffn_down_residual(x, weight, gate, out):
    """out += x @ weight, in place. `gate` is Pi0.5's and is None here."""
    assert gate is None, "the gate is Pi0.5's"
    flat = out.view(x.shape[0], -1)
    _residual_gemm(x, weight, flat, "expert_ffn_down")
    return out


def llm_backbone_out_proj_residual(x, weight, out):
    """out += attn @ weight, in place -- one launch.

    The hand-written GEMM takes this one: 43.23 us against cuBLAS's 49.33 at
    768 x 2048 x 2048, because a 64 x 64 tile puts 384 CTAs on the output where
    cuBLAS's 128 x 128 puts 96 on a 170-SM part. The residual rides in its
    epilogue, aliased to the destination, which is safe because each element is
    read by the one thread that writes it.
    """
    flat = out.view(x.shape[0], -1)
    _residual_gemm(x, weight, flat, "backbone_out_proj")
    return out


def llm_backbone_ffn_down_residual(x, weight, out):
    """out += hidden @ weight, in place."""
    flat = out.view(x.shape[0], -1)
    _residual_gemm(x, weight, flat, "backbone_ffn_down")
    return out


# --------------------------------------------------------------------------
# Vision tower and projector: LayerNorm in one kernel, bias folded into the GEMM.
# --------------------------------------------------------------------------
def llm_backbone_projector(x, norm_w, norm_b, proj_w, proj_b, out, x_norm):
    """LayerNorm the vision output, then project into encoder width.

    `x` and `x_norm` are (views, tokens, vision_dim) bf16, `out` at least
    (views*tokens, encoder_dim). Written in place; safe during graph capture.
    """
    m = x.shape[0] * VISION_TOKENS
    x2, n2 = x.view(m, VISION_DIM), x_norm.view(m, VISION_DIM)
    cu.layer_norm(x2, norm_w, norm_b, n2)
    torch.addmm(proj_b, n2, proj_w, beta=1, alpha=1, out=out[:m])
    return out


def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out, *, scratch):
    """LayerNorm then the packed QKV projection -- two launches."""
    m = x.shape[0] * VISION_TOKENS
    x2 = x.view(m, VISION_DIM)
    normed = scratch("vision_norm", (m, VISION_DIM), x.dtype, x.device)
    cu.layer_norm(x2, norm_w, norm_b, normed)
    flat = out.view(m, qkv_w.shape[1])
    if not cg.gemm(normed, qkv_w, flat, config=GEMM_CONFIG["vision_qkv"],
                   c=qkv_b, beta=1.0, broadcast_c=True):
        torch.addmm(qkv_b, normed, qkv_w, beta=1, alpha=1, out=flat)
    return out


def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out, *, scratch):
    """LayerNorm then the GELU feed-forward expansion -- three launches."""
    m = x.shape[0] * VISION_TOKENS
    x2 = x.view(m, VISION_DIM)
    normed = scratch("vision_norm", (m, VISION_DIM), x.dtype, x.device)
    cu.layer_norm(x2, norm_w, norm_b, normed)
    flat = out.view(m, VISION_FFN)
    if not cg.gemm(normed, weight, flat, config=GEMM_CONFIG["vision_ffn_up"],
                   c=bias, beta=1.0, broadcast_c=True):
        torch.addmm(bias, normed, weight, beta=1, alpha=1, out=flat)
    cu.gelu_(flat)
    return out


def _residual_gemm(x, weight, out_flat, key):
    """out_flat = res + x @ weight, where `res` is already in `out_flat`.

    The residual goes in as the epilogue's C, aliased to D. That is the same
    aliasing `_proj_bias_residual` documents and the reason it exists: adding
    the residual afterwards reads a buffer the GEMM has already overwritten.
    """
    a = x.view(-1, weight.shape[0])
    if not cg.gemm(a, weight, out_flat, config=GEMM_CONFIG[key], c=out_flat,
                   beta=1.0):
        torch.addmm(out_flat, a, weight, beta=1, alpha=1, out=out_flat)
    return out_flat


def _proj_bias_residual(x, weight, bias, res, out, key):
    """out = x @ weight + bias + res -- two launches, alias-safe.

    `res` MAY BE `out`: the graph binds the residual to the buffer being written
    on these call sites. So the residual goes in as the GEMM's beta term rather
    than being added afterwards -- adding it afterwards reads a `res` the GEMM
    has already overwritten, which measured cos 0.962 against the torch form.
    The bias is added after, which is safe because by then nothing needs `res`.
    """
    flat = out.view(-1, weight.shape[1])
    a = x.view(-1, weight.shape[0])
    r = res.view_as(flat)
    if not cg.gemm(a, weight, flat, config=GEMM_CONFIG[key], c=r, beta=1.0):
        torch.addmm(r, a, weight, beta=1, alpha=1, out=flat)
    flat.add_(bias)
    return out


def vision_encoder_out_proj_residual(x, weight, bias, res, out):
    """out = attn @ weight + bias + res -- two launches."""
    return _proj_bias_residual(x, weight, bias, res, out, "vision_out_proj")


def vision_encoder_ffn_down_residual(x, weight, bias, res, out):
    """out = hidden @ weight + bias + res -- two launches."""
    return _proj_bias_residual(x, weight, bias, res, out, "vision_ffn_down")


# The action tokens' output projection. What it costs is launches, not
# arithmetic: the torch spelling takes eleven of them because it writes through
# a fresh temporary for the norm, the projection and the bias, and the site
# measured 15.14 us per call against a 3.5 us ceiling.
#
# The three sibling projections were measured the same way and are NOT routed
# here. `action_expert_state_proj` and `action_expert_action_in_proj` came back
# inside the noise floor, +0.015 and +0.007 ms on a 31.1 ms forward, so the
# launches they save do not show. `action_expert_action_mlp` as a single
# `torch.addmm` was 0.202 ms SLOWER than torch's mm-add-copy at its 50 x 1024 x
# 1024 shape, which is a larger effect than the launches it removes; whatever
# cuBLAS picks for that shape through `addmm` is worse than what `mm` gets.


def action_expert_action_out_proj(x, weight, bias, out, norm_factor, *, scratch):
    """out += rms(x) @ weight + bias -- three launches.

    `out` is the residual as well as the destination, so it goes in as the
    GEMM's beta term for the same aliasing reason `_proj_bias_residual`
    documents.
    """
    m, kdim = x.shape
    normed = scratch("action_out_norm", (m, kdim), x.dtype, x.device)
    cu.rms_norm(x, normed)
    flat = out.view(-1, weight.shape[1])
    torch.addmm(flat, normed, weight, beta=1, alpha=1, out=flat)
    flat.add_(bias)
    return out


ALL_WRAPPERS = {
    "action_expert_norm_qkv_rope": action_expert_norm_qkv_rope,
    "action_expert_out_proj_residual": action_expert_out_proj_residual,
    "action_expert_ffn_down_residual": action_expert_ffn_down_residual,
    "llm_backbone_out_proj_residual": llm_backbone_out_proj_residual,
    "llm_backbone_ffn_down_residual": llm_backbone_ffn_down_residual,
    "llm_backbone_projector": llm_backbone_projector,
    "vision_encoder_norm_qkv": vision_encoder_norm_qkv,
    "vision_encoder_norm_ffn_up": vision_encoder_norm_ffn_up,
    "vision_encoder_out_proj_residual": vision_encoder_out_proj_residual,
    "vision_encoder_ffn_down_residual": vision_encoder_ffn_down_residual,
    "llm_backbone_norm_qkv_rope": llm_backbone_norm_qkv_rope,
    "llm_backbone_norm_gated_ffn": llm_backbone_norm_gated_ffn,
    "action_expert_norm_gated_ffn": action_expert_norm_gated_ffn,
    "action_expert_attention": action_expert_attention,
    "action_expert_action_out_proj": action_expert_action_out_proj,
}
#: Only the wrappers that stage an intermediate take the allocator; the rest
#: write straight into the caller's buffers.
_TAKES_SCRATCH = ("action_expert_norm_gated_ffn",
                  "action_expert_attention", "llm_backbone_norm_qkv_rope",
                  "llm_backbone_norm_gated_ffn", "vision_encoder_norm_qkv",
                  "vision_encoder_norm_ffn_up", "action_expert_action_out_proj")


#: FLASH_VLA_PDL=0 turns programmatic dependent launch off for the A/B. The
#: trigger POSITION is a separate, compile-time knob; see
#: `pointwise.FLASH_VLA_PDL_TRIGGER`.
PDL_ENABLED = os.environ.get("FLASH_VLA_PDL", "1") != "0"


def make_wrappers(scratch, selected_names=None) -> dict:
    """The wrappers of `selected_names` (default: all), bound to `scratch`."""
    cu.set_pdl(PDL_ENABLED)
    cg.set_pdl(PDL_ENABLED)
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - NAMES
    if unknown:
        raise KeyError(f"the rtx5090 cuda backend does not implement {sorted(unknown)}")
    return {name: (partial(ALL_WRAPPERS[name], scratch=scratch)
                   if name in _TAKES_SCRATCH else ALL_WRAPPERS[name])
            for name in names}


__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "make_wrappers"]
