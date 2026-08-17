"""Fused decoder wrappers: fewer, larger kernels for the same maths.

Each function here replaces the like-named wrapper in `wrappers.py` with an
identical signature, so selecting between them is a table lookup (`ops.py`).
Two fusions, both decoder-only:

Lazy pre-norm. RMS normalization commutes with the GEMM's reduction --
rms(x) @ W equals (x @ W) * rstd(x)[:, None] -- so the scale factor does not
have to exist before the GEMM starts. These kernels compute the row sum of
squares inside the mainloop, from the same shared-memory tile the GEMM is
already consuming, and apply the factor to the fp32 accumulator in the epilogue.
That removes one kernel launch per call site, and removes a serializing bf16
scale from the mainloop, so the fused kernel beats even the unfused GEMM alone.

FlashDecoding attention. The three-kernel scores/softmax/attn@V chain becomes a
split plus a combine, and the (queries, keys) score matrix never leaves SRAM.

Neither fusion applies outside the decoder. On the encoder's shapes the norm
prologue costs more than it saves: those GEMMs are already bound on the shared
memory datapath at one CTA per SM, so there are no spare warps to absorb the
extra traffic. The decoder is the opposite -- latency-bound with idle issue
slots, which the sum-of-squares stream fills for free.

Numerically the fused kernels are not bit-identical to the two-kernel path: they
scale the fp32 accumulator instead of rounding x*factor to bf16 per element, so
they are strictly closer to the fp32 reference.
"""
from __future__ import annotations

import torch

from .kernels import base as kernels
from .kernels import fused_norm as fused_norm_kernels
from .wrappers import _compiled, scratch

_FUSED_GATE = dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=3, THREADS=128, PRO_K=64)
_FUSED_OUT_PROJ = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=3, THREADS=128, PRO_K=128)

_FD_SPLIT = dict(BLOCK_M=64, BLOCK_N=64, NUM_SPLIT=7, NUM_STAGES=1, THREADS=128)
_FD_COMBINE_BLOCK_M = 2
_LOG2E = 1.4426950408889634
DECODER_HEADS = 8


def decoder_norm_gated_ffn(x, gate_w, up_w, out, norm_factor):
    """Gated FFN with the RMS factor computed inside the kernel.

    `norm_factor` is accepted for signature parity but never written -- the
    factor exists only inside the kernel, and nothing downstream reads it.
    """
    M, K = x.shape
    _compiled(fused_norm_kernels.tl_fused_rms_gate, M=M, N=gate_w.shape[1], K=K,
              **_FUSED_GATE)(x, gate_w, up_w, out)
    return out


def decoder_action_out_proj(x, weight, bias, out, norm_factor):
    """Out-projection with the RMS factor computed inside the kernel, written in place.

    The unfused path allocates a result and copies it back, one extra
    device-to-device graph node per call; here `out` is both the residual input
    and the destination.
    """
    M, K = x.shape
    _compiled(fused_norm_kernels.tl_fused_rms_matmul_bias_res, M=M, N=weight.shape[1], K=K,
              **_FUSED_OUT_PROJ)(x, weight, bias, out, out)
    return out


def _num_splits(keys: int, block_n: int, requested: int) -> tuple[int, int]:
    """Largest split count <= `requested` for which no split starts past `keys`.

    An empty split is not merely wasted work. A TMA box whose first row is out of
    bounds reads garbage rather than zeroes -- only the tail of a box that starts
    in bounds is zero-filled -- and that garbage becomes inf, then NaN, in the
    score initialization. It then survives the running-max update, because
    fmax(NaN, x) hides it. At keys=819 this leaves the tuned 7 splits untouched;
    at keys=1075 it selects 6.
    """
    def chunk_blocks(n: int) -> int:
        return ((keys + n - 1) // n + block_n - 1) // block_n

    while requested > 1 and (requested - 1) * chunk_blocks(requested) * block_n >= keys:
        requested -= 1
    return requested, chunk_blocks(requested)


def decoder_attention(Q, K, V, scores, out, encoder_seq_len):
    """FlashDecoding attention: split over keys, then merge by log-sum-exp.

    `scores` is accepted for signature parity and left untouched -- the score
    matrix stays in SRAM. `out` aliases Q; the split kernel only reads Q and the
    combine's write is ordered after it on the stream.
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

    split_fn = _compiled(kernels.tl_fd_flat_split, M=M, HD=head_dim, KEYS=keys,
                         ENC_LEN=encoder_seq_len, NUM_HEADS=DECODER_HEADS, QPAD=q_pad,
                         CHUNK=chunk_blocks * block_n, CHUNK_BLOCKS=chunk_blocks,
                         SCALE_L2=float(head_dim ** -0.5) * _LOG2E, **config)
    combine_fn = _compiled(kernels.tl_fd_flat_combine, M=M, HD=head_dim, QPAD=q_pad,
                           NUM_SPLIT=num_split, BLOCK_M=_FD_COMBINE_BLOCK_M, THREADS=128)
    split_fn(Q, K, V, partial, glse)
    combine_fn(partial, glse, out)
    return out


FUSED_WRAPPERS = {
    "decoder_norm_gated_ffn": decoder_norm_gated_ffn,
    "decoder_action_out_proj": decoder_action_out_proj,
    "decoder_attention": decoder_attention,
}
