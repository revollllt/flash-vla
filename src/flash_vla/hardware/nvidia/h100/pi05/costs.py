"""Cost declarations of the H100/Pi0.5 call sites, per segment.

Minimal traffic and math at the Target's shapes: each activation read once,
each weight read once, each output written once, a residual read where the
call site accumulates into it. Nothing here knows which backend runs a call
site; that is what makes the roofline tier of the floor model plan-independent.
The invocation counts mirror `pipeline.py`: the last encoder layer runs only
its QKV projection, and the decoder body runs `steps x layers` times.
"""
from __future__ import annotations

from flash_vla.models.pi05.spec import (
    ACTION_DIM,
    DECODER_DIM,
    DECODER_FFN,
    DECODER_HEADS,
    ENCODER_DIM,
    ENCODER_FFN,
    HEAD_DIM,
    QKV_WIDTH,
    VISION_DIM,
    VISION_FFN,
    VISION_LAYERS,
    VISION_TOKENS,
)
from flash_vla.runtime.cost import BF16, Cost, Invocation, SegmentCosts, attention, dual_gemm, gemm

#: SigLIP So400m/14 runs 16 heads of width 72 over 256 tokens per view.
VISION_HEADS = 16
VISION_HEAD_DIM = VISION_DIM // VISION_HEADS
#: 14 x 14 x 3 input channels per patch.
PATCH = 14 * 14 * 3


def segment_costs(num_views: int, chunk_size: int, steps: int, layers: int,
                  prompt_len: int) -> SegmentCosts:
    """The three segments' invocations at this shape profile."""
    tokens = num_views * VISION_TOKENS
    seq = tokens + prompt_len
    keys = seq + chunk_size
    m = chunk_size

    vision = [
        Invocation("vision_patch_embed",
                   gemm(tokens, PATCH, VISION_DIM, extra_read=tokens * VISION_DIM * BF16), 1),
        Invocation("vision_norm_qkv",
                   gemm(tokens, VISION_DIM, 3 * VISION_DIM), VISION_LAYERS),
        # Bidirectional attention within each view: 256 keys per query.
        Invocation("vision_attention",
                   attention(tokens, VISION_TOKENS, VISION_HEAD_DIM, VISION_HEADS, VISION_HEADS),
                   VISION_LAYERS),
        Invocation("vision_out_proj_residual",
                   gemm(tokens, VISION_DIM, VISION_DIM, residual=True), VISION_LAYERS),
        Invocation("vision_norm_ffn_up",
                   gemm(tokens, VISION_DIM, VISION_FFN), VISION_LAYERS),
        Invocation("vision_ffn_down_residual",
                   gemm(tokens, VISION_FFN, VISION_DIM, residual=True), VISION_LAYERS),
    ]
    prefix = [
        Invocation("encoder_projector", gemm(tokens, VISION_DIM, ENCODER_DIM), 1),
        # A gather of `prompt_len` vocabulary rows, scaled: read and write once.
        Invocation("encoder_embed_prompt",
                   Cost(bytes_read=prompt_len * ENCODER_DIM * BF16,
                        bytes_written=prompt_len * ENCODER_DIM * BF16, flops=0), 1),
        Invocation("encoder_norm_qkv_rope", gemm(seq, ENCODER_DIM, QKV_WIDTH), layers),
        Invocation("encoder_attention",
                   attention(seq, seq, HEAD_DIM, DECODER_HEADS, 1), layers - 1),
        Invocation("encoder_out_proj_residual",
                   gemm(seq, ENCODER_DIM, ENCODER_DIM, residual=True), layers - 1),
        Invocation("encoder_norm_gated_ffn",
                   dual_gemm(seq, ENCODER_DIM, ENCODER_FFN), layers - 1),
        Invocation("encoder_ffn_down_residual",
                   gemm(seq, ENCODER_FFN, ENCODER_DIM, residual=True), layers - 1),
    ]
    body = steps * layers
    decoder = [
        Invocation("decoder_action_in_proj", gemm(m, ACTION_DIM, DECODER_DIM), steps),
        Invocation("decoder_norm_qkv_rope", gemm(m, DECODER_DIM, QKV_WIDTH), body),
        Invocation("decoder_attention",
                   attention(m, keys, HEAD_DIM, DECODER_HEADS, 1), body),
        Invocation("decoder_out_proj_residual",
                   gemm(m, DECODER_HEADS * HEAD_DIM, DECODER_DIM, residual=True), body),
        Invocation("decoder_norm_gated_ffn", dual_gemm(m, DECODER_DIM, DECODER_FFN), body),
        Invocation("decoder_ffn_down_residual",
                   gemm(m, DECODER_FFN, DECODER_DIM, residual=True), body),
        Invocation("decoder_action_out_proj",
                   gemm(m, DECODER_DIM, ACTION_DIM, residual=True), steps),
    ]
    return {"vision": vision, "prefix": prefix, "decoder": decoder}


__all__ = ["segment_costs"]
