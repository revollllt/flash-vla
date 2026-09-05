"""Cost declarations of the H100/Pi0 call sites, for its one segment.

Minimal traffic and math at the Target's shapes, backend-independent; the
invocation counts mirror `pipeline.py`: every encoder layer runs in full, the
state token is projected once, and the decoder body runs `steps x layers`
times on the chunk plus its state token.
"""
from __future__ import annotations

from flash_vla.models.pi0.spec import DECODER_HEADS, ENCODER_LAYERS, HEAD_DIM, VISION_LAYERS
from flash_vla.runtime.cost import BF16, Invocation, SegmentCosts, attention, dual_gemm, gemm

VISION_DIM, VISION_FFN, VISION_TOKENS = 1152, 4304, 256
VISION_HEADS = 16
VISION_HEAD_DIM = VISION_DIM // VISION_HEADS
ENCODER_DIM, ENCODER_FFN, QKV_WIDTH = 2048, 16384, 2560
DECODER_DIM, DECODER_FFN, ACTION_DIM = 1024, 4096, 32
PATCH = 14 * 14 * 3


def segment_costs(num_views: int, chunk_size: int, steps: int, layers: int,
                  prompt_len: int) -> SegmentCosts:
    """The single segment's invocations at this shape profile."""
    tokens = num_views * VISION_TOKENS
    seq = tokens + prompt_len
    m = chunk_size + 1                      # the chunk and its state token
    keys = seq + m
    body = steps * layers
    forward = [
        Invocation("vision_patch_embed",
                   gemm(tokens, PATCH, VISION_DIM, extra_read=tokens * VISION_DIM * BF16), 1),
        Invocation("vision_norm_qkv", gemm(tokens, VISION_DIM, 3 * VISION_DIM), VISION_LAYERS),
        Invocation("vision_attention",
                   attention(tokens, VISION_TOKENS, VISION_HEAD_DIM, VISION_HEADS, VISION_HEADS),
                   VISION_LAYERS),
        Invocation("vision_out_proj_residual",
                   gemm(tokens, VISION_DIM, VISION_DIM, residual=True), VISION_LAYERS),
        Invocation("vision_norm_ffn_up", gemm(tokens, VISION_DIM, VISION_FFN), VISION_LAYERS),
        Invocation("vision_ffn_down_residual",
                   gemm(tokens, VISION_FFN, VISION_DIM, residual=True), VISION_LAYERS),
        Invocation("encoder_projector", gemm(tokens, VISION_DIM, ENCODER_DIM), 1),
        Invocation("encoder_norm_qkv_rope", gemm(seq, ENCODER_DIM, QKV_WIDTH), ENCODER_LAYERS),
        Invocation("encoder_attention",
                   attention(seq, seq, HEAD_DIM, DECODER_HEADS, 1), ENCODER_LAYERS),
        Invocation("encoder_out_proj_residual",
                   gemm(seq, ENCODER_DIM, ENCODER_DIM, residual=True), ENCODER_LAYERS),
        Invocation("encoder_norm_gated_ffn",
                   dual_gemm(seq, ENCODER_DIM, ENCODER_FFN), ENCODER_LAYERS),
        Invocation("encoder_ffn_down_residual",
                   gemm(seq, ENCODER_FFN, ENCODER_DIM, residual=True), ENCODER_LAYERS),
        Invocation("decoder_state_proj", gemm(1, ACTION_DIM, DECODER_DIM), 1),
        Invocation("decoder_action_in_proj", gemm(chunk_size, ACTION_DIM, DECODER_DIM), steps),
        Invocation("decoder_action_mlp", gemm(chunk_size, DECODER_DIM, DECODER_DIM), steps),
        Invocation("decoder_norm_qkv_rope", gemm(m, DECODER_DIM, QKV_WIDTH), body),
        Invocation("decoder_attention", attention(m, keys, HEAD_DIM, DECODER_HEADS, 1), body),
        Invocation("decoder_out_proj_residual",
                   gemm(m, DECODER_HEADS * HEAD_DIM, DECODER_DIM, residual=True), body),
        Invocation("decoder_norm_gated_ffn", dual_gemm(m, DECODER_DIM, DECODER_FFN), body),
        Invocation("decoder_ffn_down_residual",
                   gemm(m, DECODER_FFN, DECODER_DIM, residual=True), body),
        Invocation("decoder_action_out_proj",
                   gemm(chunk_size, DECODER_DIM, ACTION_DIM, residual=True), steps),
    ]
    return {"forward": forward}


__all__ = ["segment_costs"]
