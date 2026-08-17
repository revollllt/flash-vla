"""Hardware-independent Pi0 constants and checkpoint schema."""

VISION_LAYERS = 27
ENCODER_LAYERS = 18
HEAD_DIM = 256
DECODER_HEADS = 8
ROPE_THETA = 10000


def weight_shapes(prompt_len: int) -> dict[str, tuple[int, ...]]:
    """Expected checkpoint tensors for a Pi0 model with this prompt length."""
    return {
        "vision_patch_embedding_w": (14, 14, 3, 1152), "vision_patch_embedding_b": (1152,),
        "vision_position_embedding": (256, 1152),
        "vision_attn_qkv_w": (VISION_LAYERS, 1152, 3 * 1152),
        "vision_attn_qkv_b": (VISION_LAYERS, 3 * 1152),
        "vision_attn_o_w": (VISION_LAYERS, 1152, 1152),
        "vision_attn_o_b": (VISION_LAYERS, 1152),
        "vision_ffn_up_w": (VISION_LAYERS, 1152, 4304),
        "vision_ffn_up_b": (VISION_LAYERS, 4304),
        "vision_ffn_down_w": (VISION_LAYERS, 4304, 1152),
        "vision_ffn_down_b": (VISION_LAYERS, 1152),
        "vision_pre_attn_norm_w": (VISION_LAYERS, 1152),
        "vision_pre_attn_norm_b": (VISION_LAYERS, 1152),
        "vision_pre_ffn_norm_w": (VISION_LAYERS, 1152),
        "vision_pre_ffn_norm_b": (VISION_LAYERS, 1152),
        "vision_final_norm_w": (1152,), "vision_final_norm_b": (1152,),
        "encoder_multi_modal_projector_w": (1152, 2048),
        "encoder_multi_modal_projector_b": (2048,),
        "encoder_attn_qkv_w": (ENCODER_LAYERS, 2048, 2560),
        "encoder_attn_o_w": (ENCODER_LAYERS, 2048, 2048),
        "encoder_ffn_gate_w": (ENCODER_LAYERS, 2048, 16384),
        "encoder_ffn_up_w": (ENCODER_LAYERS, 2048, 16384),
        "encoder_ffn_down_w": (ENCODER_LAYERS, 16384, 2048),
        "decoder_state_in_proj_w": (32, 1024), "decoder_state_in_proj_b": (1024,),
        "decoder_action_fused_in_proj_w": (32, 1024),
        "decoder_action_fused_time_biases": (10, 1024),
        "decoder_action_mlp_w": (1024, 1024), "decoder_action_mlp_b": (1024,),
        "decoder_attn_qkv_w": (ENCODER_LAYERS, 1024, 2560),
        "decoder_attn_o_w": (ENCODER_LAYERS, 2048, 1024),
        "decoder_ffn_gate_w": (ENCODER_LAYERS, 1024, 4096),
        "decoder_ffn_up_w": (ENCODER_LAYERS, 1024, 4096),
        "decoder_ffn_down_w": (ENCODER_LAYERS, 4096, 1024),
        "decoder_action_fused_out_proj_w": (1024, 32),
        "decoder_action_fused_out_proj_b": (32,),
        "language_embeds": (prompt_len, 2048),
    }
