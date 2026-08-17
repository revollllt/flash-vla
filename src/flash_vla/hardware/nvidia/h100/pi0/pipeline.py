"""The Pi0 forward pass: vision encoder, prefix encoder, action-expert decoder.

Pure orchestration -- every line is a call into the operation table, which
supplies either the standard or the fused implementation. Buffers are allocated
once by `Pi0Inference` and written in place, so nothing here allocates and the
whole pass captures into a single CUDA graph.

Shapes for the reference configuration (3 views, empty prompt, chunk 50):
vision runs 27 layers at 768 tokens, the encoder 18 layers at 768, and the
decoder 18 layers at 51 rows, repeated for each of the 10 diffusion steps.
"""
from __future__ import annotations

from flash_vla.models.pi0.spec import ENCODER_LAYERS, HEAD_DIM, VISION_LAYERS

from .backends.tilelang.kernels.attention import encoder_attention, vision_attention

DECODER_HEAD_DIM = HEAD_DIM


def vision_encoder(ops, weights, buffers, num_views):
    """Patch-embed the images and run the vision transformer, in place on `vision_x`."""
    assert num_views != 2, "the num_views==2 two-part vision branch is not implemented"

    ops.vision_patch_embed(
        buffers["observation_images_normalized"],
        weights["vision_patch_embedding_w"],
        weights["vision_patch_embedding_b"],
        weights["vision_position_embedding"],
        buffers["vision_x"])

    for i in range(VISION_LAYERS):
        ops.vision_norm_qkv(
            buffers["vision_x"],
            weights["vision_pre_attn_norm_w"][i], weights["vision_pre_attn_norm_b"][i],
            weights["vision_attn_qkv_w"][i], weights["vision_attn_qkv_b"][i],
            buffers["vision_QKV"], buffers["vision_x_norm"])

        attn = vision_attention(buffers["vision_QKV"])

        ops.vision_out_proj_residual(
            attn, weights["vision_attn_o_w"][i], weights["vision_attn_o_b"][i],
            buffers["vision_x"], buffers["vision_x"])

        ops.vision_norm_ffn_up(
            buffers["vision_x"],
            weights["vision_pre_ffn_norm_w"][i], weights["vision_pre_ffn_norm_b"][i],
            weights["vision_ffn_up_w"][i], weights["vision_ffn_up_b"][i],
            buffers["vision_hidden"], buffers["vision_x_norm"])

        ops.vision_ffn_down_residual(
            buffers["vision_hidden"],
            weights["vision_ffn_down_w"][i], weights["vision_ffn_down_b"][i],
            buffers["vision_x"], buffers["vision_x"])


def transformer_encoder(ops, weights, buffers, encoder_seq_len):
    """Project the vision output into encoder width and build the prefix KV cache.

    The last layer runs only its QKV projection: nothing downstream reads that
    layer's output, only its K and V, which the decoder attends over.
    """
    ops.encoder_projector(
        buffers["vision_x"],
        weights["vision_final_norm_w"], weights["vision_final_norm_b"],
        weights["encoder_multi_modal_projector_w"], weights["encoder_multi_modal_projector_b"],
        buffers["encoder_x"], buffers["vision_x_norm"])

    scale = DECODER_HEAD_DIM ** -0.5
    for i in range(ENCODER_LAYERS):
        ops.encoder_norm_qkv_rope(
            buffers["encoder_x"], weights["encoder_attn_qkv_w"][i],
            buffers["encoder_rope_weights"], buffers["encoder_Q"],
            buffers["encoder_K"][i, :encoder_seq_len], buffers["encoder_V"][i, :encoder_seq_len],
            buffers["encoder_x_norm"])

        if i == ENCODER_LAYERS - 1:
            break

        attn = encoder_attention(
            buffers["encoder_Q"], buffers["encoder_K"][i, :encoder_seq_len],
            buffers["encoder_V"][i, :encoder_seq_len], scale)

        ops.encoder_out_proj_residual(attn, weights["encoder_attn_o_w"][i], buffers["encoder_x"])
        ops.encoder_norm_gated_ffn(
            buffers["encoder_x"], weights["encoder_ffn_gate_w"][i], weights["encoder_ffn_up_w"][i],
            buffers["encoder_hidden"], buffers["encoder_x_norm"])
        ops.encoder_ffn_down_residual(
            buffers["encoder_hidden"], weights["encoder_ffn_down_w"][i], buffers["encoder_x"])


def transformer_decoder(ops, weights, buffers, encoder_seq_len, steps=10, layers=18):
    """Denoise the action chunk, attending over the prefix KV cache each step.

    The sequence is one state token followed by the action chunk. Each step
    re-projects the current noise, runs the transformer, and writes the update
    back into `diffusion_noise`, which is both the input and the output.
    """
    ops.decoder_state_proj(
        buffers["observation_state_normalized"], weights["decoder_state_in_proj_w"],
        weights["decoder_state_in_proj_b"], buffers["decoder_state_buf"])

    for step in range(steps):
        buffers["decoder_x"][:1].copy_(buffers["decoder_state_buf"])
        ops.decoder_action_in_proj(
            buffers["diffusion_noise"], weights["decoder_action_fused_in_proj_w"],
            weights["decoder_action_fused_time_biases"][step % 10], buffers["decoder_x_buf"])
        ops.decoder_action_mlp(
            buffers["decoder_x_buf"], weights["decoder_action_mlp_w"],
            weights["decoder_action_mlp_b"], buffers["decoder_x"][1:])

        for i in range(layers):
            ops.decoder_norm_qkv_rope(
                buffers["decoder_x"], weights["decoder_attn_qkv_w"][i],
                buffers["decoder_rope_weights"], buffers["decoder_q_buf"],
                buffers["encoder_K"][i][encoder_seq_len:],
                buffers["encoder_V"][i][encoder_seq_len:],
                buffers["decoder_norm_factor_buf"])
            ops.decoder_attention(
                buffers["decoder_q_buf"], buffers["encoder_K"][i], buffers["encoder_V"][i],
                buffers["decoder_attn_buf"], buffers["decoder_q_buf"], encoder_seq_len)
            ops.decoder_out_proj_residual(
                buffers["decoder_q_buf"].view(-1, 2048), weights["decoder_attn_o_w"][i],
                buffers["decoder_x"])
            ops.decoder_norm_gated_ffn(
                buffers["decoder_x"], weights["decoder_ffn_gate_w"][i],
                weights["decoder_ffn_up_w"][i], buffers["decoder_hidden"],
                buffers["decoder_norm_factor_buf"])
            ops.decoder_ffn_down_residual(
                buffers["decoder_hidden"], weights["decoder_ffn_down_w"][i], buffers["decoder_x"])

        ops.decoder_action_out_proj(
            buffers["decoder_x"][1:], weights["decoder_action_fused_out_proj_w"],
            weights["decoder_action_fused_out_proj_b"], buffers["diffusion_noise"],
            buffers["decoder_norm_factor_buf"])
