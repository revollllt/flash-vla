"""The Pi0.5 forward pass: vision, prompt embedding, encoder, action decoder.

Pure orchestration -- every line is a call into the operation table, which
supplies either the standard or the fused implementation. Buffers are allocated
once by the engine and written in place, so nothing here allocates and each
stage captures into a CUDA graph.

Shapes for the reference configuration (3 views, prompt padded to 200, chunk
50): vision runs 27 layers at 768 tokens, and the encoder 18 layers at 968, of
which `768 + n_valid` carry data and the rest are masked padding.

The pass is three stages and three graphs, split on data dependencies rather
than for tidiness: `vision` reads only the images,
`prefix` needs the prompt embeddings, and `decoder` needs the KV cache `prefix`
built. Splitting vision from the rest is what lets the host tokenize while the
GPU works; splitting prefix from decoder costs one extra graph launch and buys a
per-stage timing breakdown, which is how the roofline in PLAN.md §1.2 is read.
"""
from __future__ import annotations

from flash_vla.models.pi05.spec import (
    DECODER_HEADS,
    ENCODER_LAYERS,
    HEAD_DIM,
    VISION_LAYERS,
    VISION_TOKENS,
)

from .backends.tilelang.kernels.attention import encoder_attention, vision_attention


def vision(ops, weights, buffers, num_views):
    """Patch-embed the images and run the vision transformer, in place on `vision_x`.

    Depends only on `observation_images_normalized`, which is what makes it
    capturable separately from everything the prompt touches.
    """
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


def prefix(ops, weights, buffers, num_views, encoder_seq_len, layers=ENCODER_LAYERS):
    """Assemble the prefix and build the KV cache the decoder attends over.

    The prefix is image tokens followed by prompt tokens: the projector fills
    the first `num_views*256` rows from the vision output, and the embedding
    gather fills the rest from the vocabulary table. Padded prompt rows come out
    zero, because `prompt_embed_scale` is zero there.

    The last layer runs only its QKV projection: nothing downstream reads that
    layer's output, only its K and V, which the decoder attends over.
    """
    image_tokens = num_views * VISION_TOKENS

    ops.encoder_projector(
        buffers["vision_x"],
        weights["vision_final_norm_w"], weights["vision_final_norm_b"],
        weights["encoder_multi_modal_projector_w"], weights["encoder_multi_modal_projector_b"],
        buffers["encoder_x"], buffers["vision_x_norm"])

    ops.encoder_embed_prompt(
        buffers["prompt_token_ids"], weights["vocab_embeddings"],
        buffers["prompt_embed_scale"], buffers["encoder_x"][image_tokens:encoder_seq_len])

    scale = HEAD_DIM ** -0.5
    mask = buffers["prefix_mask_bias"][:encoder_seq_len]
    for i in range(layers):
        ops.encoder_norm_qkv_rope(
            buffers["encoder_x"], weights["encoder_attn_qkv_w"][i],
            buffers["encoder_rope_weights"], buffers["encoder_Q"],
            buffers["encoder_K"][i, :encoder_seq_len], buffers["encoder_V"][i, :encoder_seq_len],
            buffers["encoder_x_norm"])

        if i == layers - 1:
            break

        attn = encoder_attention(
            buffers["encoder_Q"], buffers["encoder_K"][i, :encoder_seq_len],
            buffers["encoder_V"][i, :encoder_seq_len], scale, mask)

        ops.encoder_out_proj_residual(attn, weights["encoder_attn_o_w"][i], buffers["encoder_x"])
        ops.encoder_norm_gated_ffn(
            buffers["encoder_x"], weights["encoder_ffn_gate_w"][i], weights["encoder_ffn_up_w"][i],
            buffers["encoder_hidden"], buffers["encoder_x_norm"])
        ops.encoder_ffn_down_residual(
            buffers["encoder_hidden"], weights["encoder_ffn_down_w"][i], buffers["encoder_x"])


def decoder(ops, weights, buffers, encoder_seq_len, steps=10, layers=ENCODER_LAYERS):
    """Denoise the action chunk, attending over the prefix KV cache each step.

    The sequence is the action chunk alone -- Pi0 carried a state token at row 0
    and Pi0.5 does not, so M is 50 rather than 51 and the two-block attention
    mask that existed to keep that token out of the action block is gone.

    Each step re-projects the current noise, runs the transformer, and writes the
    update back into `diffusion_noise`, which is both the input and the output.
    The Euler `dt` is folded into `decoder_action_out_proj_w`, so the step is an
    accumulate rather than an explicit `x + dt*v`.

    AdaRMSNorm arrives entirely as per-(step, layer) constants: `_ada_*_scale`
    scales the GEMM's A operand, `_shift_bias` is an epilogue bias, and
    `_ada_*_gate` multiplies the residual branch. See PLAN.md §1.1 for why none
    of the 116 M modulation parameters is streamed, and
    `backends/tilelang/kernels/adarms.py` for how each lands in a kernel.
    """
    for step in range(steps):
        ops.decoder_action_in_proj(
            buffers["diffusion_noise"], weights["decoder_action_in_proj_w"],
            weights["decoder_action_in_proj_b"], buffers["decoder_x"])

        for i in range(layers):
            ops.decoder_norm_qkv_rope(
                buffers["decoder_x"],
                weights["decoder_ada_attn_scale"][step, i],
                weights["decoder_attn_qkv_w"][i],
                weights["decoder_qkv_shift_bias"][step, i],
                buffers["decoder_rope_weights"],
                buffers["decoder_q_buf"],
                buffers["encoder_K"][i][encoder_seq_len:],
                buffers["encoder_V"][i][encoder_seq_len:],
                buffers["decoder_norm_factor_buf"])
            ops.decoder_attention(
                buffers["decoder_q_buf"], buffers["encoder_K"][i], buffers["encoder_V"][i],
                buffers["prefix_mask_bias"], buffers["decoder_q_buf"])
            ops.decoder_out_proj_residual(
                buffers["decoder_q_buf"].view(-1, DECODER_HEADS * HEAD_DIM),
                weights["decoder_attn_o_w"][i],
                weights["decoder_ada_attn_gate"][step, i],
                buffers["decoder_x"])
            ops.decoder_norm_gated_ffn(
                buffers["decoder_x"],
                weights["decoder_ada_ffn_scale"][step, i],
                weights["decoder_ffn_gate_w"][i], weights["decoder_ffn_up_w"][i],
                weights["decoder_ffn_gate_shift_bias"][step, i],
                weights["decoder_ffn_up_shift_bias"][step, i],
                buffers["decoder_hidden"], buffers["decoder_norm_factor_buf"])
            ops.decoder_ffn_down_residual(
                buffers["decoder_hidden"], weights["decoder_ffn_down_w"][i],
                weights["decoder_ada_ffn_gate"][step, i], buffers["decoder_x"])

        ops.decoder_action_out_proj(
            buffers["decoder_x"],
            weights["decoder_action_out_proj_w"][step],
            weights["decoder_action_out_proj_b"][step],
            buffers["diffusion_noise"], buffers["decoder_norm_factor_buf"])
