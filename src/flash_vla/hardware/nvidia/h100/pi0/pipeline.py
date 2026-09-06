"""The Pi0 computation graph: vision encoder, LLM backbone, action expert.

`build` writes the forward pass with the graph API (`flash_vla.runtime.graph`):
every op names its inputs, outputs and weights explicitly, buffers are
declared where the graph first needs them, and Python loops only unroll
layers and diffusion steps into nodes. Nothing here runs.

Shapes for the reference configuration (3 views, empty prompt, chunk 50): the
vision encoder runs 27 layers at 768 tokens, the backbone 18 layers at 768,
and the action expert 18 layers at 51 rows (the state token and the chunk),
repeated for each of the 10 diffusion steps.

Pi0's prompt is fixed at load time, so every table is static: both RoPE
tables are baked at construction and the prompt embeddings are a checkpoint
tensor copied into the prefix rows by `llm_backbone_embed_prompt`. The robot
state enters as a token: projected once per forward, copied into row 0 of the
expert's sequence every step, ahead of the action chunk.
"""
from __future__ import annotations

from typing import Mapping

import torch

from flash_vla.models.pi0.spec import (
    DECODER_HEADS,
    ENCODER_LAYERS,
    HEAD_DIM,
    ROPE_THETA,
    VISION_LAYERS,
)
from flash_vla.runtime.graph import Graph

VISION_TOKENS, VISION_DIM, VISION_FFN = 256, 1152, 4304
ENCODER_DIM, ENCODER_FFN = 2048, 16384
DECODER_DIM, DECODER_FFN = 1024, 4096
STATE_DIM = 32


def rope_table(seq_len: int, offset: int, head_dim: int, device,
               theta: float = ROPE_THETA) -> torch.Tensor:
    """Interleaved (cos, sin) rotary table for positions [offset, offset + seq_len)."""
    positions = torch.arange(seq_len, device=device) + offset
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                             device=device) / head_dim))
    phase = inv_freq[None, :] * positions[:, None]
    cos = torch.cos(phase).to(torch.bfloat16)
    sin = torch.sin(phase).to(torch.bfloat16)
    return torch.cat([cos[:, :, None], sin[:, :, None]], 2).view(-1, head_dim)


def build(g: Graph, shape: Mapping[str, int]) -> None:
    """Write the Pi0 graph at `shape` (num_views, chunk, steps, layers, prompt_len)."""
    num_views, chunk, steps = shape["num_views"], shape["chunk"], shape["steps"]
    layers, prompt_len = shape["layers"], shape["prompt_len"]
    assert num_views != 2, "the num_views==2 two-part vision branch is not implemented"

    image_tokens = num_views * VISION_TOKENS
    prefix_len = image_tokens + prompt_len
    # The expert's sequence is the state token followed by the action chunk.
    rows = chunk + 1
    cache_len = prefix_len + rows
    g.derived.update(image_tokens=image_tokens, prefix_len=prefix_len, cache_len=cache_len)

    # -- inputs ---------------------------------------------------------------
    images = g.buf("images", (num_views, 224, 224, 3))
    state = g.buf("state", (STATE_DIM,))
    actions = g.buf("actions", (chunk, STATE_DIM))

    # -- vision encoder -------------------------------------------------------
    g.stage("vision_encoder")
    vx = g.buf("vision_encoder_x", (num_views, VISION_TOKENS, VISION_DIM))
    vnorm = g.buf("vision_encoder_norm", (num_views, VISION_TOKENS, VISION_DIM))
    vqkv = g.buf("vision_encoder_qkv", (num_views, VISION_TOKENS, 3 * VISION_DIM))
    vattn = g.buf("vision_encoder_attn", (num_views, VISION_TOKENS, VISION_DIM))
    vhidden = g.buf("vision_encoder_hidden", (num_views, VISION_TOKENS, VISION_FFN))

    g.op("vision_encoder_patch_embed", images=images,
         patch_w=g.w("vision_patch_embedding_w"), patch_b=g.w("vision_patch_embedding_b"),
         pos_emb=g.w("vision_position_embedding"), out=vx)
    for i in range(VISION_LAYERS):
        g.op("vision_encoder_norm_qkv", x=vx,
             norm_w=g.w("vision_pre_attn_norm_w")[i], norm_b=g.w("vision_pre_attn_norm_b")[i],
             qkv_w=g.w("vision_attn_qkv_w")[i], qkv_b=g.w("vision_attn_qkv_b")[i],
             out=vqkv, x_norm=vnorm)
        g.op("vision_encoder_attention", qkv=vqkv, out=vattn)
        g.op("vision_encoder_out_proj_residual", x=vattn,
             weight=g.w("vision_attn_o_w")[i], bias=g.w("vision_attn_o_b")[i], res=vx, out=vx)
        g.op("vision_encoder_norm_ffn_up", x=vx,
             norm_w=g.w("vision_pre_ffn_norm_w")[i], norm_b=g.w("vision_pre_ffn_norm_b")[i],
             weight=g.w("vision_ffn_up_w")[i], bias=g.w("vision_ffn_up_b")[i],
             out=vhidden, x_norm=vnorm)
        g.op("vision_encoder_ffn_down_residual", x=vhidden,
             weight=g.w("vision_ffn_down_w")[i], bias=g.w("vision_ffn_down_b")[i],
             res=vx, out=vx)

    # -- llm backbone: the prefix and its KV cache ----------------------------
    g.stage("llm_backbone")
    bx = g.buf("llm_backbone_x", (prefix_len, ENCODER_DIM))
    bnorm = g.buf("llm_backbone_norm", (prefix_len, ENCODER_DIM))
    brope = g.buf("llm_backbone_rope", (prefix_len, HEAD_DIM),
                  init=lambda device: rope_table(prefix_len, 0, HEAD_DIM, device))
    # Zeroed so the slots of layers a bisected run never writes stay finite;
    # a full-depth run overwrites every slot.
    kv_k = g.buf("kv_k", (ENCODER_LAYERS, cache_len, HEAD_DIM), init="zero")
    kv_v = g.buf("kv_v", (ENCODER_LAYERS, cache_len, HEAD_DIM), init="zero")
    g.buf("prefix_k", alias="kv_k", view=(slice(None), slice(0, prefix_len)))
    g.buf("prefix_v", alias="kv_v", view=(slice(None), slice(0, prefix_len)))
    g.buf("suffix_k", alias="kv_k", view=(slice(None), slice(prefix_len, cache_len)))
    g.buf("suffix_v", alias="kv_v", view=(slice(None), slice(prefix_len, cache_len)))
    bq = g.buf("llm_backbone_q", (prefix_len * DECODER_HEADS, HEAD_DIM))
    battn = g.buf("llm_backbone_attn", (prefix_len * DECODER_HEADS, HEAD_DIM))
    bhidden = g.buf("llm_backbone_hidden", (prefix_len, ENCODER_FFN))

    # The prompt embeddings are a checkpoint tensor: copied into the language
    # rows every replay, the same op Pi0.5 implements as a vocabulary gather.
    g.op("llm_backbone_embed_prompt", token_ids=None, table=g.w("language_embeds"), scale=None,
         out=bx[image_tokens:prefix_len])
    g.op("llm_backbone_projector", x=vx,
         norm_w=g.w("vision_final_norm_w"), norm_b=g.w("vision_final_norm_b"),
         proj_w=g.w("encoder_multi_modal_projector_w"),
         proj_b=g.w("encoder_multi_modal_projector_b"), out=bx, x_norm=vnorm)
    scale = HEAD_DIM ** -0.5
    for i in range(ENCODER_LAYERS):
        g.op("llm_backbone_norm_qkv_rope", x=bx, weight_qkv=g.w("encoder_attn_qkv_w")[i],
             rope=brope, q=bq, k=kv_k[i, :prefix_len], v=kv_v[i, :prefix_len], x_norm=bnorm)
        # The last layer runs only its QKV projection: nothing downstream reads
        # its output, only its K and V, which the action expert attends over.
        if i == ENCODER_LAYERS - 1:
            break
        g.op("llm_backbone_attention", q=bq, k=kv_k[i, :prefix_len], v=kv_v[i, :prefix_len],
             scale=scale, mask=None, out=battn)
        g.op("llm_backbone_out_proj_residual", x=battn.view(prefix_len, DECODER_HEADS * HEAD_DIM),
             weight=g.w("encoder_attn_o_w")[i], out=bx)
        g.op("llm_backbone_norm_gated_ffn", x=bx, gate_w=g.w("encoder_ffn_gate_w")[i],
             up_w=g.w("encoder_ffn_up_w")[i], out=bhidden, x_norm=bnorm)
        g.op("llm_backbone_ffn_down_residual", x=bhidden, weight=g.w("encoder_ffn_down_w")[i],
             out=bx)

    # -- action expert: denoise the chunk over the KV cache -------------------
    g.stage("action_expert")
    ex = g.buf("action_expert_x", (rows, DECODER_DIM))
    erope = g.buf("action_expert_rope", (rows, HEAD_DIM),
                  init=lambda device: rope_table(rows, prefix_len, HEAD_DIM, device))
    est = g.buf("action_expert_state", (1, DECODER_DIM))
    ain = g.buf("action_expert_action_in", (chunk, DECODER_DIM))
    nf = g.buf("action_expert_norm_factor", (rows,))
    eq = g.buf("action_expert_q", (rows * DECODER_HEADS, HEAD_DIM))
    ehidden = g.buf("action_expert_hidden", (rows, DECODER_FFN))

    g.op("action_expert_state_proj", x=state, weight=g.w("decoder_state_in_proj_w"),
         bias=g.w("decoder_state_in_proj_b"), out=est)
    for step in range(steps):
        g.copy(ex[:1], est)
        g.op("action_expert_action_in_proj", x=actions,
             weight=g.w("decoder_action_fused_in_proj_w"),
             bias=g.w("decoder_action_fused_time_biases")[step % 10], out=ain)
        g.op("action_expert_action_mlp", x=ain, weight=g.w("decoder_action_mlp_w"),
             bias=g.w("decoder_action_mlp_b"), out=ex[1:])
        for i in range(layers):
            g.op("action_expert_norm_qkv_rope", x=ex, scale=None,
                 weight_qkv=g.w("decoder_attn_qkv_w")[i], bias=None, rope=erope,
                 q=eq, k=kv_k[i][prefix_len:], v=kv_v[i][prefix_len:], norm_factor=nf)
            g.op("action_expert_attention", q=eq, k=kv_k[i], v=kv_v[i], mask=None, out=eq,
                 prefix_len=prefix_len)
            g.op("action_expert_out_proj_residual", x=eq.view(-1, DECODER_HEADS * HEAD_DIM),
                 weight=g.w("decoder_attn_o_w")[i], gate=None, out=ex)
            g.op("action_expert_norm_gated_ffn", x=ex, scale=None,
                 gate_w=g.w("decoder_ffn_gate_w")[i], up_w=g.w("decoder_ffn_up_w")[i],
                 gate_b=None, up_b=None, out=ehidden, norm_factor=nf)
            g.op("action_expert_ffn_down_residual", x=ehidden,
                 weight=g.w("decoder_ffn_down_w")[i], gate=None, out=ex)
        g.op("action_expert_action_out_proj", x=ex[1:],
             weight=g.w("decoder_action_fused_out_proj_w"),
             bias=g.w("decoder_action_fused_out_proj_b"), out=actions, norm_factor=nf)


__all__ = ["build", "rope_table"]
