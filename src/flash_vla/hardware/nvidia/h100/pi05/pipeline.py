"""The Pi0.5 computation graph: vision encoder, LLM backbone, action expert.

`build` writes the forward pass with the graph API (`flash_vla.runtime.graph`):
every op names its inputs, outputs and weights explicitly, buffers are
declared where the graph first needs them, and Python loops only unroll
layers and denoising steps into nodes. Nothing here runs; the runner
allocates, routes and executes what is declared.

Shapes for the reference configuration (3 views, prompt padded to 200, chunk
50): the vision encoder runs 27 layers at 768 tokens, the backbone 18 layers
at 968, of which `768 + n_valid` carry data and the rest are masked padding,
and the action expert 18 layers at 50 rows for each of the 10 flow steps.

The pass is three stages and three graphs, split on data dependencies:
`vision_encoder` reads only the images, the host slot `prompt` tokenizes the
state while vision runs, `llm_backbone` needs the prompt embeddings, and
`action_expert` needs the KV cache the backbone built.

Two things differ from Pi0 and both come from the state moving into the
prompt. The prefix is longer and partly padding: only the first
`768 + n_valid` rows carry data on any call, and padding is masked rather than
skipped so the addresses stay static. The action expert's RoPE offset is
data-dependent (`n_valid + 0..chunk-1`), so `action_expert_rope` is filled per
inference by `PrefixInputs`; the backbone table stays static because padding
sits at the end of the language block.

The action-expert buffers are allocated with rows padded to a multiple of 64
and exposed as views: the hand-written CUDA kernels run on the padded
allocation (64 query rows, 1024 cache keys), recovering it from the view's
storage; TileLang sees the 50-row / 1018-key views. Pad rows are zero or
masked (`MASK_NEG` on keys `[cache_len, cache_pad)`) so any kernel reading
them computes finite garbage that nothing consumes.
"""
from __future__ import annotations

from typing import Mapping

import torch

from flash_vla.models.pi05.spec import (
    DECODER_DIM,
    DECODER_FFN,
    DECODER_HEADS,
    ENCODER_DIM,
    ENCODER_FFN,
    ENCODER_LAYERS,
    HEAD_DIM,
    MASK_NEG,
    ROPE_THETA,
    VISION_DIM,
    VISION_FFN,
    VISION_LAYERS,
    VISION_TOKENS,
)
from flash_vla.runtime.graph import Graph

#: Leading extents of the action-expert buffers are padded to this multiple:
#: one wgmma m64 tile of query rows, one 64-key attention stage.
ROW_PAD = 64


def rope_table(seq_len: int, offset: int, head_dim: int, device) -> torch.Tensor:
    """Interleaved (cos, sin) rotary table for positions [offset, offset + seq_len)."""
    positions = torch.arange(seq_len, device=device) + offset
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                                  device=device) / head_dim))
    phase = inv_freq[None, :] * positions[:, None]
    cos = torch.cos(phase).to(torch.bfloat16)
    sin = torch.sin(phase).to(torch.bfloat16)
    return torch.cat([cos[:, :, None], sin[:, :, None]], 2).view(-1, head_dim)


def _rows(n: int) -> tuple[slice, ...]:
    return (slice(0, n),)


def build(g: Graph, shape: Mapping[str, int]) -> None:
    """Write the Pi0.5 graph at `shape` (num_views, chunk, steps, layers, prompt_len)."""
    num_views, chunk, steps = shape["num_views"], shape["chunk"], shape["steps"]
    layers, prompt_len = shape["layers"], shape["prompt_len"]
    assert num_views != 2, "the num_views==2 two-part vision branch is not implemented"

    image_tokens = num_views * VISION_TOKENS
    prefix_len = image_tokens + prompt_len
    cache_len = prefix_len + chunk
    chunk_pad = -(-chunk // ROW_PAD) * ROW_PAD
    cache_pad = -(-cache_len // ROW_PAD) * ROW_PAD
    g.derived.update(image_tokens=image_tokens, prefix_len=prefix_len, cache_len=cache_len)

    def mask_init(device) -> torch.Tensor:
        bias = torch.zeros(cache_pad, dtype=torch.bfloat16, device=device)
        bias[cache_len:] = MASK_NEG
        return bias

    # -- inputs and the prompt buffers the host slot fills ------------------
    images = g.buf("images", (num_views, 224, 224, 3))
    actions = g.buf("actions", (chunk, 32))
    # `prompt_scale` carries sqrt(width) on valid rows and zero on padding, so
    # one multiply both scales the embedding and zeroes the padded rows. Both
    # start zeroed: warmup runs the graph before the first `forward`, and an
    # out-of-range token id traps in `index_select` and poisons the context.
    prompt_ids = g.buf("prompt_ids", (prompt_len,), torch.int32, init="zero")
    prompt_scale = g.buf("prompt_scale", (prompt_len, 1), init="zero")
    # One vector serves both attentions: the backbone reads [:prefix_len] and
    # the action expert the whole view, since the suffix is never masked.
    mask_bias = g.buf("mask_bias", (cache_pad,), init=mask_init, view=_rows(cache_len))

    # -- vision encoder -----------------------------------------------------
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

    # -- host: tokenize the state into the prompt buffers while vision runs --
    g.host("prompt")

    # -- llm backbone: the prefix and its KV cache ---------------------------
    g.stage("llm_backbone")
    bx = g.buf("llm_backbone_x", (prefix_len, ENCODER_DIM))
    bnorm = g.buf("llm_backbone_norm", (prefix_len, ENCODER_DIM))
    # Static: valid language token j always lands at position image_tokens + j.
    brope = g.buf("llm_backbone_rope", (prefix_len, HEAD_DIM),
                  init=lambda device: rope_table(prefix_len, 0, HEAD_DIM, device))
    kv_k = g.buf("kv_k", (ENCODER_LAYERS, cache_pad, HEAD_DIM), init="zero",
                 view=(slice(None), slice(0, cache_len)))
    kv_v = g.buf("kv_v", (ENCODER_LAYERS, cache_pad, HEAD_DIM), init="zero",
                 view=(slice(None), slice(0, cache_len)))
    # The prefix rows are this stage's contract and the suffix rows the action
    # expert's; both are views on the same cache, so a harness can compare or
    # inject one without knowing the layout.
    g.buf("prefix_k", alias="kv_k", view=(slice(None), slice(0, prefix_len)))
    g.buf("prefix_v", alias="kv_v", view=(slice(None), slice(0, prefix_len)))
    g.buf("suffix_k", alias="kv_k", view=(slice(None), slice(prefix_len, cache_len)))
    g.buf("suffix_v", alias="kv_v", view=(slice(None), slice(prefix_len, cache_len)))
    bq = g.buf("llm_backbone_q", (prefix_len * DECODER_HEADS, HEAD_DIM))
    battn = g.buf("llm_backbone_attn", (prefix_len * DECODER_HEADS, HEAD_DIM))
    bhidden = g.buf("llm_backbone_hidden", (prefix_len, ENCODER_FFN))

    g.op("llm_backbone_projector", x=vx,
         norm_w=g.w("vision_final_norm_w"), norm_b=g.w("vision_final_norm_b"),
         proj_w=g.w("encoder_multi_modal_projector_w"),
         proj_b=g.w("encoder_multi_modal_projector_b"), out=bx, x_norm=vnorm)
    g.op("llm_backbone_embed_prompt", token_ids=prompt_ids, table=g.w("vocab_embeddings"),
         scale=prompt_scale, out=bx[image_tokens:prefix_len])
    scale = HEAD_DIM ** -0.5
    mask = mask_bias[:prefix_len]
    for i in range(layers):
        g.op("llm_backbone_norm_qkv_rope", x=bx, weight_qkv=g.w("encoder_attn_qkv_w")[i],
             rope=brope, q=bq, k=kv_k[i, :prefix_len], v=kv_v[i, :prefix_len], x_norm=bnorm)
        # The last layer runs only its QKV projection: nothing downstream reads
        # its output, only its K and V, which the action expert attends over.
        if i == layers - 1:
            break
        g.op("llm_backbone_attention", q=bq, k=kv_k[i, :prefix_len], v=kv_v[i, :prefix_len],
             scale=scale, mask=mask, out=battn)
        g.op("llm_backbone_out_proj_residual", x=battn.view(prefix_len, DECODER_HEADS * HEAD_DIM),
             weight=g.w("encoder_attn_o_w")[i], out=bx)
        g.op("llm_backbone_norm_gated_ffn", x=bx, gate_w=g.w("encoder_ffn_gate_w")[i],
             up_w=g.w("encoder_ffn_up_w")[i], out=bhidden, x_norm=bnorm)
        g.op("llm_backbone_ffn_down_residual", x=bhidden, weight=g.w("encoder_ffn_down_w")[i],
             out=bx)

    # -- action expert: denoise the chunk over the KV cache -----------------
    # No state token, so the sequence is the action chunk alone. AdaRMSNorm
    # arrives as per-(step, layer) constants: `_ada_*_scale` scales the GEMM's
    # A operand, `_shift_bias` is an epilogue bias, `_ada_*_gate` multiplies
    # the residual branch; the Euler dt is folded into the output projection.
    g.stage("action_expert")
    ex = g.buf("action_expert_x", (chunk_pad, DECODER_DIM), init="zero", view=_rows(chunk))
    # Filled per inference by `PrefixInputs`; zeroed so a missed update fails
    # loudly in the reference check rather than reusing stale phase.
    erope = g.buf("action_expert_rope", (chunk_pad, HEAD_DIM), init="zero", view=_rows(chunk))
    nf = g.buf("action_expert_norm_factor", (chunk_pad,), init="zero", view=_rows(chunk))
    eq = g.buf("action_expert_q", (chunk * DECODER_HEADS, HEAD_DIM))
    ehidden = g.buf("action_expert_hidden", (chunk_pad, DECODER_FFN), init="zero",
                    view=_rows(chunk))

    for step in range(steps):
        g.op("action_expert_action_in_proj", x=actions, weight=g.w("decoder_action_in_proj_w"),
             bias=g.w("decoder_action_in_proj_b"), out=ex)
        for i in range(layers):
            g.op("action_expert_norm_qkv_rope", x=ex,
                 scale=g.w("decoder_ada_attn_scale")[step, i],
                 weight_qkv=g.w("decoder_attn_qkv_w")[i],
                 bias=g.w("decoder_qkv_shift_bias")[step, i], rope=erope,
                 q=eq, k=kv_k[i][prefix_len:], v=kv_v[i][prefix_len:], norm_factor=nf)
            g.op("action_expert_attention", q=eq, k=kv_k[i], v=kv_v[i], mask=mask_bias,
                 out=eq, prefix_len=prefix_len)
            g.op("action_expert_out_proj_residual", x=eq.view(-1, DECODER_HEADS * HEAD_DIM),
                 weight=g.w("decoder_attn_o_w")[i], gate=g.w("decoder_ada_attn_gate")[step, i],
                 out=ex)
            g.op("action_expert_norm_gated_ffn", x=ex,
                 scale=g.w("decoder_ada_ffn_scale")[step, i],
                 gate_w=g.w("decoder_ffn_gate_w")[i], up_w=g.w("decoder_ffn_up_w")[i],
                 gate_b=g.w("decoder_ffn_gate_shift_bias")[step, i],
                 up_b=g.w("decoder_ffn_up_shift_bias")[step, i], out=ehidden, norm_factor=nf)
            g.op("action_expert_ffn_down_residual", x=ehidden,
                 weight=g.w("decoder_ffn_down_w")[i], gate=g.w("decoder_ada_ffn_gate")[step, i],
                 out=ex)
        g.op("action_expert_action_out_proj", x=ex,
             weight=g.w("decoder_action_out_proj_w")[step],
             bias=g.w("decoder_action_out_proj_b")[step], out=actions, norm_factor=nf)


__all__ = ["ROW_PAD", "build", "rope_table"]
