"""Static buffers for the H100/Pi0.5 execution plan.

Three things differ from Pi0 and all three come from the state moving into the
prompt.

The prefix is longer and partly padding. `encoder_seq_len` is
`num_views*256 + 200` rather than `num_views*256`, and only the first
`num_views*256 + n_valid` rows carry data on any given call. Padding is masked
rather than skipped, so the addresses stay static and the shapes stay
capturable.

The prompt embeddings are an input, not a weight. Pi0's prompt is fixed at load
time, so `language_embeds` is a checkpoint tensor copied into `encoder_x` once.
Pi0.5's prompt contains the discretized state and changes every call, so the
target holds the raw vocabulary table and gathers into `encoder_x` per
inference.

The decoder's RoPE offset is data-dependent. Suffix positions are
`n_valid_prefix + 0..chunk-1`, and `n_valid_prefix` moves with the number of
digits in the state, so `decoder_rope_weights` is a per-inference input rather
than a table baked at construction. The *encoder* table stays static: padding
sits at the end of the language block, so a valid language token `j` always
lands at position `num_views*256 + j`.
"""
from __future__ import annotations

import torch

from flash_vla.models.pi05.spec import (
    DECODER_HEADS,
    ENCODER_DIM,
    ENCODER_FFN,
    ENCODER_LAYERS,
    HEAD_DIM,
    MAX_TOKEN_LEN,
    ROPE_THETA,
    VISION_DIM,
    VISION_FFN,
    VISION_TOKENS,
)

#: Masked keys get a large finite negative rather than -inf, matching OpenPI
#: (`models/gemma.py:225`) and Pi0's kernels. An all-masked row then softmaxes
#: to uniform instead of NaN, which is what upstream produces for the padded
#: query rows.
MASK_NEG = -3.0e38


def rope_table(seq_len: int, offset: int, head_dim: int, device) -> torch.Tensor:
    """Interleaved (cos, sin) rotary table for positions [offset, offset + seq_len)."""
    positions = torch.arange(seq_len, device=device) + offset
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                                  device=device) / head_dim))
    phase = inv_freq[None, :] * positions[:, None]
    cos = torch.cos(phase).to(torch.bfloat16)
    sin = torch.sin(phase).to(torch.bfloat16)
    return torch.cat([cos[:, :, None], sin[:, :, None]], 2).view(-1, head_dim)


def allocate_static_buffers(num_views: int, chunk_size: int, device: str,
                            prompt_len: int = MAX_TOKEN_LEN) -> tuple[dict[str, torch.Tensor], int]:
    """Allocate and initialize every persistent buffer used by this target."""
    bf16 = torch.bfloat16
    image_tokens = num_views * VISION_TOKENS
    encoder_seq_len = image_tokens + prompt_len
    cache_len = encoder_seq_len + chunk_size

    def buf(*shape, dtype=bf16):
        return torch.empty(shape, dtype=dtype, device=device)

    buffers = {
        "observation_images_normalized": buf(num_views, 224, 224, 3),
        "diffusion_noise": buf(chunk_size, 32),

        # Prompt inputs. `prompt_embed_scale` carries sqrt(width) on valid rows
        # and zero on padding, so one multiply both scales the embedding as the
        # embedder does and zeroes the padded rows. Leaving them uninitialized
        # would be a live hazard: a padded row's attention output feeds the next
        # layer, and 0 * NaN is NaN, which no mask can remove.
        "prompt_token_ids": buf(prompt_len, dtype=torch.int32),
        "prompt_embed_scale": buf(prompt_len, 1),
        # One vector serves both attentions: the encoder reads [:encoder_seq_len]
        # and the decoder the whole thing, since the suffix is never masked.
        "prefix_mask_bias": buf(cache_len),

        "vision_x": buf(num_views, VISION_TOKENS, VISION_DIM),
        "vision_x_norm": buf(num_views, VISION_TOKENS, VISION_DIM),
        "vision_QKV": buf(num_views, VISION_TOKENS, 3 * VISION_DIM),
        "vision_hidden": buf(num_views, VISION_TOKENS, VISION_FFN),

        "encoder_rope_weights": buf(encoder_seq_len, HEAD_DIM),
        "encoder_x": buf(encoder_seq_len, ENCODER_DIM),
        "encoder_x_norm": buf(encoder_seq_len, ENCODER_DIM),
        "encoder_K": buf(ENCODER_LAYERS, cache_len, HEAD_DIM),
        "encoder_V": buf(ENCODER_LAYERS, cache_len, HEAD_DIM),
        "encoder_Q": buf(encoder_seq_len * DECODER_HEADS, HEAD_DIM),
        "encoder_hidden": buf(encoder_seq_len, ENCODER_FFN),

        # Suffix. No state token, so the decoder sequence is the action chunk
        # alone -- Pi0's chunk + 1.
        "decoder_rope_weights": buf(chunk_size, HEAD_DIM),
        "decoder_x": buf(chunk_size, 1024),
        "decoder_norm_factor_buf": buf(chunk_size),
        "decoder_q_buf": buf(chunk_size * DECODER_HEADS, HEAD_DIM),
        "decoder_attn_buf": buf(chunk_size * DECODER_HEADS, cache_len),
        "decoder_hidden": buf(chunk_size, 4096),
    }

    # Anything the host rewrites every call still has to be valid during warmup,
    # which runs the pipeline three times before the first `forward`. Garbage
    # token ids are not merely wrong: `index_select` traps on an out-of-range
    # index and poisons the CUDA context, so this is a hard crash rather than a
    # bad number. Zero is a legal token and zero scale makes the row inert.
    buffers["prompt_token_ids"].zero_()
    buffers["prompt_embed_scale"].zero_()
    buffers["prefix_mask_bias"].zero_()

    # Static: valid language token j always lands at position image_tokens + j,
    # because padding is at the end of the language block and positions are
    # `cumsum(input_mask) - 1`.
    buffers["encoder_rope_weights"].copy_(rope_table(encoder_seq_len, 0, HEAD_DIM, device))
    # The suffix table is filled per inference by `prefix.PrefixInputs`; zero it
    # so a missed update fails loudly in parity rather than reusing stale phase.
    buffers["decoder_rope_weights"].zero_()
    return buffers, encoder_seq_len
