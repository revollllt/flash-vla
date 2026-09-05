"""The buffer plan for the H100/Pi0.5 execution plan, declared as data.

`buffer_plan` returns one `Buffer` per name; the runtime's `StaticArena`
materializes it. Three things differ from Pi0 and all three come from the
state moving into the prompt.

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

The decoder-side buffers are allocated with padded rows and exposed as views.
The hand-written CUDA kernels of the attention half
(`backends/cuda/kernels/attn_taskloop.cu`) run on the padded allocation --
64 query rows, 1024 cache keys -- as `specs/tile/attention_block_contract.md`
section 3 specifies, recovering the base from the view's storage; the TileLang
kernels see the same 50-row / 1018-key views they always did. Pad rows are
zero (activations, RoPE, norm factor, cache) or masked (`MASK_NEG` on keys
`[cache_len, cache_pad)`) so that any kernel reading them computes finite
garbage that nothing consumes. The attention kernels never write pad rows; the
persistent FFN is the one exception -- it writes all 64 rows of
`decoder_hidden` and read-modify-writes all 64 rows of `decoder_x`, so on that
plan the `decoder_x` pad rows hold finite garbage that no valid row ever
consumes (every downstream op is per-row, pad cache keys stay masked, and the
XFS producer reads rows [:50] only).
"""
from __future__ import annotations

import torch

from flash_vla.models.pi05.spec import (
    DECODER_DIM,
    DECODER_FFN,
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
from flash_vla.runtime.cuda import Buffer

#: Masked keys get a large finite negative rather than -inf, matching OpenPI
#: (`models/gemma.py:225`) and Pi0's kernels. An all-masked row then softmaxes
#: to uniform instead of NaN, which is what upstream produces for the padded
#: query rows.
MASK_NEG = -3.0e38

#: Leading extents of the decoder-side buffers are padded to this multiple: one
#: wgmma m64 tile of query rows, one 64-key attention stage.
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


def buffer_plan(num_views: int, chunk_size: int,
                prompt_len: int = MAX_TOKEN_LEN) -> tuple[dict[str, Buffer], int]:
    """Every persistent buffer of this target, and the prefix length.

    Returns the plan the runtime materializes and `encoder_seq_len`. Buffers
    whose leading extent is padded declare the padded allocation and expose
    the pipeline's view; the initial value covers the whole allocation.
    """
    image_tokens = num_views * VISION_TOKENS
    encoder_seq_len = image_tokens + prompt_len
    cache_len = encoder_seq_len + chunk_size
    chunk_pad = -(-chunk_size // ROW_PAD) * ROW_PAD
    cache_pad = -(-cache_len // ROW_PAD) * ROW_PAD

    def rows(n: int) -> tuple[slice, ...]:
        return (slice(0, n),)

    def mask_bias(device) -> torch.Tensor:
        bias = torch.zeros(cache_pad, dtype=torch.bfloat16, device=device)
        bias[cache_len:] = MASK_NEG
        return bias

    def encoder_rope(device) -> torch.Tensor:
        return rope_table(encoder_seq_len, 0, HEAD_DIM, device)

    plan = {
        "observation_images_normalized": Buffer((num_views, 224, 224, 3)),
        "diffusion_noise": Buffer((chunk_size, 32)),

        # Prompt inputs. `prompt_embed_scale` carries sqrt(width) on valid rows
        # and zero on padding, so one multiply both scales the embedding as the
        # embedder does and zeroes the padded rows. Both start zeroed: warmup
        # runs the pipeline before the first `forward`, and a garbage token id
        # is not merely wrong -- `index_select` traps on an out-of-range index
        # and poisons the CUDA context. Zero is a legal token and zero scale
        # makes the row inert.
        "prompt_token_ids": Buffer((prompt_len,), torch.int32, init="zero"),
        "prompt_embed_scale": Buffer((prompt_len, 1), init="zero"),
        # One vector serves both attentions: the encoder reads [:encoder_seq_len]
        # and the decoder the whole thing, since the suffix is never masked.
        # Pad keys beyond `cache_len` carry the mask in the allocation itself.
        "prefix_mask_bias": Buffer((cache_pad,), init=mask_bias, view=rows(cache_len)),

        "vision_x": Buffer((num_views, VISION_TOKENS, VISION_DIM)),
        "vision_x_norm": Buffer((num_views, VISION_TOKENS, VISION_DIM)),
        "vision_QKV": Buffer((num_views, VISION_TOKENS, 3 * VISION_DIM)),
        "vision_hidden": Buffer((num_views, VISION_TOKENS, VISION_FFN)),

        # Static: valid language token j always lands at position
        # image_tokens + j, because padding is at the end of the language block
        # and positions are `cumsum(input_mask) - 1`.
        "encoder_rope_weights": Buffer((encoder_seq_len, HEAD_DIM), init=encoder_rope),
        "encoder_x": Buffer((encoder_seq_len, ENCODER_DIM)),
        "encoder_x_norm": Buffer((encoder_seq_len, ENCODER_DIM)),
        "encoder_K": Buffer((ENCODER_LAYERS, cache_pad, HEAD_DIM), init="zero",
                            view=(slice(None), slice(0, cache_len))),
        "encoder_V": Buffer((ENCODER_LAYERS, cache_pad, HEAD_DIM), init="zero",
                            view=(slice(None), slice(0, cache_len))),
        # The prefix rows are the prefix segment's contract and the suffix rows
        # the decoder's; both are views on the same cache, named so a harness
        # can compare or inject one without knowing the layout.
        "prefix_K": Buffer(alias="encoder_K", view=(slice(None), slice(0, encoder_seq_len))),
        "prefix_V": Buffer(alias="encoder_V", view=(slice(None), slice(0, encoder_seq_len))),
        "suffix_K": Buffer(alias="encoder_K",
                           view=(slice(None), slice(encoder_seq_len, cache_len))),
        "suffix_V": Buffer(alias="encoder_V",
                           view=(slice(None), slice(encoder_seq_len, cache_len))),
        "encoder_Q": Buffer((encoder_seq_len * DECODER_HEADS, HEAD_DIM)),
        # Destination for the fused CUDA encoder attention; the reference torch
        # route allocates its own result and leaves this idle. Either way the
        # output projection reads (encoder_seq_len, ENCODER_DIM), the same bytes.
        "encoder_attn_out": Buffer((encoder_seq_len * DECODER_HEADS, HEAD_DIM)),
        "encoder_hidden": Buffer((encoder_seq_len, ENCODER_FFN)),

        # Suffix. No state token, so the decoder sequence is the action chunk
        # alone -- Pi0's chunk + 1. The suffix RoPE table is filled per
        # inference by `prefix.PrefixInputs`; it starts zeroed so a missed
        # update fails loudly in parity rather than reusing stale phase.
        "decoder_rope_weights": Buffer((chunk_pad, HEAD_DIM), init="zero", view=rows(chunk_size)),
        "decoder_x": Buffer((chunk_pad, DECODER_DIM), init="zero", view=rows(chunk_size)),
        "decoder_norm_factor_buf": Buffer((chunk_pad,), init="zero", view=rows(chunk_size)),
        # Direct input to the persistent FFN. K-major makes the padded token
        # axis a contiguous 128-byte TMA row; the producer overwrites all rows.
        "decoder_ffn_xfs": Buffer((DECODER_DIM, chunk_pad)),
        # No score buffer: the only attention implementation is FlashDecoding,
        # which keeps the (queries, keys) matrix in SRAM. Pi0 allocated one for
        # its unfused three-kernel reference path, which Pi0.5 does not carry.
        "decoder_q_buf": Buffer((chunk_size * DECODER_HEADS, HEAD_DIM)),
        "decoder_hidden": Buffer((chunk_pad, DECODER_FFN), init="zero", view=rows(chunk_size)),
    }
    return plan, encoder_seq_len
