"""The buffer plan for the H100/Pi0 execution plan, declared as data.

`buffer_plan` returns one `Buffer` per name; the runtime's `StaticArena`
materializes it. Pi0's prompt is fixed at load time, so every table here is
static: both RoPE tables are baked at construction and the prompt embeddings
are copied from the checkpoint inside the captured segment.
"""

from __future__ import annotations

import torch

from flash_vla.models.pi0.spec import (
    DECODER_HEADS,
    ENCODER_LAYERS,
    HEAD_DIM,
    ROPE_THETA,
)
from flash_vla.runtime.cuda import Buffer


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
                prompt_len: int) -> tuple[dict[str, Buffer], int]:
    """Every persistent buffer of this target, and the prefix length."""
    encoder_seq_len = num_views * 256 + prompt_len
    # The decoder sequence is the state token followed by the action chunk.
    decoder_seq_len = chunk_size + 1
    cache_len = encoder_seq_len + decoder_seq_len

    def encoder_rope(device) -> torch.Tensor:
        return rope_table(encoder_seq_len, 0, HEAD_DIM, device)

    def decoder_rope(device) -> torch.Tensor:
        return rope_table(decoder_seq_len, encoder_seq_len, HEAD_DIM, device)

    plan = {
        "observation_images_normalized": Buffer((num_views, 224, 224, 3)),
        "observation_state_normalized": Buffer((32,)),
        "diffusion_noise": Buffer((chunk_size, 32)),
        "vision_x": Buffer((num_views, 256, 1152)),
        "vision_x_norm": Buffer((num_views, 256, 1152)),
        "vision_QKV": Buffer((num_views, 256, 3 * 1152)),
        "vision_hidden": Buffer((num_views, 256, 4304)),
        "encoder_rope_weights": Buffer((encoder_seq_len, HEAD_DIM), init=encoder_rope),
        "encoder_x": Buffer((encoder_seq_len, 2048)),
        "encoder_x_norm": Buffer((encoder_seq_len, 2048)),
        # Zeroed so the slots of layers a bisected run never writes stay
        # finite, as in Pi0.5; a full-depth run overwrites every slot.
        "encoder_K": Buffer((ENCODER_LAYERS, cache_len, HEAD_DIM), init="zero"),
        "encoder_V": Buffer((ENCODER_LAYERS, cache_len, HEAD_DIM), init="zero"),
        "encoder_Q": Buffer((encoder_seq_len * DECODER_HEADS, HEAD_DIM)),
        "encoder_hidden": Buffer((encoder_seq_len, 16384)),
        "decoder_rope_weights": Buffer((decoder_seq_len, HEAD_DIM), init=decoder_rope),
        "decoder_x": Buffer((decoder_seq_len, 1024)),
        "decoder_x_buf": Buffer((chunk_size, 1024)),
        "decoder_state_buf": Buffer((1, 1024)),
        "decoder_norm_factor_buf": Buffer((decoder_seq_len,)),
        "decoder_q_buf": Buffer((decoder_seq_len * DECODER_HEADS, HEAD_DIM)),
        "decoder_attn_buf": Buffer((decoder_seq_len * DECODER_HEADS, cache_len)),
        "decoder_hidden": Buffer((decoder_seq_len, 4096)),
    }
    return plan, encoder_seq_len
