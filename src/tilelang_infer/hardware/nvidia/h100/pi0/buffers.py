"""Static buffers for the H100/Pi0 execution plan."""

from __future__ import annotations

import torch

from tilelang_infer.models.pi0.spec import (
    DECODER_HEADS,
    ENCODER_LAYERS,
    HEAD_DIM,
    ROPE_THETA,
)


def rope_table(seq_len: int, offset: int, head_dim: int, device) -> torch.Tensor:
    """Interleaved (cos, sin) rotary table for positions [offset, offset + seq_len)."""
    positions = torch.arange(seq_len, device=device) + offset
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                                  device=device) / head_dim))
    phase = inv_freq[None, :] * positions[:, None]
    cos = torch.cos(phase).to(torch.bfloat16)
    sin = torch.sin(phase).to(torch.bfloat16)
    return torch.cat([cos[:, :, None], sin[:, :, None]], 2).view(-1, head_dim)


def allocate_static_buffers(num_views: int, chunk_size: int, prompt_len: int,
                            device: str) -> tuple[dict[str, torch.Tensor], int]:
    """Allocate and initialize every persistent buffer used by this target."""
    bf16 = torch.bfloat16
    encoder_seq_len = num_views * 256 + prompt_len
    decoder_seq_len = chunk_size + 1
    cache_len = encoder_seq_len + decoder_seq_len

    def buf(*shape, dtype=bf16):
        return torch.empty(shape, dtype=dtype, device=device)

    buffers = {
        "observation_images_normalized": buf(num_views, 224, 224, 3),
        "observation_state_normalized": buf(32),
        "diffusion_noise": buf(chunk_size, 32),
        "vision_x": buf(num_views, 256, 1152),
        "vision_x_norm": buf(num_views, 256, 1152),
        "vision_QKV": buf(num_views, 256, 3 * 1152),
        "vision_hidden": buf(num_views, 256, 4304),
        "encoder_rope_weights": buf(encoder_seq_len, HEAD_DIM),
        "encoder_x": buf(encoder_seq_len, 2048),
        "encoder_x_norm": buf(encoder_seq_len, 2048),
        "encoder_K": buf(ENCODER_LAYERS, cache_len, HEAD_DIM),
        "encoder_V": buf(ENCODER_LAYERS, cache_len, HEAD_DIM),
        "encoder_Q": buf(encoder_seq_len * DECODER_HEADS, HEAD_DIM),
        "encoder_hidden": buf(encoder_seq_len, 16384),
        "decoder_rope_weights": buf(decoder_seq_len, HEAD_DIM),
        "decoder_x": buf(decoder_seq_len, 1024),
        "decoder_x_buf": buf(chunk_size, 1024),
        "decoder_state_buf": buf(1, 1024),
        "decoder_norm_factor_buf": buf(decoder_seq_len),
        "decoder_q_buf": buf(decoder_seq_len * DECODER_HEADS, HEAD_DIM),
        "decoder_attn_buf": buf(decoder_seq_len * DECODER_HEADS, cache_len),
        "decoder_hidden": buf(decoder_seq_len, 4096),
    }
    buffers["encoder_rope_weights"].copy_(rope_table(encoder_seq_len, 0, HEAD_DIM, device))
    buffers["decoder_rope_weights"].copy_(
        rope_table(decoder_seq_len, encoder_seq_len, HEAD_DIM, device))
    return buffers, encoder_seq_len
