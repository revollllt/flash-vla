"""Synthetic decoder weights and buffers for the decoder-only benchmarks.

The full-pipeline benchmark uses the public `random_checkpoint` helper instead; this
module covers the decoder in isolation, where there is no vision/encoder stage
to produce the KV cache, so K/V are filled with noise directly.

`wscale` defaults to 0.05 because unscaled random weights make the 10-step
diffusion loop diverge numerically -- see the Pi0 fused-vs-unfused correctness check.
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.h100.pi0.buffers import rope_table

LAYERS = 18
HEAD_DIM = 256
NUM_HEADS = 8


def decoder_weights(seed: int = 0, wscale: float = 0.05, device: str = "cuda") -> dict:
    """The 13 weight tensors the H100/Pi0 decoder pipeline reads."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def rb(*shape):
        return (torch.randn(*shape, dtype=torch.float32, device=device, generator=gen) * wscale).to(torch.bfloat16)

    return {
        "decoder_state_in_proj_w": rb(32, 1024), "decoder_state_in_proj_b": rb(1024),
        "decoder_action_fused_in_proj_w": rb(32, 1024),
        "decoder_action_fused_time_biases": rb(10, 1024),
        "decoder_action_mlp_w": rb(1024, 1024), "decoder_action_mlp_b": rb(1024),
        "decoder_attn_qkv_w": rb(LAYERS, 1024, 2560),
        "decoder_attn_o_w": rb(LAYERS, 2048, 1024),
        "decoder_ffn_gate_w": rb(LAYERS, 1024, 4096), "decoder_ffn_up_w": rb(LAYERS, 1024, 4096),
        "decoder_ffn_down_w": rb(LAYERS, 4096, 1024),
        "decoder_action_fused_out_proj_w": rb(1024, 32), "decoder_action_fused_out_proj_b": rb(32),
    }


def decoder_buffers(num_views: int = 3, prompt_len: int = 0, chunk_size: int = 50,
                    seed: int = 1, device: str = "cuda") -> dict:
    """Buffers for one decoder run, with a pre-filled KV cache standing in for the encoder."""
    enc_len = num_views * 256 + prompt_len
    dec_len = chunk_size + 1
    total = enc_len + dec_len
    gen = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape, dtype=torch.float32, device=device, generator=gen) * scale).to(bf16)

    def zeros(*shape, dtype=bf16):
        return torch.zeros(*shape, dtype=dtype, device=device)

    return {
        "observation_state_normalized": rn(32),
        "diffusion_noise": rn(chunk_size, 32),
        "encoder_K": rn(LAYERS, total, HEAD_DIM, scale=0.1),
        "encoder_V": rn(LAYERS, total, HEAD_DIM, scale=0.1),
        "decoder_rope_weights": rope_table(dec_len, enc_len, HEAD_DIM, device),
        "decoder_x": zeros(dec_len, 1024),
        "decoder_x_buf": zeros(chunk_size, 1024),
        "decoder_state_buf": zeros(1, 1024),
        "decoder_norm_factor_buf": zeros(dec_len),
        "decoder_q_buf": zeros(dec_len * NUM_HEADS, HEAD_DIM),
        "decoder_attn_buf": zeros(dec_len * NUM_HEADS, total),
        "decoder_hidden": zeros(dec_len, 4096),
    }


def encoder_seq_len(num_views: int = 3, prompt_len: int = 0) -> int:
    return num_views * 256 + prompt_len
