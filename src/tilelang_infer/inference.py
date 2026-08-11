"""`Pi0Inference`: weights, buffers, and one CUDA graph for the whole forward pass.

Construction allocates every weight and buffer up front, precomputes the RoPE
tables, loads the checkpoint, warms the kernels, and captures a single graph
covering vision, encoder and decoder. `forward` then copies the three inputs
into their static buffers and replays.

Capture is what makes the numbers reproducible, and it constrains the design:
nothing inside the pass may allocate. Scratch that the wrappers need comes from
a `ScratchPool`, which is frozen after warmup so a missed pre-allocation raises
instead of silently allocating mid-capture.
"""
from __future__ import annotations

import torch

from . import pi0_infer, wrappers
from .buffers import ScratchPool
from .ops import op_table

VISION_LAYERS = 27
ENCODER_LAYERS = 18
HEAD_DIM = 256
DECODER_HEADS = 8
ROPE_THETA = 10000


def rope_table(seq_len: int, offset: int, head_dim: int, device) -> torch.Tensor:
    """Interleaved (cos, sin) rotary table for positions [offset, offset + seq_len)."""
    positions = torch.arange(seq_len, device=device) + offset
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                                  device=device) / head_dim))
    phase = inv_freq[None, :] * positions[:, None]
    cos = torch.cos(phase).to(torch.bfloat16)
    sin = torch.sin(phase).to(torch.bfloat16)
    return torch.cat([cos[:, :, None], sin[:, :, None]], 2).view(-1, head_dim)


def _weight_shapes(prompt_len: int) -> dict[str, tuple]:
    return {
        "vision_patch_embedding_w": (14, 14, 3, 1152), "vision_patch_embedding_b": (1152,),
        "vision_position_embedding": (256, 1152),
        "vision_attn_qkv_w": (VISION_LAYERS, 1152, 3 * 1152),
        "vision_attn_qkv_b": (VISION_LAYERS, 3 * 1152),
        "vision_attn_o_w": (VISION_LAYERS, 1152, 1152), "vision_attn_o_b": (VISION_LAYERS, 1152),
        "vision_ffn_up_w": (VISION_LAYERS, 1152, 4304), "vision_ffn_up_b": (VISION_LAYERS, 4304),
        "vision_ffn_down_w": (VISION_LAYERS, 4304, 1152), "vision_ffn_down_b": (VISION_LAYERS, 1152),
        "vision_pre_attn_norm_w": (VISION_LAYERS, 1152), "vision_pre_attn_norm_b": (VISION_LAYERS, 1152),
        "vision_pre_ffn_norm_w": (VISION_LAYERS, 1152), "vision_pre_ffn_norm_b": (VISION_LAYERS, 1152),
        "vision_final_norm_w": (1152,), "vision_final_norm_b": (1152,),
        "encoder_multi_modal_projector_w": (1152, 2048), "encoder_multi_modal_projector_b": (2048,),
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
        "decoder_action_fused_out_proj_w": (1024, 32), "decoder_action_fused_out_proj_b": (32,),
        "language_embeds": (prompt_len, 2048),
    }


class Pi0Inference:
    """One captured Pi0 forward pass.

    steps and layers exist for bisection: shortening either keeps the pipeline
    intact while cutting depth, which is how parity is read -- on random weights
    a deep run diverges chaotically between any two implementations that are not
    bit-identical.
    """

    def __init__(self, checkpoint, num_views: int, chunk_size: int, steps: int = 10,
                 layers: int = 18, fused: bool = True, device: str = "cuda"):
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.steps = steps
        self.layers = layers
        self.fused = fused
        self.ops = op_table(fused)
        self.prompt_len = len(checkpoint["language_embeds"])

        bf16 = torch.bfloat16
        self.weights = {name: torch.empty(shape, dtype=bf16, device=device)
                        for name, shape in _weight_shapes(self.prompt_len).items()}

        self.encoder_seq_len = num_views * 256 + self.prompt_len
        decoder_seq_len = chunk_size + 1
        cache_len = self.encoder_seq_len + decoder_seq_len

        def buf(*shape, dtype=bf16):
            return torch.empty(shape, dtype=dtype, device=device)

        self.buffers = {
            "observation_images_normalized": buf(num_views, 224, 224, 3),
            "observation_state_normalized": buf(32),
            "diffusion_noise": buf(chunk_size, 32),
            "vision_x": buf(num_views, 256, 1152),
            "vision_x_norm": buf(num_views, 256, 1152),
            "vision_QKV": buf(num_views, 256, 3 * 1152),
            "vision_hidden": buf(num_views, 256, 4304),
            "encoder_rope_weights": buf(self.encoder_seq_len, HEAD_DIM),
            "encoder_x": buf(self.encoder_seq_len, 2048),
            "encoder_x_norm": buf(self.encoder_seq_len, 2048),
            "encoder_K": buf(ENCODER_LAYERS, cache_len, HEAD_DIM),
            "encoder_V": buf(ENCODER_LAYERS, cache_len, HEAD_DIM),
            "encoder_Q": buf(self.encoder_seq_len * DECODER_HEADS, HEAD_DIM),
            "encoder_hidden": buf(self.encoder_seq_len, 16384),
            "decoder_rope_weights": buf(decoder_seq_len, HEAD_DIM),
            "decoder_x": buf(decoder_seq_len, 1024),
            "decoder_x_buf": buf(chunk_size, 1024),
            "decoder_state_buf": buf(1, 1024),
            "decoder_norm_factor_buf": buf(decoder_seq_len),
            "decoder_q_buf": buf(decoder_seq_len * DECODER_HEADS, HEAD_DIM),
            "decoder_attn_buf": buf(decoder_seq_len * DECODER_HEADS, cache_len),
            "decoder_hidden": buf(decoder_seq_len, 4096),
        }
        self.buffers["encoder_rope_weights"].copy_(
            rope_table(self.encoder_seq_len, 0, HEAD_DIM, device))
        self.buffers["decoder_rope_weights"].copy_(
            rope_table(decoder_seq_len, self.encoder_seq_len, HEAD_DIM, device))

        for name, value in checkpoint.items():
            self.weights[name].copy_(value)

        self.pool = ScratchPool()
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _run(self):
        """One full forward pass, in place on the static buffers."""
        self.buffers["encoder_x"][self.num_views * 256:].copy_(self.weights["language_embeds"])
        with wrappers.use_pool(self.pool):
            pi0_infer.vision_encoder(self.ops, self.weights, self.buffers, self.num_views)
            pi0_infer.transformer_encoder(self.ops, self.weights, self.buffers, self.encoder_seq_len)
            pi0_infer.transformer_decoder(self.ops, self.weights, self.buffers,
                                          self.encoder_seq_len, steps=self.steps, layers=self.layers)

    def _capture(self):
        """Warm up (compiling every kernel and filling the pool), freeze, then capture."""
        for _ in range(3):
            self._run()
        torch.cuda.synchronize()
        self.pool.freeze()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.graph.capture_begin()
            self._run()
            self.graph.capture_end()
        torch.cuda.synchronize()

    def forward(self, images, state, noise):
        """Copy inputs into the static buffers, replay the graph, return the denoised chunk."""
        self.buffers["observation_images_normalized"].copy_(images)
        self.buffers["observation_state_normalized"].copy_(state)
        self.buffers["diffusion_noise"].copy_(noise)
        self.graph.replay()
        return self.buffers["diffusion_noise"]


def random_checkpoint(num_views: int = 3, chunk_size: int = 50, prompt_len: int = 256,
                      wscale: float = 0.05, seed: int = 0, device: str = "cuda") -> dict:
    """Synthetic checkpoint for benchmarking.

    `wscale` is small on purpose: at unit scale the residual stream grows through
    63 bf16 layers until the diffusion loop is numerically meaningless.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        name: (torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * wscale
               ).to(torch.bfloat16)
        for name, shape in _weight_shapes(prompt_len).items()
    }
