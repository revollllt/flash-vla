# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only PyTorch composition of Isaac-GR00T N1.7.

Adapted from NVIDIA/Isaac-GR00T, revision 51d4c89, gr00t_n1d7.py,
modules/dit.py and modules/embodiment_conditioned_mlp.py. Qwen3-VL and
Diffusers supply their ordinary PyTorch modules; no custom GPU kernel is used.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig, Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel, Qwen3VLVisionModel,
)


class CategoryLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(32, in_dim, out_dim))
        self.b = nn.Parameter(torch.empty(32, out_dim))

    def forward(self, x, embodiment):
        return torch.bmm(x, self.W[embodiment]) + self.b[embodiment].unsqueeze(1)


class CategoryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer1 = CategoryLinear(in_dim, hidden_dim)
        self.layer2 = CategoryLinear(hidden_dim, out_dim)

    def forward(self, x, embodiment):
        return self.layer2(F.relu(self.layer1(x, embodiment)), embodiment)


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = CategoryLinear(132, 1536)
        self.W2 = CategoryLinear(3072, 1536)
        self.W3 = CategoryLinear(1536, 1536)

    def forward(self, actions, timestep, embodiment):
        encoded = self.W1(actions, embodiment)
        t = timestep[:, None].expand(-1, actions.shape[1]).float()
        frequency = -torch.arange(768, dtype=torch.float32, device=actions.device)
        frequency = frequency * (torch.log(torch.tensor(10000.0, device="cpu")) / 768)
        phase = t.unsqueeze(-1) * frequency.exp()
        time = torch.cat((phase.sin(), phase.cos()), dim=-1).to(encoded.dtype)
        hidden = self.W2(torch.cat((encoded, time), dim=-1), embodiment)
        return self.W3(hidden * torch.sigmoid(hidden), embodiment)


class TimestepEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(256, 1536)

    def forward(self, timestep):
        dtype = self.timestep_embedder.linear_1.weight.dtype
        return self.timestep_embedder(self.time_proj(timestep).to(dtype))


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x, time):
        scale, shift = self.linear(F.silu(time)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, head_dim: int, *, adaptive: bool, cross_dim=None):
        super().__init__()
        self.norm1 = (AdaLayerNorm(dim) if adaptive else nn.LayerNorm(dim, eps=1e-5))
        self.attn1 = Attention(query_dim=dim, heads=32, dim_head=head_dim,
                               bias=True, cross_attention_dim=cross_dim, dropout=0.2)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=not adaptive)
        self.ff = FeedForward(dim, activation_fn="gelu-approximate", dropout=0.2,
                              final_dropout=True)
        self.adaptive = adaptive

    def forward(self, x, *, context=None, mask=None, time=None):
        normalized = self.norm1(x, time) if self.adaptive else self.norm1(x)
        x = self.attn1(normalized, encoder_hidden_states=context, attention_mask=mask) + x
        return self.ff(self.norm3(x)) + x


class VisionLanguageRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(2048, 64, adaptive=False) for _ in range(4)])

    def forward(self, x):
        x = x.contiguous()
        for block in self.transformer_blocks:
            x = block(x)
        return x


class AlternateVLDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep_encoder = TimestepEncoder()
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(1536, 48, adaptive=True, cross_dim=2048 if i % 2 == 0 else None)
            for i in range(32)
        ])
        self.norm_out = nn.LayerNorm(1536, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(1536, 3072)
        self.proj_out_2 = nn.Linear(1536, 1024)

    def forward(self, x, context, timestep, image_mask, attention_mask):
        time = self.timestep_encoder(timestep)
        x, context = x.contiguous(), context.contiguous()
        image = image_mask & attention_mask
        text = ~image_mask & attention_mask
        for i, block in enumerate(self.transformer_blocks):
            if i % 2:
                x = block(x, time=time)
            else:
                x = block(x, context=context, mask=text if i % 4 == 0 else image, time=time)
        shift, scale = self.proj_out_1(F.silu(time)).chunk(2, dim=1)
        return self.proj_out_2(self.norm_out(x) * (1 + scale[:, None]) + shift[:, None])


class ActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AlternateVLDiT()
        self.state_encoder = CategoryMLP(132, 1024, 1536)
        self.action_encoder = ActionEncoder()
        self.action_decoder = CategoryMLP(1024, 1024, 132)
        self.vlln = nn.LayerNorm(2048)
        self.vl_self_attention = VisionLanguageRefiner()
        self.position_embedding = nn.Embedding(1024, 1536)

    def forward(self, backbone, state, noise, embodiment, image_mask, attention_mask, *, steps):
        """Return BF16 [1,40,132] actions and first velocity, without modifying inputs.

        Noise is an explicit input so official/eager/captured runs use the same
        sample. All tensor inputs are on one device; execution is capture-safe.
        """
        context = self.vl_self_attention(self.vlln(backbone))
        state_features = self.state_encoder(state.view(state.shape[0], 1, -1), embodiment)
        actions = noise.clone()
        strength = torch.ones_like(actions)
        first_velocity = None
        for step in range(steps):
            timestep = torch.full((actions.shape[0],), int(step / float(steps) * 1000),
                                  device=actions.device)
            encoded = self.action_encoder(actions, timestep, embodiment)
            positions = torch.arange(actions.shape[1], device=actions.device)
            encoded = encoded + self.position_embedding(positions).unsqueeze(0)
            hidden = self.model(torch.cat((state_features, encoded), dim=1), context,
                                timestep, image_mask, attention_mask)
            velocity = self.action_decoder(hidden, embodiment)[:, -40:]
            if step == 0:
                first_velocity = velocity
            actions = actions + (1.0 / steps) * velocity * strength
        return actions, first_velocity


PREFIXES = {
    "vision": "backbone.model.model.visual.",
    "backbone": "backbone.model.model.language_model.",
    "action": "action_head.",
}


def make_modules(*, device="meta") -> dict[str, nn.Module]:
    """Construct the LIBERO checkpoint architecture; meta construction allocates no weights."""
    vision = Qwen3VLVisionConfig(depth=24, hidden_size=1024, intermediate_size=4096,
        num_heads=16, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        out_hidden_size=2048, num_position_embeddings=2304,
        deepstack_visual_indexes=[5, 11, 17], hidden_act="gelu_pytorch_tanh")
    text = Qwen3VLTextConfig(hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=16, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, rms_norm_eps=1e-6, vocab_size=151936,
        max_position_embeddings=262144, rope_theta=5000000,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [24, 20, 20],
                      "rope_type": "default"})
    vision._attn_implementation = text._attn_implementation = "sdpa"
    with torch.device(device):
        modules = {"vision": Qwen3VLVisionModel(vision),
                   "backbone": Qwen3VLTextModel(text), "action": ActionHead()}
    return {name: module.eval().requires_grad_(False) for name, module in modules.items()}


def bind_module(module: nn.Module, weights, *, prefix: str, device):
    """Bind runner-owned BF16 weights without a second copy; restore analytical RoPE buffers."""
    module.load_state_dict({name: weights[prefix + name] for name in module.state_dict()},
                           strict=True, assign=True)
    if isinstance(module, Qwen3VLVisionModel):
        module.rotary_pos_emb = type(module.rotary_pos_emb)(32).to(device=device, dtype=torch.bfloat16)
        module.patch_embed.proj.register_forward_pre_hook(_channels_last_patch)
    elif isinstance(module, Qwen3VLTextModel):
        module.rotary_emb = type(module.rotary_emb)(config=module.config, device=device).to(
            device=device, dtype=torch.bfloat16)
    return module.eval().requires_grad_(False)


def _channels_last_patch(_module, inputs):
    # Match the official Qwen3Backbone's PyTorch Conv3d input layout.
    return (inputs[0].contiguous(memory_format=torch.channels_last_3d),)
