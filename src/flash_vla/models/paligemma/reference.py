# SPDX-FileCopyrightText: Copyright 2024 Google Inc., HuggingFace Inc. and Physical Intelligence
# SPDX-License-Identifier: Apache-2.0
"""PaliGemma with an action expert, in plain torch: what Pi0 and Pi0.5 share.

Translated from OpenPI (github.com/Physical-Intelligence/openpi, revision
215abfb), `src/openpi/models_pytorch/{pi0_pytorch.py, gemma_pytorch.py}`, and
the Transformers 4.53.2 models OpenPI patches
(`models_pytorch/transformers_replace/models/{siglip,paligemma,gemma}`). Only
inference is translated: no dropout, no gradient checkpointing, no training
masks.

The model is a SigLIP So400m vision tower and a linear projector, a Gemma-2B
language backbone over the prefix (image tokens, then prompt tokens), and a
Gemma-300M action expert that attends over the backbone's cached keys and
values plus its own tokens. The two Gemma stacks share the attention geometry
(8 query heads, one KV head of width 256) and never mix weights.

Upstream's inference numerics, which this file reproduces:

- parameters are bfloat16, except the ones `FLOAT32_SELECTORS` names and the
  model's own heads (projections outside PaliGemma-with-expert), which stay
  float32 (`gemma_pytorch.py`, `to_bfloat16_for_selected_params`);
- the vision embeddings run in float32 and the encoder in bfloat16 (the
  patched `SiglipVisionTransformer` casts to the first layer's dtype);
- the image features are NOT divided by sqrt(width) (OpenPI removed that line
  from `PaliGemmaModel.get_image_features`), and the Gemma stacks apply no
  embedding normalizer (the prompt is scaled by sqrt(width) in `embed_prefix`);
- SigLIP attention goes through PyTorch SDPA, Transformers' default; the Gemma
  attention is eager (OpenPI forces it at inference): scores in the
  activation dtype, the float32 additive mask added, softmax in float32;
- RMSNorm reduces in float32 and scales by (1 + weight) in float32; the
  adaptive RMSNorm of Pi0.5's expert modulates in float32 and gates the
  residual.

Deliberate differences from the PyTorch upstream, none of them in the model's math:

- RoPE's inverse frequencies stay float32. Upstream's `Module.to(bfloat16)`
  also rounds Gemma's `inv_freq` buffer to bfloat16, which moves a prefix
  key's phase by up to ~1.3 rad at position 967; the JAX OpenPI the
  checkpoints were trained with computes them in float32, and flash-vla's own
  OpenPI oracle restores that (`models.pi05.openpi.restore_rope_precision`).
- float32 matmuls run in float32: OpenPI calls
  `torch.set_float32_matmul_precision("high")`, so on CUDA its heads, adaptive
  norms' dense layers and RoPE outer product may run in TF32.
- eager execution: upstream compiles `sample_actions` by default
  (`pytorch_compile_mode="max-autotune"`), fusing elementwise chains and so
  dropping some intermediate bfloat16 roundings.
- the views go through SigLIP as one batch, where upstream runs one call per
  view: the same math, not the same bits.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..official import Precision, bind, bind_parts

#: Additive bias of a key a query may not attend to (`pi0_pytorch.py`,
#: `_prepare_attention_masks_4d`).
BLOCKED = -2.3819763e38
#: Gemma's `rms_norm_eps` and SigLIP's `layer_norm_eps`.
RMS_EPS = 1e-6
LAYER_NORM_EPS = 1e-6
#: Gemma's default RoPE base.
ROPE_THETA = 10_000.0
#: PaliGemma's vocabulary (`gemma_pytorch.py`, `vlm_config_hf._vocab_size`).
VOCABULARY = 257152
#: Substrings of the official names OpenPI keeps in float32 when the rest of
#: PaliGemma-with-expert runs in bfloat16 (`to_bfloat16_for_selected_params`).
FLOAT32_SELECTORS = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)
#: The official prefix of each shared part.
PREFIXES = {
    "vision": "paligemma_with_expert.paligemma.model.vision_tower.vision_model.",
    "projector": "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.",
    "embedding": "paligemma_with_expert.paligemma.model.language_model.embed_tokens.",
    "backbone": "paligemma_with_expert.paligemma.model.language_model.",
    "expert": "paligemma_with_expert.gemma_expert.model.",
}

#: One layer's keys and values after RoPE: [batch, kv_heads, tokens, head_dim] each.
KeyValue = tuple[torch.Tensor, torch.Tensor]
#: RoPE cosines and sines: [batch, tokens, head_dim] each.
Rotary = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class SiglipGeometry:
    """SigLIP So400m/14 at 224 px, as PaliGemma configures it."""
    width: int = 1152
    depth: int = 27
    mlp_dim: int = 4304
    heads: int = 16
    patch: int = 14
    tokens: int = 256
    channels: int = 3


@dataclass(frozen=True)
class GemmaGeometry:
    """One Gemma stack (`openpi.models.gemma.get_config`)."""
    width: int
    depth: int
    mlp_dim: int
    heads: int = 8
    kv_heads: int = 1
    head_dim: int = 256


SIGLIP = SiglipGeometry()
GEMMA_2B = GemmaGeometry(width=2048, depth=18, mlp_dim=16384)
GEMMA_300M = GemmaGeometry(width=1024, depth=18, mlp_dim=4096)


# -- vision ------------------------------------------------------------------


class SiglipEmbeddings(nn.Module):
    def __init__(self, geometry: SiglipGeometry) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(geometry.channels, geometry.width,
                                         kernel_size=geometry.patch, stride=geometry.patch)
        self.position_embedding = nn.Embedding(geometry.tokens, geometry.width)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """[images, channels, 224, 224] -> [images, tokens, width], in the patch weight's dtype."""
        patches = self.patch_embedding(pixels.to(self.patch_embedding.weight.dtype))
        return patches.flatten(2).transpose(1, 2) + self.position_embedding.weight


class SiglipAttention(nn.Module):
    def __init__(self, geometry: SiglipGeometry) -> None:
        super().__init__()
        self.heads = geometry.heads
        self.q_proj = nn.Linear(geometry.width, geometry.width)
        self.k_proj = nn.Linear(geometry.width, geometry.width)
        self.v_proj = nn.Linear(geometry.width, geometry.width)
        self.out_proj = nn.Linear(geometry.width, geometry.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images, tokens, width = x.shape
        head_dim = width // self.heads
        split = (images, tokens, self.heads, head_dim)
        query = self.q_proj(x).view(split).transpose(1, 2)
        key = self.k_proj(x).view(split).transpose(1, 2)
        value = self.v_proj(x).view(split).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, scale=head_dim**-0.5)
        return self.out_proj(attended.transpose(1, 2).reshape(images, tokens, width))


class SiglipMLP(nn.Module):
    def __init__(self, geometry: SiglipGeometry) -> None:
        super().__init__()
        self.fc1 = nn.Linear(geometry.width, geometry.mlp_dim)
        self.fc2 = nn.Linear(geometry.mlp_dim, geometry.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SiglipLayer(nn.Module):
    def __init__(self, geometry: SiglipGeometry) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(geometry.width, eps=LAYER_NORM_EPS)
        self.self_attn = SiglipAttention(geometry)
        self.layer_norm2 = nn.LayerNorm(geometry.width, eps=LAYER_NORM_EPS)
        self.mlp = SiglipMLP(geometry)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SiglipEncoder(nn.Module):
    def __init__(self, geometry: SiglipGeometry) -> None:
        super().__init__()
        self.layers = nn.ModuleList(SiglipLayer(geometry) for _ in range(geometry.depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SiglipVision(nn.Module):
    """The vision tower: its parameter names are the official ones below `PREFIXES["vision"]`."""

    def __init__(self, geometry: SiglipGeometry = SIGLIP) -> None:
        super().__init__()
        self.embeddings = SiglipEmbeddings(geometry)
        self.encoder = SiglipEncoder(geometry)
        self.post_layernorm = nn.LayerNorm(geometry.width, eps=LAYER_NORM_EPS)

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The encoder's last hidden state before and after the final LayerNorm."""
        embedded = self.embeddings(pixels)
        hidden = self.encoder(embedded.to(self.encoder.layers[0].self_attn.q_proj.weight.dtype))
        return hidden, self.post_layernorm(hidden)


# -- Gemma -------------------------------------------------------------------


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps), the mean in float32; the product is float32."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + RMS_EPS)


class GemmaRMSNorm(nn.Module):
    """RMSNorm scaled by (1 + weight); it gates nothing."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> tuple[torch.Tensor, None]:
        return (rms_normalize(x) * (1.0 + self.weight.float())).to(x.dtype), None


class AdaptiveRMSNorm(nn.Module):
    """Pi0.5's adaptive RMSNorm: scale, shift and a residual gate from the condition."""

    def __init__(self, width: int, cond_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(cond_dim, 3 * width)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """`cond` is required; it is optional only to share the plain norm's
        signature, and an adaptive stack is always given one (`expert_pass`)."""
        scale, shift, gate = self.dense(cond).unsqueeze(1).chunk(3, dim=-1)
        normed = rms_normalize(x) * (1 + scale.float()) + shift.float()
        return normed.to(x.dtype), gate.to(x.dtype)


def rotary(positions: torch.Tensor, head_dim: int, dtype: torch.dtype) -> Rotary:
    """Gemma's RoPE tables for integer `positions` [batch, tokens], computed in
    float32 and cast to the activation dtype, as `GemmaRotaryEmbedding` does.
    The inverse frequencies are float32 (see the module docstring), computed
    on the CPU as upstream's are at construction."""
    exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).float()
    inverse_frequency = (1.0 / (ROPE_THETA ** (exponents / head_dim))).to(positions.device)
    angles = positions.float()[..., None] * inverse_frequency
    doubled = torch.cat((angles, angles), dim=-1)
    return doubled.cos().to(dtype), doubled.sin().to(dtype)


def rotate(x: torch.Tensor, table: Rotary) -> torch.Tensor:
    """Rotate [batch, heads, tokens, head_dim] by RoPE in half-split channel order."""
    cos, sin = (value.unsqueeze(1) for value in table)
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class GemmaAttention(nn.Module):
    def __init__(self, geometry: GemmaGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        queries, keys = geometry.heads * geometry.head_dim, geometry.kv_heads * geometry.head_dim
        self.q_proj = nn.Linear(geometry.width, queries, bias=False)
        self.k_proj = nn.Linear(geometry.width, keys, bias=False)
        self.v_proj = nn.Linear(geometry.width, keys, bias=False)
        self.o_proj = nn.Linear(queries, geometry.width, bias=False)

    def forward(self, x: torch.Tensor, *, table: Rotary, mask: torch.Tensor,
                past: KeyValue | None) -> tuple[torch.Tensor, KeyValue]:
        """Attend over `past` (a cached prefix) and `x`'s own keys; return the
        output and `x`'s own keys and values after RoPE."""
        batch, tokens, _ = x.shape
        head_dim, groups = self.geometry.head_dim, self.geometry.heads // self.geometry.kv_heads
        query = rotate(self.q_proj(x).view(batch, tokens, -1, head_dim).transpose(1, 2), table)
        key = rotate(self.k_proj(x).view(batch, tokens, -1, head_dim).transpose(1, 2), table)
        value = self.v_proj(x).view(batch, tokens, -1, head_dim).transpose(1, 2)
        keys, values = (key, value) if past is None else (torch.cat((past[0], key), dim=2),
                                                          torch.cat((past[1], value), dim=2))
        keys, values = (keys.repeat_interleave(groups, dim=1), values.repeat_interleave(groups, dim=1))
        scores = torch.matmul(query, keys.transpose(2, 3)) * head_dim**-0.5 + mask
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(probabilities, values).transpose(1, 2).reshape(batch, tokens, -1)
        return self.o_proj(attended), (key, value)


class GemmaMLP(nn.Module):
    def __init__(self, geometry: GemmaGeometry) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(geometry.width, geometry.mlp_dim, bias=False)
        self.up_proj = nn.Linear(geometry.width, geometry.mlp_dim, bias=False)
        self.down_proj = nn.Linear(geometry.mlp_dim, geometry.width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaLayer(nn.Module):
    def __init__(self, geometry: GemmaGeometry, adaptive: bool) -> None:
        super().__init__()
        self.input_layernorm = (AdaptiveRMSNorm(geometry.width, geometry.width) if adaptive
                                else GemmaRMSNorm(geometry.width))
        self.self_attn = GemmaAttention(geometry)
        self.post_attention_layernorm = (AdaptiveRMSNorm(geometry.width, geometry.width) if adaptive
                                         else GemmaRMSNorm(geometry.width))
        self.mlp = GemmaMLP(geometry)

    def forward(self, x: torch.Tensor, *, table: Rotary, mask: torch.Tensor,
                cond: torch.Tensor | None, past: KeyValue | None) -> tuple[torch.Tensor, KeyValue]:
        normed, gate = self.input_layernorm(x, cond)
        attended, own = self.self_attn(normed, table=table, mask=mask, past=past)
        x = x + attended if gate is None else x + attended * gate
        normed, gate = self.post_attention_layernorm(x, cond)
        updated = self.mlp(normed)
        return (x + updated if gate is None else x + updated * gate), own


class GemmaStack(nn.Module):
    """One Gemma decoder stack without its embedding: layers and final norm."""

    def __init__(self, geometry: GemmaGeometry, adaptive: bool) -> None:
        super().__init__()
        self.geometry = geometry
        self.layers = nn.ModuleList(GemmaLayer(geometry, adaptive) for _ in range(geometry.depth))
        self.norm = (AdaptiveRMSNorm(geometry.width, geometry.width) if adaptive
                     else GemmaRMSNorm(geometry.width))

    def forward(self, x: torch.Tensor, *, positions: torch.Tensor, mask: torch.Tensor, depth: int,
                cond: torch.Tensor | None = None,
                past: list[KeyValue] | None = None) -> tuple[torch.Tensor, list[KeyValue]]:
        """Run the first `depth` layers and the final norm; return the normed
        output and each layer's own keys and values.

        `mask` is the additive float32 mask [batch, 1, tokens, keys]; `past`
        holds a cached prefix per layer, attended before `x`'s own keys.
        """
        x = x.to(self.layers[0].self_attn.q_proj.weight.dtype)
        table = rotary(positions, self.geometry.head_dim, x.dtype)
        cache: list[KeyValue] = []
        for index, layer in enumerate(self.layers[:depth]):
            x, own = layer(x, table=table, mask=mask, cond=cond,
                       past=None if past is None else past[index])
            cache.append(own)
        normed, _ = self.norm(x, cond)
        return normed, cache


# -- masks -------------------------------------------------------------------


def attention_mask(*, pad_masks: torch.Tensor, block_starts: torch.Tensor) -> torch.Tensor:
    """big_vision's `make_att_2d_masks`: a token attends to every valid token
    whose block index (the cumulative sum of `block_starts`) is not greater
    than its own. Boolean [batch, queries, keys]."""
    blocks = torch.cumsum(block_starts, dim=1)
    return (blocks[:, None, :] <= blocks[:, :, None]) & (pad_masks[:, None, :] & pad_masks[:, :, None])


def additive(allowed: torch.Tensor) -> torch.Tensor:
    """A boolean [batch, queries, keys] mask as the float32 additive [batch, 1, queries, keys]."""
    return torch.where(allowed[:, None], 0.0, BLOCKED)


# -- the shared model ----------------------------------------------------------


@dataclass(frozen=True)
class Prefix:
    """The backbone's pass over the prefix: each layer's keys and values after
    RoPE, and which prefix positions are valid."""
    cache: tuple[KeyValue, ...]
    pad_masks: torch.Tensor


class PaliGemmaWithExpert(nn.Module):
    """The vision tower, projector, prompt embedding, backbone and action
    expert of Pi0 and Pi0.5; `parts()` maps each to its `PREFIXES` entry."""

    def __init__(self, *, adaptive_expert: bool) -> None:
        super().__init__()
        self.vision = SiglipVision()
        self.projector = nn.Linear(SIGLIP.width, GEMMA_2B.width)
        self.embedding = nn.Embedding(VOCABULARY, GEMMA_2B.width)
        self.backbone = GemmaStack(GEMMA_2B, adaptive=False)
        self.expert = GemmaStack(GEMMA_300M, adaptive=adaptive_expert)

    def parts(self) -> dict[str, nn.Module]:
        return {"vision": self.vision, "projector": self.projector, "embedding": self.embedding,
                "backbone": self.backbone, "expert": self.expert}

    def embed_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[views, 224, 224, 3] in [-1, 1] -> the vision tower's hidden state
        before its final LayerNorm, and the projected image tokens [views, 256, 2048]."""
        hidden, normed = self.vision(images.permute(0, 3, 1, 2))
        return hidden, self.projector(normed)

    def prefix(self, image_tokens: torch.Tensor, *, image_masks: torch.Tensor,
               prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, depth: int) -> Prefix:
        """The backbone over [image tokens of every view, prompt tokens], one
        batch, full attention among valid tokens (`embed_prefix`, `sample_actions`)."""
        views, tokens, width = image_tokens.shape
        prompt = self.embedding(prompt_ids) * math.sqrt(self.embedding.embedding_dim)
        embeddings = torch.cat((image_tokens.reshape(1, views * tokens, width), prompt[None]), dim=1)
        pad_masks = torch.cat((image_masks[:, None].expand(views, tokens).reshape(1, -1),
                               prompt_mask[None]), dim=1)
        positions = torch.cumsum(pad_masks, dim=1) - 1
        mask = additive(attention_mask(pad_masks=pad_masks, block_starts=torch.zeros_like(pad_masks)))
        # The backbone's own output is never read: the expert reads its cache.
        _, cache = self.backbone(embeddings, positions=positions, mask=mask, depth=depth)
        return Prefix(cache=tuple(cache), pad_masks=pad_masks)

    def expert_pass(self, prefix: Prefix, suffix: torch.Tensor, *, block_starts: torch.Tensor,
                    cond: torch.Tensor | None, depth: int) -> tuple[torch.Tensor, list[KeyValue]]:
        """The expert over its suffix tokens [1, tokens, width], attending over
        the valid prefix and, within the suffix, by block (`denoise_step`)."""
        tokens = suffix.shape[1]
        suffix_pad = torch.ones((1, tokens), dtype=torch.bool, device=suffix.device)
        prefix_allowed = prefix.pad_masks[:, None, :].expand(1, tokens, prefix.pad_masks.shape[1])
        allowed = torch.cat((prefix_allowed,
                             attention_mask(pad_masks=suffix_pad, block_starts=block_starts)), dim=2)
        positions = prefix.pad_masks.sum(dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
        return self.expert(suffix, positions=positions, mask=additive(allowed), depth=depth,
                           cond=cond, past=prefix.cache)


def sinusoidal(time: torch.Tensor, dimension: int) -> torch.Tensor:
    """OpenPI's timestep embedding (`create_sinusoidal_pos_embedding`, periods
    4e-3 to 4.0), computed in float64 and returned in `time`'s dtype."""
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=time.device)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    phase = (1.0 / period * 2 * math.pi)[None, :] * time[:, None]
    return torch.cat((phase.sin(), phase.cos()), dim=1).to(time.dtype)


def bind_upstream(parts: Mapping[str, nn.Module], *, prefixes: Mapping[str, str],
                  weights: Mapping[str, torch.Tensor], precision: Precision) -> None:
    """Bind official tensors to a PaliGemma-with-expert model's parts -- its
    `PREFIXES`' parts, the model's own heads as part `head` -- in OpenPI's
    inference dtypes (`bfloat16`: PaliGemma-with-expert in bfloat16 except the
    `FLOAT32_SELECTORS`, the heads in float32) or entirely in float32."""
    if precision == "float32":
        bind_parts(parts, prefixes=prefixes, weights=weights, precision=precision)
        return
    bind(parts, prefixes=prefixes, weights=weights)
    for part, module in parts.items():
        for name, parameter in module.named_parameters():
            official = prefixes[part] + name
            keep = part == "head" or any(selector in official for selector in FLOAT32_SELECTORS)
            parameter.data = parameter.data.to(torch.float32 if keep else torch.bfloat16)


__all__ = ["BLOCKED", "FLOAT32_SELECTORS", "GEMMA_2B", "GEMMA_300M", "GemmaGeometry",
           "GemmaStack", "KeyValue", "PREFIXES", "PaliGemmaWithExpert", "Prefix", "SIGLIP",
           "SiglipGeometry", "SiglipVision", "VOCABULARY", "additive", "attention_mask",
           "bind_upstream", "rotary", "sinusoidal"]
