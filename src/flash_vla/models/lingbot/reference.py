# SPDX-FileCopyrightText: Copyright 2026 The LingBot-VLA Authors
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
"""LingBot-VLA end to end, in plain torch, on the official checkpoint.

Translated from lingbot-vla (github.com/robbyant/lingbot-vla, revision
4eb34b7), `lingbotvla/models/vla/pi0/`: `modeling_lingbot_vla.py`
(`FlowMatching.embed_prefix`, `embed_suffix`, `sample_actions` and
`predict_velocity`; `QwenvlWithExpertModel.forward`; the expert's
`Qwen2DecoderLayer`, `AdaRMSNorm` and `FixQwen2RMSNorm`), `qwenvl_in_vla.py`
(the Qwen2.5-VL vision tower and decoder layer) and `utils.py` (`apply_rope`,
`our_eager_attention_forward`, `make_att_2d_masks`,
`create_sinusoidal_pos_embedding`). The configuration is
`QwenvlWithExpertConfig`'s defaults with the checkpoint's own switches:
time-conditioned expert norms (`adanorm_time`), the time fused into the
action tokens rather than projected apart (`separate_time_proj=False`), a
plain final expert norm and no query/key norms. Only inference is translated.

The three stages:

    vision encoder  Qwen2.5-VL's vision tower over every view's 16x16 patches:
                    32 blocks with 2D RoPE, attending within windows of 4x4
                    merged tokens except blocks 7, 15, 23 and 31, which attend
                    across the image; the 2x2 patch merger's 64 tokens
                    [views, 64, 2048]
    LLM backbone    the visual tokens and the prompt's embeddings through the
                    Qwen2.5 language model, bidirectionally among valid
                    tokens; every layer's keys and values after RoPE
    action expert   `steps` Euler steps of the 768-wide Qwen2 expert from
                    `noise` at t = 1 down to t = 0. Each step's suffix is the
                    robot state as one token, then the noisy chunk fused with
                    the timestep's sinusoid by the action-time MLP; the state
                    token sees the prefix and itself, the chunk the prefix, the
                    state and the whole chunk. The expert's layer norms are
                    modulated by the sinusoid (AdaRMSNorm), and it attends over
                    the backbone's cached keys and values in every layer

The inputs are already prepared, as the model receives them: the views'
patches [views, 256, 1176] and each view's patch grid, a per-view validity
mask, the tokenized prompt [1, 72] and its mask, the state [1, 75] and the
noise [1, 50, 75]. Weights are the official checkpoint's tensors by their
official names (`PREFIXES`); every one of them is bound, and all but the
backbone's final norm, whose output upstream discards too, are run.

Upstream's inference numerics, which this file reproduces:

- the deployment casts the whole model to bfloat16 (`deploy/lingbot_vla_policy.py`,
  `vla.to(torch.bfloat16)`), buffers included: the vision rotary table is
  built from bfloat16 inverse frequencies, in bfloat16;
- vision RoPE is applied in float32 and cast back; RMSNorm normalizes in
  float32 and scales in the input's dtype; the expert's AdaRMSNorm scales,
  modulates and shifts in float32;
- the joint attention runs in float32: queries, keys and values are widened
  after their projections, rotated in float32 by `apply_rope` -- at its
  default wavelength 10,000, not the configurations' `rope_theta` -- at the
  positions of the valid tokens, and the prefix's keys and values are cached
  in float32; masked scores take -2.3819763e38 before a float32 softmax, and
  the output is cast back to bfloat16 for the output projection;
- the timestep's sinusoid is float32, cast to the activation dtype; the Euler
  schedule accumulates in the activation dtype.

Deliberate differences from upstream, none of them in the model's math:

- the vision attention's FlashAttention kernels (its rotary embedding and
  variable-length attention) are the same math in torch: RoPE in float32,
  SDPA per window and per image. The same math, not the same bits;
- float32 matmuls run in float32. Upstream's policy calls
  `torch.set_float32_matmul_precision("high")` (`LingbotVlaPolicy.__init__`),
  so its float32 joint attention runs on TF32 inputs; the deployment also
  sets `cudnn.deterministic`, which chooses convolution algorithms, not math.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..official import PRECISION_DTYPES, bind, random_official
from .spec import (
    ACTION_DIM,
    BACKBONE_DIM,
    BACKBONE_FFN,
    EXPERT_DIM,
    EXPERT_FFN,
    FULL_ATTENTION_BLOCKS,
    HEAD_DIM,
    KV_HEADS,
    LAYERS,
    PATCH_SIZE,
    QUERY_HEADS,
    STATE_DIM,
    VISION_DIM,
    VISION_FFN,
    VISION_HEAD_DIM,
    VISION_HEADS,
    VISION_LAYERS,
    VOCABULARY,
    WINDOW_SIZE,
)

#: Official prefix of each part: Qwen2.5-VL's vision tower and language model,
#: the Qwen2 action expert, and the heads at the top of `model`.
PREFIXES = {
    "vision": "model.qwenvl_with_expert.qwenvl.visual.",
    "backbone": "model.qwenvl_with_expert.qwenvl.model.",
    "expert": "model.qwenvl_with_expert.qwen_expert.model.",
    "head": "model.",
}
#: Frames per temporal patch of the Conv3d embedding.
TEMPORAL_PATCH = 2
#: The merger's neighbourhood side: 2x2 patches become one visual token.
MERGE = 2
#: Side of an attention window, in merged tokens.
WINDOW = WINDOW_SIZE // MERGE // PATCH_SIZE
VISION_ROPE_THETA = 10000.0
RMS_EPS = 1e-6

#: `apply_rope`'s default wavelength, which the joint attention uses.
ROPE_WAVELENGTH = 10_000.0
#: The score of a key a query may not attend to (`our_eager_attention_forward`).
BLOCKED = -2.3819763e38
#: Periods of the timestep's sinusoid (`embed_suffix`).
TIME_MIN_PERIOD = 4e-3
TIME_MAX_PERIOD = 4.0

#: Keys and values of one layer, [batch, tokens, KV_HEADS, HEAD_DIM], float32.
KeyValue = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    """Qwen2's RMSNorm (also `FixQwen2RMSNorm`)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Normalized in float32, scaled in the input's dtype."""
        widened = hidden.float()
        normalized = widened * torch.rsqrt(widened.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return self.weight * normalized.to(hidden.dtype)


class AdaRMSNorm(nn.Module):
    """RMSNorm, then a scale and shift projected from the time conditioning."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.gamma = nn.Linear(width, width)
        self.beta = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """`hidden` [batch, tokens, width] under `cond` [batch, width]; float32 inside."""
        widened = hidden.float()
        normalized = self.weight * (widened * torch.rsqrt(widened.pow(2).mean(-1, keepdim=True) + RMS_EPS))
        gamma, beta = self.gamma(cond)[:, None].float(), self.beta(cond)[:, None].float()
        return ((1 + gamma) * normalized + beta).to(hidden.dtype)


class GatedMLP(nn.Module):
    def __init__(self, width: int, ffn: int, *, bias: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, ffn, bias=bias)
        self.up_proj = nn.Linear(width, ffn, bias=bias)
        self.down_proj = nn.Linear(ffn, width, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


@dataclass(frozen=True)
class VisionTables:
    """What the vision tower derives from the images' patch grids
    (`VisionTower.tables`): upstream's `preprcess_grid_thw`, with the rotary
    table already in window order."""
    #: The 2D rotary table's cos and sin in window order [patches, 80].
    cos: torch.Tensor
    sin: torch.Tensor
    #: The merged tokens in window order, and the order that restores them.
    window_order: torch.Tensor
    restore_order: torch.Tensor
    #: Patches of each window, and of each image, attended separately.
    window_lengths: tuple[int, ...]
    image_lengths: tuple[int, ...]


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = (TEMPORAL_PATCH, PATCH_SIZE, PATCH_SIZE)
        self.proj = nn.Conv3d(3, VISION_DIM, kernel_size=kernel, stride=kernel, bias=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Flattened patches [..., 3 * 2 * 14 * 14] -> embeddings [patches, 1280]."""
        volumes = patches.reshape(-1, 3, TEMPORAL_PATCH, PATCH_SIZE, PATCH_SIZE).to(self.proj.weight.dtype)
        return self.proj(volumes).view(-1, VISION_DIM)


class VisionAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(VISION_DIM, 3 * VISION_DIM)
        self.proj = nn.Linear(VISION_DIM, VISION_DIM)

    def forward(self, hidden: torch.Tensor, *, cos: torch.Tensor, sin: torch.Tensor,
                lengths: tuple[int, ...]) -> torch.Tensor:
        """Attention within each run of `lengths` patches of `hidden` [patches, 1280]."""
        tokens = hidden.shape[0]
        query, key, value = (self.qkv(hidden).reshape(tokens, 3, VISION_HEADS, -1)
                             .permute(1, 0, 2, 3).unbind(0))
        # `apply_rotary_pos_emb_flashatt`: rotated in float32 over half-split
        # channels by the table's first half, cast back.
        half_cos, half_sin = (table.chunk(2, dim=-1)[0].float()[:, None] for table in (cos, sin))
        rotated = []
        for states in (query, key):
            first, second = states.float().chunk(2, dim=-1)
            rotated.append(torch.cat((first * half_cos - second * half_sin,
                                      second * half_cos + first * half_sin), dim=-1).to(states.dtype))
        query, key = rotated
        attended = [F.scaled_dot_product_attention(run_query.transpose(0, 1)[None],
                                                   run_key.transpose(0, 1)[None],
                                                   run_value.transpose(0, 1)[None])[0].transpose(0, 1)
                    for run_query, run_key, run_value in zip(
                        *(states.split(lengths) for states in (query, key, value)))]
        return self.proj(torch.cat(attended).reshape(tokens, -1))


class VisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = RMSNorm(VISION_DIM)
        self.norm2 = RMSNorm(VISION_DIM)
        self.attn = VisionAttention()
        self.mlp = GatedMLP(VISION_DIM, VISION_FFN, bias=True)

    def forward(self, hidden: torch.Tensor, *, cos: torch.Tensor, sin: torch.Tensor,
                lengths: tuple[int, ...]) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), cos=cos, sin=sin, lengths=lengths)
        return hidden + self.mlp(self.norm2(hidden))


class PatchMerger(nn.Module):
    """Each 2x2 neighbourhood's normalized patches as one 5120-wide vector,
    through an exact-GELU MLP to the backbone width. `mlp` is upstream's
    `Sequential`: the projection in (`mlp.0`), the GELU, the projection out
    (`mlp.2`)."""

    def __init__(self) -> None:
        super().__init__()
        merged = VISION_DIM * MERGE**2
        self.ln_q = RMSNorm(VISION_DIM)
        self.mlp = nn.ModuleDict({"0": nn.Linear(merged, merged), "2": nn.Linear(merged, BACKBONE_DIM)})

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """[patches, 1280] -> [patches / 4, 2048]."""
        normalized = self.ln_q(hidden).view(-1, VISION_DIM * MERGE**2)
        return self.mlp["2"](F.gelu(self.mlp["0"](normalized)))


class VisionTower(nn.Module):
    """`Qwen2_5_VisionTransformerPretrainedModel`, its forward split into the
    grids' tables (`tables`) and the tower that consumes them (`forward`)."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList(VisionBlock() for _ in range(VISION_LAYERS))
        self.merger = PatchMerger()
        # Analytic, so not in the checkpoint: built here on the host, in
        # float32, and cast with the model (`load`).
        rotary_dim = VISION_HEAD_DIM // 2
        exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")
        self.register_buffer("inverse_frequency", 1.0 / (VISION_ROPE_THETA ** (exponents / rotary_dim)),
                             persistent=False)

    def tables(self, grid: Sequence[tuple[int, int, int]]) -> VisionTables:
        """`rot_pos_emb` and `get_window_index` for images of patch grids
        `grid` (frames, rows, columns each), built from host-side index lists
        as upstream builds them."""
        coordinates: list[torch.Tensor] = []
        orders: list[torch.Tensor] = []
        window_lengths: list[int] = []
        offset = 0
        for frames, rows, columns in grid:
            # Every patch's (row, column), 2x2 neighbourhoods consecutive.
            neighbourhoods = (rows // MERGE, MERGE, columns // MERGE, MERGE)
            row_index = (torch.arange(rows)[:, None].expand(-1, columns).reshape(neighbourhoods)
                         .permute(0, 2, 1, 3).flatten())
            column_index = (torch.arange(columns)[None].expand(rows, -1).reshape(neighbourhoods)
                            .permute(0, 2, 1, 3).flatten())
            coordinates.append(torch.stack((row_index, column_index), dim=-1).repeat(frames, 1))
            # The merged tokens window by window, each window a 4x4 block of
            # them. Upstream pads a whole extra window where the side already
            # divides; its windows are empty and drop out of the lengths.
            merged_rows, merged_columns = rows // MERGE, columns // MERGE
            index = torch.arange(frames * merged_rows * merged_columns).reshape(
                frames, merged_rows, merged_columns)
            pad_rows, pad_columns = WINDOW - merged_rows % WINDOW, WINDOW - merged_columns % WINDOW
            windows_down = (merged_rows + pad_rows) // WINDOW
            windows_across = (merged_columns + pad_columns) // WINDOW
            padded = (F.pad(index, (0, pad_columns, 0, pad_rows), "constant", -100)
                      .reshape(frames, windows_down, WINDOW, windows_across, WINDOW)
                      .permute(0, 1, 3, 2, 4).reshape(-1, WINDOW * WINDOW))
            window_lengths.extend(count * MERGE**2 for count in (padded != -100).sum(-1).tolist() if count)
            flat = padded.flatten()
            orders.append(flat[flat != -100] + offset)
            offset += frames * merged_rows * merged_columns
        frequency = self.inverse_frequency
        longest = max(max(rows, columns) for _, rows, columns in grid)
        table = torch.outer(torch.arange(longest, device=frequency.device, dtype=frequency.dtype), frequency)
        angles = table[torch.cat(coordinates).to(frequency.device)].flatten(1)
        window_order = torch.cat(orders).to(frequency.device)
        # The rotary table follows the patches into window order.
        angles = angles.reshape(-1, MERGE**2, angles.shape[-1])[window_order].flatten(0, 1)
        doubled = torch.cat((angles, angles), dim=-1)
        return VisionTables(cos=doubled.cos(), sin=doubled.sin(), window_order=window_order,
                            restore_order=torch.argsort(window_order),
                            window_lengths=tuple(window_lengths),
                            image_lengths=tuple(rows * columns for frames, rows, columns in grid
                                                for _ in range(frames)))

    def forward(self, pixel_values: torch.Tensor, *, tables: VisionTables) -> torch.Tensor:
        """The views' patches [views, 256, 1176] -> the merged tokens, in
        image order, [views * 64, 2048]."""
        hidden = self.patch_embed(pixel_values)
        patches = hidden.shape[0]
        hidden = hidden.reshape(patches // MERGE**2, MERGE**2, -1)[tables.window_order].reshape(patches, -1)
        for index, block in enumerate(self.blocks):
            lengths = tables.image_lengths if index in FULL_ATTENTION_BLOCKS else tables.window_lengths
            hidden = block(hidden, cos=tables.cos, sin=tables.sin, lengths=lengths)
        return self.merger(hidden)[tables.restore_order]


class Attention(nn.Module):
    """The projections of a Qwen2 attention: biased queries, keys and values,
    an unbiased output; the attention itself is `attend`, joint across the
    backbone and the expert."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, QUERY_HEADS * HEAD_DIM)
        self.k_proj = nn.Linear(width, KV_HEADS * HEAD_DIM)
        self.v_proj = nn.Linear(width, KV_HEADS * HEAD_DIM)
        self.o_proj = nn.Linear(QUERY_HEADS * HEAD_DIM, width, bias=False)

    def project(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries [batch, tokens, 16, 128], keys and values [batch, tokens, 2, 128],
        widened to float32 as the joint attention takes them."""
        shape = (*normalized.shape[:-1], -1, HEAD_DIM)
        return (self.q_proj(normalized).view(shape).float(), self.k_proj(normalized).view(shape).float(),
                self.v_proj(normalized).view(shape).float())


class BackboneLayer(nn.Module):
    """`Qwen2_5_VLDecoderLayer`, split as the joint attention runs it."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention(BACKBONE_DIM)
        self.mlp = GatedMLP(BACKBONE_DIM, BACKBONE_FFN, bias=False)
        self.input_layernorm = RMSNorm(BACKBONE_DIM)
        self.post_attention_layernorm = RMSNorm(BACKBONE_DIM)

    def project(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.self_attn.project(self.input_layernorm(hidden))

    def finish(self, hidden: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        """The layer's output from its input and the attention's [batch, tokens, 2048]."""
        residual = self.self_attn.o_proj(attended.to(self.self_attn.o_proj.weight.dtype)) + hidden
        return self.mlp(self.post_attention_layernorm(residual)) + residual


class ExpertLayer(nn.Module):
    """The expert's `Qwen2DecoderLayer` with time-conditioned norms, split as
    the joint attention runs it."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention(EXPERT_DIM)
        self.mlp = GatedMLP(EXPERT_DIM, EXPERT_FFN, bias=False)
        self.input_layernorm = AdaRMSNorm(EXPERT_DIM)
        self.post_attention_layernorm = AdaRMSNorm(EXPERT_DIM)

    def project(self, hidden: torch.Tensor,
                cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.self_attn.project(self.input_layernorm(hidden, cond))

    def finish(self, hidden: torch.Tensor, attended: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.self_attn.o_proj(attended.to(self.self_attn.o_proj.weight.dtype)) + hidden
        return self.mlp(self.post_attention_layernorm(residual, cond)) + residual


class LanguageModel(nn.Module):
    """Qwen2.5-VL's language model. Its final `norm` is in the checkpoint and
    held here; the prefix pass only keeps the layers' keys and values."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCABULARY, BACKBONE_DIM)
        self.layers = nn.ModuleList(BackboneLayer() for _ in range(LAYERS))
        self.norm = RMSNorm(BACKBONE_DIM)


class ExpertModel(nn.Module):
    """The Qwen2 action expert, without the token embedding upstream deletes."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ExpertLayer() for _ in range(LAYERS))
        self.norm = RMSNorm(EXPERT_DIM)


class LingBotHead(nn.Module):
    """The projections outside the two towers: the state token, the chunk in,
    the action-time MLP that fuses the timestep into it, and the velocity out."""

    def __init__(self) -> None:
        super().__init__()
        self.state_proj = nn.Linear(STATE_DIM, EXPERT_DIM)
        self.action_in_proj = nn.Linear(ACTION_DIM, EXPERT_DIM)
        self.action_out_proj = nn.Linear(EXPERT_DIM, ACTION_DIM)
        self.action_time_mlp_in = nn.Linear(2 * EXPERT_DIM, EXPERT_DIM)
        self.action_time_mlp_out = nn.Linear(EXPERT_DIM, EXPERT_DIM)


def rotate(states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """`apply_rope`: `states` [batch, tokens, heads, 128] rotated at integer
    `positions` [batch, tokens] in float32, half-split channel pairs."""
    half = HEAD_DIM // 2
    exponents = (2.0 / HEAD_DIM) * torch.arange(half, dtype=torch.float32, device=states.device)
    radians = torch.einsum("bl,h->blh", positions.float(), 1.0 / ROPE_WAVELENGTH**exponents)[..., None, :]
    sin, cos = radians.sin(), radians.cos()
    first, second = states.float().split(half, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(states.dtype)


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
           mask: torch.Tensor) -> torch.Tensor:
    """`our_eager_attention_forward`: queries [batch, queries, 16, 128] over
    keys and values [batch, keys, 2, 128] -- each repeated to its 8 query
    heads -- where `mask` [batch, queries, keys] allows; [batch, queries, 2048]."""
    batch, queries = query.shape[:2]
    groups = QUERY_HEADS // KV_HEADS
    key, value = (states.repeat_interleave(groups, dim=2) for states in (key, value))
    scores = torch.einsum("bhqd,bhkd->bhqk", query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3))
    scores = scores * HEAD_DIM**-0.5
    probabilities = F.softmax(torch.where(mask[:, None], scores, BLOCKED), dim=-1).to(value.dtype)
    attended = torch.einsum("bhqk,bhkv->bhqv", probabilities, value.permute(0, 2, 1, 3))
    return attended.permute(0, 2, 1, 3).reshape(batch, queries, QUERY_HEADS * HEAD_DIM)


def sinusoid(time: torch.Tensor, width: int) -> torch.Tensor:
    """`create_sinusoidal_pos_embedding`: sin, then cos, of `time` [batch] at
    periods from 4e-3 to 4 on a float32 log scale -> [batch, width], float32."""
    fraction = torch.linspace(0.0, 1.0, width // 2, dtype=torch.float32, device=time.device)
    period = TIME_MIN_PERIOD * (TIME_MAX_PERIOD / TIME_MIN_PERIOD) ** fraction
    angles = (1.0 / period * 2 * math.pi)[None, :] * time[:, None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


@dataclass(frozen=True)
class Prefix:
    """The backbone's pass over the prefix, which every denoising step attends to."""
    #: Each layer's keys and values after RoPE.
    cache: tuple[KeyValue, ...]
    #: Which prefix positions hold a valid token [batch, prefix].
    pad_masks: torch.Tensor


@dataclass(frozen=True)
class LingBotOutputs:
    """Every stage's output, laid out as the engine's stage outputs are."""
    #: The merged visual tokens of each view [views, 64, 2048].
    vision_embeddings: torch.Tensor
    #: The backbone's pass: each layer's keys and values, the valid positions.
    prefix: Prefix
    #: The first denoising step's velocity [1, 50, 75].
    velocity_step_0: torch.Tensor
    #: The denoised chunk [1, 50, 75].
    actions: torch.Tensor


class LingBotReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = VisionTower()
        self.backbone = LanguageModel()
        self.expert = ExpertModel()
        self.head = LingBotHead()

    def parts(self) -> dict[str, nn.Module]:
        return {"vision": self.vision, "backbone": self.backbone, "expert": self.expert,
                "head": self.head}

    def prefix(self, image_tokens: torch.Tensor, *, image_masks: torch.Tensor,
               language_tokens: torch.Tensor, language_masks: torch.Tensor, depth: int) -> Prefix:
        """`embed_prefix` and the backbone's pass: the visual tokens
        [views, 64, 2048] of the views `image_masks` [views] marks valid, then
        the prompt [1, 72] and its mask, through the first `depth` layers."""
        embeddings = torch.cat((image_tokens.reshape(1, -1, BACKBONE_DIM),
                                self.backbone.embed_tokens(language_tokens)), dim=1)
        pad_masks = torch.cat((image_masks[None].repeat_interleave(image_tokens.shape[1], dim=1),
                               language_masks), dim=1)
        # `make_att_2d_masks` without causal blocks: every valid token sees every valid token.
        mask = pad_masks[:, None, :] & pad_masks[:, :, None]
        positions = pad_masks.cumsum(dim=1) - 1
        hidden = embeddings
        cache: list[KeyValue] = []
        for layer in self.backbone.layers[:depth]:
            query, key, value = layer.project(hidden)
            query, key = rotate(query, positions), rotate(key, positions)
            cache.append((key, value))
            hidden = layer.finish(hidden, attend(query, key, value, mask))
        return Prefix(cache=tuple(cache), pad_masks=pad_masks)

    def velocity(self, prefix: Prefix, noisy: torch.Tensor, *, state: torch.Tensor,
                 time: torch.Tensor) -> torch.Tensor:
        """One denoising step (`embed_suffix`, `predict_velocity`): the
        velocity [1, 50, 75] of the noisy chunk at `time` [1], through as many
        expert layers as the prefix has."""
        head = self.head
        chunk = noisy.shape[1]
        cond = sinusoid(time, EXPERT_DIM).to(state.dtype)
        action_time_input = torch.cat((head.action_in_proj(noisy), cond[:, None].expand(-1, chunk, -1)),
                                      dim=-1)
        actions = head.action_time_mlp_out(F.silu(head.action_time_mlp_in(action_time_input)))
        hidden = torch.cat((head.state_proj(state)[:, None], actions), dim=1)
        # The state token opens one block and the chunk another: each token
        # sees the valid prefix, its own block and the blocks before it.
        suffix_block = (torch.arange(1 + chunk, device=noisy.device) < 2).cumsum(0)
        suffix_mask = (suffix_block[None, :] <= suffix_block[:, None])[None]
        mask = torch.cat((prefix.pad_masks[:, None, :].expand(-1, 1 + chunk, -1), suffix_mask), dim=2)
        positions = (prefix.pad_masks.sum(-1)[:, None]
                     + torch.arange(1, 2 + chunk, device=noisy.device)[None] - 1)
        for layer, (prefix_key, prefix_value) in zip(self.expert.layers, prefix.cache):
            query, key, value = layer.project(hidden, cond)
            query, key = rotate(query, positions), rotate(key, positions)
            attended = attend(query, torch.cat((prefix_key, key), dim=1),
                              torch.cat((prefix_value, value), dim=1), mask)
            hidden = layer.finish(hidden, attended, cond)
        return head.action_out_proj(self.expert.norm(hidden)[:, -chunk:])

    def denoise(self, prefix: Prefix, noise: torch.Tensor, *, state: torch.Tensor,
                steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`sample_actions`' Euler loop: the chunk [1, 50, 75] after `steps`
        steps from `noise`, and the first step's velocity. Upstream walks
        t = 1, 1 + dt, ... while t >= -dt / 2 in the activation dtype, which is
        exactly `steps` iterations for 1 to 10 steps."""
        dtype = self.head.state_proj.weight.dtype
        state, noisy = state.to(dtype), noise.to(dtype)
        dt = torch.full((), -1.0 / steps, dtype=dtype, device=noise.device)
        time = torch.ones((), dtype=dtype, device=noise.device)
        velocities = []
        for _ in range(steps):
            velocity = self.velocity(prefix, noisy, state=state, time=time.expand(1))
            velocities.append(velocity)
            noisy = noisy + dt * velocity
            time = time + dt
        return noisy, velocities[0]

    def forward(self, pixel_values: torch.Tensor, *, grid: Sequence[tuple[int, int, int]],
                image_masks: torch.Tensor, language_tokens: torch.Tensor,
                language_masks: torch.Tensor, state: torch.Tensor, noise: torch.Tensor,
                steps: int = 10, depth: int = LAYERS) -> LingBotOutputs:
        """`sample_actions` for one observation; `grid` is every view's patch
        grid (frames, rows, columns), which `embed_image` derives from the
        patches as one frame of a square grid. `depth` runs the first `depth`
        layers of both towers and skips the rest, for bisection."""
        vision = self.vision(pixel_values, tables=self.vision.tables(grid))
        image_tokens = vision.view(pixel_values.shape[0], -1, BACKBONE_DIM)
        prefix = self.prefix(image_tokens, image_masks=image_masks, language_tokens=language_tokens,
                             language_masks=language_masks, depth=depth)
        actions, velocity = self.denoise(prefix, noise, state=state, steps=steps)
        return LingBotOutputs(vision_embeddings=image_tokens, prefix=prefix, velocity_step_0=velocity,
                              actions=actions)


def make_reference(*, device: str | torch.device = "meta") -> LingBotReference:
    """The model without weights; on `meta` it allocates none (only the small
    analytic RoPE buffer, on the host), and its `official_schema` is the
    official checkpoint's."""
    with torch.device(device):
        return LingBotReference().eval().requires_grad_(False)


def bind_parts(reference: LingBotReference, parts: Sequence[str], weights: Mapping[str, torch.Tensor], *,
               precision: Literal["bfloat16", "float32"]) -> None:
    """Bind `parts` (`PREFIXES`) to their official tensors -- without a copy
    where they are already in `precision`'s dtype -- and put their analytic
    buffers beside them, in that dtype too: the deployment casts the whole
    model to bfloat16."""
    modules = {part: reference.parts()[part] for part in parts}
    bind(modules, prefixes=PREFIXES, weights=weights)
    for module in modules.values():
        module.to(device=next(module.parameters()).device, dtype=PRECISION_DTYPES[precision])


def load(weights: Mapping[str, torch.Tensor], *,
         precision: Literal["bfloat16", "float32"] = "bfloat16") -> LingBotReference:
    """The model bound to official tensors, in upstream's inference dtype
    (`bfloat16`) or entirely in float32."""
    reference = make_reference()
    bind_parts(reference, tuple(PREFIXES), weights, precision=precision)
    return reference


def random_weights(seed: int, *, device: str | torch.device = "cuda",
                   scale: float = 0.05) -> dict[str, torch.Tensor]:
    """Seeded official-layout weights (`official.random_official`); the small
    scale keeps deep random residual streams numerically useful."""
    return random_official(make_reference().parts(), prefixes=PREFIXES, seed=seed, device=device,
                           scale=scale)


__all__ = ["KeyValue", "LingBotOutputs", "LingBotReference", "PREFIXES", "Prefix", "VisionTables",
           "bind_parts", "load", "make_reference", "random_weights"]
