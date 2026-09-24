# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 end to end, in plain torch, on the official checkpoint.

Translated from NVIDIA/Isaac-GR00T (github.com/NVIDIA/Isaac-GR00T, revision
51d4c89): `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`, `modules/qwen3_backbone.py`,
`modules/dit.py` and `modules/embodiment_conditioned_mlp.py`, and from the
library modules those build on: Transformers 4.57.3's Qwen3-VL
(`models/qwen3_vl/modeling_qwen3_vl.py`, with the SDPA attention of
`integrations/sdpa_attention.py`) and Diffusers 0.35.1's `Attention` under
`AttnProcessor2_0`, `FeedForward`, `Timesteps` and `TimestepEmbedding`. Only
inference is translated: no dropout, no training noise, no real-time-chunking
inpainting.

The three stages:

    vision encoder  Qwen3-VL's vision tower over every view's patches: the
                    patch embedding plus bilinearly interpolated learned
                    positions, 24 blocks with 2D RoPE attending within each
                    image, and 2x2 patch mergers -- the final one's visual
                    tokens [128, 2048], and one DeepStack feature from each of
                    blocks 5, 11 and 17
    LLM backbone    the token embeddings with the visual tokens written in,
                    then 16 Qwen3 layers (interleaved M-RoPE, causal over the
                    valid tokens), the DeepStack features added to the visual
                    tokens after layers 0-2; the hidden state before the final
                    RMSNorm (GR00T drops the layers after `select_layer` and
                    reads `hidden_states[-1]`)
    action expert   a LayerNorm and four self-attention blocks refine the
                    backbone features, the embodiment's MLP encodes the state,
                    and `steps` Euler steps of the 32-block DiT run from
                    `noise`: its even blocks cross-attend to the text tokens and
                    to the image tokens in turn, its odd blocks to themselves;
                    the embodiment's decoder reads the velocity

The inputs are already prepared, as the model receives them after GR00T's
processor: the patches [512, 1536] and each image's patch grid, the token ids
and mask, the M-RoPE position ids [3, 1, tokens], the visual tokens'
positions, the state [1, 1, 132], the embodiment id and the noise
[1, 40, 132]. Weights are the official checkpoint's tensors by their official
names (`PREFIXES`); the language model head is not read.

Upstream's inference numerics, which this file reproduces:

- GR00T's policy casts the whole model to bfloat16 (`gr00t_policy.py`,
  `model.to(dtype=torch.bfloat16)`), buffers included: RoPE's inverse
  frequencies, which `Qwen3Backbone` rebuilds in float32 at load, are rounded
  to bfloat16. The vision rotary table is then built in bfloat16; the text
  one in float32 from the rounded frequencies, its cos and sin cast to
  bfloat16;
- the patch embedding's Conv3d reads a `channels_last_3d` input
  (`Qwen3Backbone._apply_vision_patch_embed_channels_last`);
- vision attention is SDPA per frame, its RoPE applied in float32; Qwen3's
  RMSNorm normalizes in float32 and scales in the activation dtype; text
  attention is SDPA over the grouped keys and values repeated to every head,
  under a boolean causal-and-padding mask;
- the vision MLP and the DiT and refiner FFNs use the tanh GELU, the patch
  mergers the exact one;
- DiT attention is SDPA with the boolean key mask repeated per head; both
  timestep embeddings are built in float32 and cast to the weights' dtype.

Deliberate differences from upstream, none of them in the model's math:

- the attention is PyTorch SDPA. The checkpoint asks for FlashAttention-2
  (`use_flash_attention`), which `Qwen3Backbone` takes whenever flash-attn is
  installed; this is the SDPA path it falls back to, which the official
  parity oracle forces (`eval/groot_n17/parity.py`). The DiT's math-only SDPA
  on sm_121 (`dit.py`, `_sdpa_context`) is not reproduced either;
- the positions come prepared: the M-RoPE position ids and the visual tokens'
  positions are inputs, where upstream derives them inside
  `Qwen3VLModel.forward` (`get_rope_index`, boolean image-token masks); the
  DeepStack features join through a fixed-size gather and scatter;
- the text attention always takes the boolean mask. Upstream drops a mask
  with no padding and lets SDPA apply causality over the grouped keys
  directly (`is_causal`, `enable_gqa`): the same math, not the same bits;
- upstream also converts the Conv3d's weight to `channels_last_3d`; here it
  keeps the checkpoint's layout, so cuDNN may choose another algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..official import PRECISION_DTYPES, bind
from .spec import (
    ACTION_DIM,
    BACKBONE_DIM,
    BACKBONE_FFN,
    DEEPSTACK_BLOCKS,
    DIT_BLOCKS,
    DIT_DIM,
    HEAD_HIDDEN,
    KV_DIM,
    LAYERS,
    STATE_DIM,
    STEPS,
    TIMESTEP_CHANNELS,
    VISION_BLOCKS,
    VISION_DIM,
    VISION_FFN,
)

#: Official prefix of each part: the Qwen3-VL vision tower and language model
#: inside GR00T's backbone, and the action head.
PREFIXES = {
    "vision": "backbone.model.model.visual.",
    "backbone": "backbone.model.model.language_model.",
    "action": "action_head.",
}

VISION_HEADS = 16
VISION_HEAD_DIM = VISION_DIM // VISION_HEADS
#: Patch side and frames per temporal patch of the Conv3d embedding.
PATCH = 16
TEMPORAL_PATCH = 2
#: The mergers' neighbourhood side: 2x2 patches become one visual token.
MERGE = 2
#: Side of the learned position grid (`num_position_embeddings` = 48 * 48).
POSITION_GRID = 48
VISION_ROPE_THETA = 10000.0
VISION_NORM_EPS = 1e-6

VOCABULARY = 151936
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 5_000_000
#: M-RoPE frequency channels of the temporal, height and width positions.
MROPE_SECTION = (24, 20, 20)
RMS_EPS = 1e-6
#: Qwen3-VL's image-pad token, the placeholder of every visual token.
IMAGE_TOKEN = 151655

#: Embodiments with their own state, action encoder and decoder weights.
EMBODIMENTS = 32
#: Heads of every DiT and refiner attention (head width 48 and 64).
ATTENTION_HEADS = 32
REFINER_BLOCKS = 4
#: The DiT's cross-attending blocks take the text tokens every this many of
#: them and the image tokens between (`attend_text_every_n_blocks`).
ATTEND_TEXT_EVERY = 2
#: Flow-matching time enters the model discretized into this many buckets.
TIMESTEP_BUCKETS = 1000
#: Learned positions of the action tokens (`max_seq_len`).
ACTION_POSITIONS = 1024
LAYER_NORM_EPS = 1e-5
#: The DiT's output LayerNorm's epsilon (`norm_out`).
OUTPUT_NORM_EPS = 1e-6


def rotate_half(states: torch.Tensor) -> torch.Tensor:
    """RoPE's rotation in half-split channel order: (-second half, first half)."""
    first, second = states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@dataclass(frozen=True)
class VisionTables:
    """What the vision tower's forward derives from the images' patch grids
    (`VisionModel.tables`), in the mergers' patch order."""
    #: Interpolated learned positions [patches, 1024].
    position: torch.Tensor
    #: The 2D rotary table's cos and sin [patches, 64].
    cos: torch.Tensor
    sin: torch.Tensor
    #: Patches of each frame, attended separately.
    lengths: tuple[int, ...]


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = (TEMPORAL_PATCH, PATCH, PATCH)
        self.proj = nn.Conv3d(3, VISION_DIM, kernel_size=kernel, stride=kernel)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Flattened patches [patches, 3 * 2 * 16 * 16] -> embeddings [patches, 1024]."""
        volumes = patches.view(-1, 3, TEMPORAL_PATCH, PATCH, PATCH).to(self.proj.weight.dtype)
        return self.proj(volumes.contiguous(memory_format=torch.channels_last_3d)).view(-1, VISION_DIM)


class VisionAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(VISION_DIM, 3 * VISION_DIM)
        self.proj = nn.Linear(VISION_DIM, VISION_DIM)

    def forward(self, hidden: torch.Tensor, *, tables: VisionTables) -> torch.Tensor:
        tokens = hidden.shape[0]
        query, key, value = (self.qkv(hidden).reshape(tokens, 3, VISION_HEADS, -1)
                             .permute(1, 0, 2, 3).unbind(0))
        # `apply_rotary_pos_emb_vision`: rotated in float32, cast back.
        cos, sin = tables.cos.unsqueeze(-2).float(), tables.sin.unsqueeze(-2).float()
        query, key = ((states.float() * cos + rotate_half(states.float()) * sin).to(states.dtype)
                      for states in (query, key))
        query, key, value = (states.transpose(0, 1)[None] for states in (query, key, value))
        # Transformers' SDPA path attends within each frame, one call per
        # frame; it passes `enable_gqa` although every head has its own keys.
        attended = [F.scaled_dot_product_attention(frame_query, frame_key, frame_value, dropout_p=0.0,
                                                   scale=VISION_HEAD_DIM**-0.5, is_causal=False,
                                                   enable_gqa=True).transpose(1, 2)
                    for frame_query, frame_key, frame_value in zip(
                        *(states.split(tables.lengths, dim=2) for states in (query, key, value)))]
        return self.proj(torch.cat(attended, dim=1).reshape(tokens, -1))


class VisionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(VISION_DIM, VISION_FFN)
        self.linear_fc2 = nn.Linear(VISION_FFN, VISION_DIM)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden), approximate="tanh"))


class VisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(VISION_DIM, eps=VISION_NORM_EPS)
        self.norm2 = nn.LayerNorm(VISION_DIM, eps=VISION_NORM_EPS)
        self.attn = VisionAttention()
        self.mlp = VisionMLP()

    def forward(self, hidden: torch.Tensor, *, tables: VisionTables) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), tables=tables)
        return hidden + self.mlp(self.norm2(hidden))


class PatchMerger(nn.Module):
    """Each 2x2 neighbourhood (consecutive patches in the tower's order) as
    one 4096-wide vector, through an exact-GELU MLP to the backbone width.
    The final merger normalizes every patch before the concatenation, the
    DeepStack mergers the concatenation (`use_postshuffle_norm`)."""

    def __init__(self, *, postshuffle_norm: bool) -> None:
        super().__init__()
        merged = VISION_DIM * MERGE**2
        self.postshuffle_norm = postshuffle_norm
        self.norm = nn.LayerNorm(merged if postshuffle_norm else VISION_DIM, eps=VISION_NORM_EPS)
        self.linear_fc1 = nn.Linear(merged, merged)
        self.linear_fc2 = nn.Linear(merged, BACKBONE_DIM)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """[patches, 1024] -> [patches / 4, 2048]."""
        merged = VISION_DIM * MERGE**2
        normalized = (self.norm(hidden.view(-1, merged)) if self.postshuffle_norm
                      else self.norm(hidden).view(-1, merged))
        return self.linear_fc2(F.gelu(self.linear_fc1(normalized)))


class VisionModel(nn.Module):
    """`Qwen3VLVisionModel`, its forward split into the grid's tables (`tables`)
    and the tower that consumes them (`forward`)."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Embedding(POSITION_GRID**2, VISION_DIM)
        self.blocks = nn.ModuleList(VisionBlock() for _ in range(VISION_BLOCKS))
        self.merger = PatchMerger(postshuffle_norm=False)
        self.deepstack_merger_list = nn.ModuleList(PatchMerger(postshuffle_norm=True)
                                                   for _ in DEEPSTACK_BLOCKS)
        # Analytic, so not in the checkpoint: built here on the host, in
        # float32, and cast with the model (`bind_part`).
        rotary_dim = VISION_HEAD_DIM // 2
        exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")
        self.register_buffer("inverse_frequency", 1.0 / (VISION_ROPE_THETA ** (exponents / rotary_dim)),
                             persistent=False)

    def tables(self, grid: Sequence[tuple[int, int, int]]) -> VisionTables:
        """`fast_pos_embed_interpolate` and `rot_pos_emb` for images of patch
        grids `grid` (frames, rows, columns each). Upstream builds them inside
        its forward from host-side index lists, as here, which a CUDA graph
        cannot capture."""
        weight = self.pos_embed.weight
        side = POSITION_GRID
        positions: list[torch.Tensor] = []
        coordinates: list[torch.Tensor] = []
        for frames, rows, columns in grid:
            # Bilinear interpolation of the learned 48x48 grid at the image's
            # rows and columns: four corners, each embedding times its weight
            # in the table's dtype, summed in upstream's order.
            row_at = torch.linspace(0, side - 1, rows, device="cpu")
            column_at = torch.linspace(0, side - 1, columns, device="cpu")
            row_low, column_low = row_at.int(), column_at.int()
            row_high = (row_low + 1).clip(max=side - 1)
            column_high = (column_low + 1).clip(max=side - 1)
            row_fraction, column_fraction = row_at - row_low, column_at - column_low
            corners = (
                (row_low, column_low, (1 - row_fraction)[:, None] * (1 - column_fraction)[None]),
                (row_low, column_high, (1 - row_fraction)[:, None] * column_fraction[None]),
                (row_high, column_low, row_fraction[:, None] * (1 - column_fraction)[None]),
                (row_high, column_high, row_fraction[:, None] * column_fraction[None]),
            )
            weighted = [self.pos_embed((row[:, None] * side + column[None]).flatten().long()
                                       .to(weight.device))
                        * blend.flatten().to(device=weight.device, dtype=weight.dtype)[:, None]
                        for row, column, blend in corners]
            interpolated = (weighted[0] + weighted[1] + weighted[2] + weighted[3]).repeat(frames, 1)
            merged_rows, merged_columns = rows // MERGE, columns // MERGE
            positions.append(interpolated.view(frames, merged_rows, MERGE, merged_columns, MERGE, -1)
                             .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
            # Every patch's (row, column), in the same merged order.
            offsets = torch.arange(MERGE)
            neighbourhood = (merged_rows, merged_columns, MERGE, MERGE)
            row_index = (torch.arange(merged_rows)[:, None, None, None] * MERGE
                         + offsets[None, None, :, None]).expand(neighbourhood).reshape(-1)
            column_index = (torch.arange(merged_columns)[None, :, None, None] * MERGE
                            + offsets[None, None, None, :]).expand(neighbourhood).reshape(-1)
            coordinates.append(torch.stack((row_index, column_index), dim=-1).repeat(frames, 1))
        # `Qwen3VLVisionRotaryEmbedding` over the longest side, in the
        # frequencies' dtype; each patch takes its row's and its column's angles.
        frequency = self.inverse_frequency
        longest = max(max(rows, columns) for _, rows, columns in grid)
        side_positions = torch.arange(longest, device=frequency.device, dtype=frequency.dtype)
        table = torch.outer(side_positions, frequency)
        angles = table[torch.cat(coordinates).to(frequency.device)].flatten(1)
        doubled = torch.cat((angles, angles), dim=-1)
        return VisionTables(position=torch.cat(positions), cos=doubled.cos(), sin=doubled.sin(),
                            lengths=tuple(rows * columns for frames, rows, columns in grid
                                          for _ in range(frames)))

    def forward(self, pixel_values: torch.Tensor, *,
                tables: VisionTables) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The final merger's visual tokens [patches / 4, 2048] and the
        DeepStack features, one per `DEEPSTACK_BLOCKS` entry."""
        hidden = self.patch_embed(pixel_values) + tables.position
        deepstack = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, tables=tables)
            if index not in DEEPSTACK_BLOCKS:
                continue
            deepstack.append(self.deepstack_merger_list[DEEPSTACK_BLOCKS.index(index)](hidden))
        return self.merger(hidden), deepstack


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Normalized in float32, scaled in the input's dtype."""
        widened = hidden.float()
        normalized = widened * torch.rsqrt(widened.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return self.weight * normalized.to(hidden.dtype)


class TextAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(BACKBONE_DIM, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(BACKBONE_DIM, KV_DIM, bias=False)
        self.v_proj = nn.Linear(BACKBONE_DIM, KV_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, BACKBONE_DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(self, hidden: torch.Tensor, *, rotary: tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor) -> torch.Tensor:
        batch, tokens = hidden.shape[:2]
        query = self.q_norm(self.q_proj(hidden).view(batch, tokens, -1, HEAD_DIM)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden).view(batch, tokens, -1, HEAD_DIM)).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, tokens, -1, HEAD_DIM).transpose(1, 2)
        cos, sin = (table[:, None] for table in rotary)
        query, key = (states * cos + rotate_half(states) * sin for states in (query, key))
        # Under a mask Transformers' SDPA repeats each key and value head to
        # its query heads (`repeat_kv`).
        key, value = (states.repeat_interleave(HEADS // KV_HEADS, dim=1) for states in (key, value))
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0,
                                                  scale=HEAD_DIM**-0.5, is_causal=False)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, tokens, -1))


class TextMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(BACKBONE_DIM, BACKBONE_FFN, bias=False)
        self.up_proj = nn.Linear(BACKBONE_DIM, BACKBONE_FFN, bias=False)
        self.down_proj = nn.Linear(BACKBONE_FFN, BACKBONE_DIM, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TextLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TextAttention()
        self.mlp = TextMLP()
        self.input_layernorm = RMSNorm(BACKBONE_DIM)
        self.post_attention_layernorm = RMSNorm(BACKBONE_DIM)

    def forward(self, hidden: torch.Tensor, *, rotary: tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), rotary=rotary, mask=mask)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TextModel(nn.Module):
    """`Qwen3VLTextModel` with GR00T's `select_layer` layers. Its final `norm`
    is in the checkpoint and held here, but GR00T reads the hidden state
    before it."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCABULARY, BACKBONE_DIM)
        self.layers = nn.ModuleList(TextLayer() for _ in range(LAYERS))
        self.norm = RMSNorm(BACKBONE_DIM)
        exponents = torch.arange(0, HEAD_DIM, 2, dtype=torch.int64, device="cpu").float()
        self.register_buffer("inverse_frequency", 1.0 / (ROPE_THETA ** (exponents / HEAD_DIM)),
                             persistent=False)

    def forward(self, input_ids: torch.Tensor, *, attention_mask: torch.Tensor,
                position_ids: torch.Tensor, image_indices: torch.Tensor, vision: torch.Tensor,
                deepstack: Sequence[torch.Tensor]) -> torch.Tensor:
        """Token ids and mask [1, tokens], M-RoPE position ids [3, 1, tokens],
        the visual tokens' positions [128], the vision encoder's tokens
        [128, 2048] and DeepStack features -> the last layer's hidden state
        [1, tokens, 2048]."""
        hidden = self.embed_tokens(input_ids).index_copy(1, image_indices, vision[None])
        tokens = input_ids.shape[1]
        causal = torch.ones((tokens, tokens), dtype=torch.bool, device=hidden.device).tril()
        mask = causal[None, None] & attention_mask[:, None, None, :].bool()
        # Interleaved M-RoPE (`Qwen3VLTextRotaryEmbedding`): float32 angles
        # from the frequencies in the model's dtype; within the first
        # 3 * section[axis] channels, channel c takes axis c % 3's position.
        angles = position_ids[..., None].float() * self.inverse_frequency.float()
        mixed = angles[0]
        for axis in (1, 2):
            channels = slice(axis, 3 * MROPE_SECTION[axis], 3)
            mixed[..., channels] = angles[axis, ..., channels]
        doubled = torch.cat((mixed, mixed), dim=-1)
        rotary = (doubled.cos().to(hidden.dtype), doubled.sin().to(hidden.dtype))
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, rotary=rotary, mask=mask)
            if index >= len(deepstack):
                continue
            # DeepStack (`_deepstack_process`): the feature joins the visual tokens.
            hidden = hidden.index_copy(1, image_indices,
                                       hidden.index_select(1, image_indices) + deepstack[index][None])
        return hidden


class CategoryLinear(nn.Module):
    """A linear layer per embodiment (`CategorySpecificLinear`)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(EMBODIMENTS, in_dim, out_dim))
        self.b = nn.Parameter(torch.empty(EMBODIMENTS, out_dim))

    def forward(self, hidden: torch.Tensor, embodiment: torch.Tensor) -> torch.Tensor:
        """[batch, rows, in_dim] and embodiment ids [batch] -> [batch, rows, out_dim]."""
        return torch.bmm(hidden, self.W[embodiment]) + self.b[embodiment].unsqueeze(1)


class CategoryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer1 = CategoryLinear(in_dim, hidden_dim)
        self.layer2 = CategoryLinear(hidden_dim, out_dim)

    def forward(self, hidden: torch.Tensor, embodiment: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(hidden, embodiment)), embodiment)


class ActionEncoder(nn.Module):
    """`MultiEmbodimentActionEncoder`: the noisy chunk, then the step's
    sinusoid beside it, through the embodiment's three projections."""

    def __init__(self) -> None:
        super().__init__()
        self.W1 = CategoryLinear(ACTION_DIM, DIT_DIM)
        self.W2 = CategoryLinear(2 * DIT_DIM, DIT_DIM)
        self.W3 = CategoryLinear(DIT_DIM, DIT_DIM)

    def forward(self, actions: torch.Tensor, timestep: torch.Tensor,
                embodiment: torch.Tensor) -> torch.Tensor:
        """The chunk [batch, 40, 132] at bucketed time `timestep` [batch] -> [batch, 40, 1536]."""
        encoded = self.W1(actions, embodiment)
        # `SinusoidalPositionalEncoding`: sin, then cos, of the timestep at 768
        # log-spaced frequencies, in float32.
        half = DIT_DIM // 2
        exponent = -torch.arange(half, dtype=torch.float32, device=actions.device) * (
            torch.log(torch.tensor(10000.0, device="cpu")) / half)
        phase = timestep[:, None].expand(-1, actions.shape[1]).float()[..., None] * exponent.exp()
        time = torch.cat((phase.sin(), phase.cos()), dim=-1).to(encoded.dtype)
        hidden = self.W2(torch.cat((encoded, time), dim=-1), embodiment)
        return self.W3(hidden * torch.sigmoid(hidden), embodiment)


class TimestepEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(TIMESTEP_CHANNELS, DIT_DIM)
        self.linear_2 = nn.Linear(DIT_DIM, DIT_DIM)


class TimestepEncoder(nn.Module):
    """Diffusers' `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1)`,
    then `TimestepEmbedding` (linear, SiLU, linear)."""

    def __init__(self) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding()

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Bucketed time [batch] -> the DiT's conditioning [batch, 1536]."""
        embedder = self.timestep_embedder
        half = TIMESTEP_CHANNELS // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32,
                                                   device=timestep.device) / (half - 1)
        angles = timestep[:, None].float() * exponent.exp()[None]
        features = torch.cat((angles.cos(), angles.sin()), dim=-1).to(embedder.linear_1.weight.dtype)
        return embedder.linear_2(F.silu(embedder.linear_1(features)))


class AdaLayerNorm(nn.Module):
    """A LayerNorm without affine, then scaled and shifted by the conditioning."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(DIT_DIM, 2 * DIT_DIM)
        self.norm = nn.LayerNorm(DIT_DIM, eps=LAYER_NORM_EPS, elementwise_affine=False)

    def forward(self, hidden: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(time)).chunk(2, dim=1)
        return self.norm(hidden) * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    """Diffusers' `Attention` under `AttnProcessor2_0` as GR00T configures it:
    32 heads, biased projections, no norms. Its output dropout (`to_out.1`)
    is inference's identity."""

    def __init__(self, dim: int, *, context_dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, hidden: torch.Tensor, *, context: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        """Queries from `hidden` [batch, queries, dim], keys and values from
        `context` [batch, keys, context_dim]; `mask` [batch, keys] marks the
        keys every query may attend to, None all of them."""
        batch = hidden.shape[0]
        head_dim = hidden.shape[-1] // ATTENTION_HEADS
        projected = (self.to_q(hidden), self.to_k(context), self.to_v(context))
        query, key, value = (states.view(batch, -1, ATTENTION_HEADS, head_dim).transpose(1, 2)
                             for states in projected)
        # `prepare_attention_mask`: the key mask, once per head.
        head_mask = (None if mask is None else
                     mask.repeat_interleave(ATTENTION_HEADS, dim=0).view(batch, ATTENTION_HEADS, 1, -1))
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=head_mask,
                                                  dropout_p=0.0, is_causal=False)
        return self.to_out[0](attended.transpose(1, 2).reshape(batch, -1, ATTENTION_HEADS * head_dim)
                              .to(query.dtype))


class GeluProjection(nn.Module):
    """Diffusers' `GELU` module: a projection, then the tanh GELU."""

    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden), approximate="tanh")


class FeedForward(nn.Module):
    """Diffusers' `FeedForward(activation_fn="gelu-approximate")`, four times
    wide. Its dropouts (`net.1`, `net.3`) are inference's identity, so `net`
    holds only the projection in (`net.0`) and out (`net.2`)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleDict({"0": GeluProjection(dim, 4 * dim), "2": nn.Linear(4 * dim, dim)})

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net["2"](self.net["0"](hidden))


class RefinerBlock(nn.Module):
    """A `BasicTransformerBlock` of the refiner: LayerNorm, self-attention,
    LayerNorm, FFN, both residual."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(BACKBONE_DIM, eps=LAYER_NORM_EPS)
        self.attn1 = Attention(BACKBONE_DIM, context_dim=BACKBONE_DIM)
        self.norm3 = nn.LayerNorm(BACKBONE_DIM, eps=LAYER_NORM_EPS)
        self.ff = FeedForward(BACKBONE_DIM)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        hidden = self.attn1(normalized, context=normalized, mask=None) + hidden
        return self.ff(self.norm3(hidden)) + hidden


class DiTBlock(nn.Module):
    """A `BasicTransformerBlock` of the DiT (`norm_type="ada_norm"`): the
    conditioned LayerNorm, attention to the backbone features (`context_dim`
    2048) or to itself, then a LayerNorm and the FFN, both residual."""

    def __init__(self, *, context_dim: int) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm()
        self.attn1 = Attention(DIT_DIM, context_dim=context_dim)
        self.norm3 = nn.LayerNorm(DIT_DIM, eps=LAYER_NORM_EPS, elementwise_affine=False)
        self.ff = FeedForward(DIT_DIM)

    def forward(self, hidden: torch.Tensor, *, time: torch.Tensor, context: torch.Tensor | None,
                mask: torch.Tensor | None) -> torch.Tensor:
        """`context` the backbone features to cross-attend to, None to self-attend."""
        normalized = self.norm1(hidden, time)
        hidden = self.attn1(normalized, context=normalized if context is None else context,
                            mask=mask) + hidden
        return self.ff(self.norm3(hidden)) + hidden


class VisionLanguageRefiner(nn.Module):
    """`SelfAttentionTransformer`: self-attention blocks over the backbone features."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(RefinerBlock() for _ in range(REFINER_BLOCKS))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden = block(hidden)
        return hidden


class AlternateVLDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.timestep_encoder = TimestepEncoder()
        self.transformer_blocks = nn.ModuleList(
            DiTBlock(context_dim=BACKBONE_DIM if index % 2 == 0 else DIT_DIM)
            for index in range(DIT_BLOCKS))
        self.norm_out = nn.LayerNorm(DIT_DIM, eps=OUTPUT_NORM_EPS, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(DIT_DIM, 2 * DIT_DIM)
        self.proj_out_2 = nn.Linear(DIT_DIM, HEAD_HIDDEN)

    def forward(self, hidden: torch.Tensor, *, context: torch.Tensor, timestep: torch.Tensor,
                image_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """The state and action tokens [batch, 41, 1536] -> [batch, 41, 1024]."""
        time = self.timestep_encoder(timestep)
        image = image_mask & attention_mask
        text = ~image_mask & attention_mask
        for index, block in enumerate(self.transformer_blocks):
            if index % 2:
                hidden = block(hidden, time=time, context=None, mask=None)
            else:
                hidden = block(hidden, time=time, context=context,
                               mask=text if index % (2 * ATTEND_TEXT_EVERY) == 0 else image)
        shift, scale = self.proj_out_1(F.silu(time)).chunk(2, dim=1)
        return self.proj_out_2(self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None])


class ActionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = AlternateVLDiT()
        self.state_encoder = CategoryMLP(STATE_DIM, HEAD_HIDDEN, DIT_DIM)
        self.action_encoder = ActionEncoder()
        self.action_decoder = CategoryMLP(HEAD_HIDDEN, HEAD_HIDDEN, ACTION_DIM)
        self.vlln = nn.LayerNorm(BACKBONE_DIM)
        self.vl_self_attention = VisionLanguageRefiner()
        self.position_embedding = nn.Embedding(ACTION_POSITIONS, DIT_DIM)

    def forward(self, backbone_features: torch.Tensor, *, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, state: torch.Tensor, embodiment_id: torch.Tensor,
                noise: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`get_action`: the chunk [1, 40, 132] after `steps` Euler steps from
        `noise`, and the first step's velocity. The real-time-chunking velocity
        strength, 1 without a previous chunk, is left out."""
        # `Qwen3Backbone.forward`'s masks: the image tokens, and the valid tokens.
        image_mask, valid = input_ids == IMAGE_TOKEN, attention_mask == 1
        dtype = self.vlln.weight.dtype
        context = self.vl_self_attention(self.vlln(backbone_features))
        state_features = self.state_encoder(state.to(dtype).view(state.shape[0], 1, -1), embodiment_id)
        actions = noise.to(dtype)
        chunk = actions.shape[1]
        positions = self.position_embedding(torch.arange(chunk, device=actions.device))[None]
        velocities = []
        for step in range(steps):
            # Time runs 0, 1 / steps, ... and enters the model bucketed.
            timestep = torch.full((actions.shape[0],), int(step / float(steps) * TIMESTEP_BUCKETS),
                                  device=actions.device)
            encoded = self.action_encoder(actions, timestep, embodiment_id) + positions
            hidden = self.model(torch.cat((state_features, encoded), dim=1), context=context,
                                timestep=timestep, image_mask=image_mask, attention_mask=valid)
            velocity = self.action_decoder(hidden, embodiment_id)[:, -chunk:]
            velocities.append(velocity)
            actions = actions + (1.0 / steps) * velocity
        return actions, velocities[0]


@dataclass(frozen=True)
class GrootOutputs:
    """Every stage's output, laid out as the engine's stage outputs are."""
    #: The final merger's visual tokens [128, 2048].
    vision_embeddings: torch.Tensor
    #: The DeepStack features, one [128, 2048] per feeding vision block.
    deepstack: list[torch.Tensor]
    #: The backbone's last hidden state [1, tokens, 2048].
    backbone_features: torch.Tensor
    #: The denoised chunk [1, 40, 132].
    actions: torch.Tensor
    #: The first denoising step's velocity [1, 40, 132].
    velocity_step_0: torch.Tensor


class GrootReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = VisionModel()
        self.backbone = TextModel()
        self.action = ActionHead()

    def parts(self) -> dict[str, nn.Module]:
        return {"vision": self.vision, "backbone": self.backbone, "action": self.action}

    def forward(self, pixel_values: torch.Tensor, *, grid: Sequence[tuple[int, int, int]],
                input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                image_indices: torch.Tensor, state: torch.Tensor, embodiment_id: torch.Tensor,
                noise: torch.Tensor, steps: int = STEPS) -> GrootOutputs:
        """One observation end to end; `grid` is every image's patch grid
        (frames, rows, columns), the processor's `image_grid_thw`."""
        vision_embeddings, deepstack = self.vision(pixel_values, tables=self.vision.tables(grid))
        backbone_features = self.backbone(input_ids, attention_mask=attention_mask,
                                          position_ids=position_ids, image_indices=image_indices,
                                          vision=vision_embeddings, deepstack=deepstack)
        actions, velocity = self.action(backbone_features, input_ids=input_ids,
                                        attention_mask=attention_mask, state=state,
                                        embodiment_id=embodiment_id, noise=noise, steps=steps)
        return GrootOutputs(vision_embeddings=vision_embeddings, deepstack=deepstack,
                            backbone_features=backbone_features, actions=actions,
                            velocity_step_0=velocity)


def make_reference(*, device: str | torch.device = "meta") -> GrootReference:
    """The model without weights; on `meta` it allocates none (only the small
    analytic RoPE buffers, on the host), and its `official_schema` is the
    official checkpoint's."""
    with torch.device(device):
        return GrootReference().eval().requires_grad_(False)


def bind_part(reference: GrootReference, part: str, weights: Mapping[str, torch.Tensor], *,
              precision: Literal["bfloat16", "float32"]) -> None:
    """Bind one part (`PREFIXES`) to its official tensors -- without a copy
    where they are already in `precision`'s dtype -- and put its analytic
    buffers beside them, in that dtype too: GR00T's policy casts the whole
    model to bfloat16."""
    module = reference.parts()[part]
    bind({part: module}, prefixes=PREFIXES, weights=weights)
    module.to(device=next(module.parameters()).device, dtype=PRECISION_DTYPES[precision])


def load(weights: Mapping[str, torch.Tensor], *,
         precision: Literal["bfloat16", "float32"] = "bfloat16") -> GrootReference:
    """The model bound to official tensors, in upstream's inference dtype
    (`bfloat16`) or entirely in float32."""
    reference = make_reference()
    for part in PREFIXES:
        bind_part(reference, part, weights, precision=precision)
    return reference


__all__ = ["GrootOutputs", "GrootReference", "PREFIXES", "VisionTables", "bind_part", "load",
           "make_reference"]
