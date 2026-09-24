# SPDX-FileCopyrightText: Copyright 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0
"""Pi0.5 end to end, in plain torch, on the official OpenPI checkpoint.

Translated from OpenPI (github.com/Physical-Intelligence/openpi, revision
215abfb), `src/openpi/models_pytorch/pi0_pytorch.py`: `PI0Pytorch` with
`pi05=True`, its `embed_prefix`, `embed_suffix`, `sample_actions` and
`denoise_step`. The PaliGemma parts Pi0 shares are
`models.paligemma.reference`, which also states the numerics reproduced.

The three stages:

    vision encoder  SigLIP over every view; `vision_hidden` is its last hidden
                    state before the final LayerNorm
    LLM backbone    the image tokens (final LayerNorm, projector) and the prompt
                    through Gemma-2B, bidirectionally among valid tokens;
                    every layer's keys and values after RoPE
    action expert   `steps` Euler steps of the Gemma-300M expert from `noise`
                    at t = 1 down to t = 0; each step embeds the noisy chunk,
                    conditions every RMSNorm on the timestep through the time
                    MLP (adaptive RMSNorm), attends over the valid prefix and
                    the whole chunk, and projects the velocity out

The inputs are already prepared, as the model receives them: images
[views, 224, 224, 3] in [-1, 1], a per-view validity mask, the tokenized
prompt -- Pi0.5 writes the discretized state into it (`tokenize.Pi05Tokenizer`)
-- with its mask, and the noise [chunk, 32]. Weights are the official
checkpoint's tensors by their official names (`PREFIXES`); the language
model head and the expert's unused head are not read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..official import Precision, random_official
from ..paligemma.reference import (
    GEMMA_2B,
    PREFIXES as PALIGEMMA_PREFIXES,
    KeyValue,
    PaliGemmaWithExpert,
    Prefix,
    bind_upstream,
    sinusoidal,
)
from .spec import ACTION_DIM, DECODER_DIM, DEFAULT_FLOW_STEPS

#: Official prefix of every part: PaliGemma-with-expert's, and the heads at the top level.
PREFIXES = {**PALIGEMMA_PREFIXES, "head": ""}


class Pi05Head(nn.Module):
    """The projections outside PaliGemma-with-expert: the chunk in and the
    velocity out, and the time MLP whose output conditions the expert."""

    def __init__(self) -> None:
        super().__init__()
        self.action_in_proj = nn.Linear(ACTION_DIM, DECODER_DIM)
        self.action_out_proj = nn.Linear(DECODER_DIM, ACTION_DIM)
        self.time_mlp_in = nn.Linear(DECODER_DIM, DECODER_DIM)
        self.time_mlp_out = nn.Linear(DECODER_DIM, DECODER_DIM)


@dataclass(frozen=True)
class Pi05Outputs:
    """Every stage's output, in clean shapes (no padding, OpenPI's layouts)."""
    #: The vision tower's last hidden state before its final LayerNorm [views, 256, 1152].
    vision_hidden: torch.Tensor
    #: The backbone's pass: each layer's keys and values, and the valid prefix positions.
    prefix: Prefix
    #: The denoised chunk [chunk, 32], float32.
    actions: torch.Tensor
    #: The expert's own keys and values at the last step, per layer.
    suffix_cache: list[KeyValue]


class Pi05Reference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma = PaliGemmaWithExpert(adaptive_expert=True)
        self.head = Pi05Head()

    def parts(self) -> dict[str, nn.Module]:
        return {**self.paligemma.parts(), "head": self.head}

    def velocity(self, prefix: Prefix, noisy: torch.Tensor, *, time: torch.Tensor,
                 depth: int) -> tuple[torch.Tensor, list[KeyValue]]:
        """One denoising step (`embed_suffix`, `denoise_step`): the velocity of
        the noisy chunk [chunk, 32] at `time`, and the expert's own keys and values."""
        head = self.head
        chunk = noisy.shape[0]
        embedded_time = sinusoidal(time[None], DECODER_DIM)
        cond = F.silu(head.time_mlp_out(F.silu(head.time_mlp_in(embedded_time))))
        # The first action token opens the chunk's one block; the chunk attends to itself fully.
        block_starts = (torch.arange(chunk, device=noisy.device) == 0)[None]
        hidden, cache = self.paligemma.expert_pass(prefix, head.action_in_proj(noisy[None]),
                                                   block_starts=block_starts, cond=cond, depth=depth)
        return head.action_out_proj(hidden[0, -chunk:].float()), cache

    def forward(self, images: torch.Tensor, *, image_masks: torch.Tensor, prompt_ids: torch.Tensor,
                prompt_mask: torch.Tensor, noise: torch.Tensor, steps: int = DEFAULT_FLOW_STEPS,
                depth: int = GEMMA_2B.depth) -> Pi05Outputs:
        """`sample_actions` for one observation. `depth` runs the first
        `depth` layers of both Gemma stacks and skips the rest, for bisection."""
        vision_hidden, image_tokens = self.paligemma.embed_images(images)
        prefix = self.paligemma.prefix(image_tokens, image_masks=image_masks, prompt_ids=prompt_ids,
                                       prompt_mask=prompt_mask, depth=depth)
        # Upstream walks t = 1, 1 + dt, ... while t >= -dt / 2, accumulating t
        # in float32; that is exactly `steps` iterations.
        dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=noise.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        noisy = noise.float()
        for _ in range(steps):
            velocity, suffix_cache = self.velocity(prefix, noisy, time=time, depth=depth)
            noisy = noisy + dt * velocity
            time = time + dt
        return Pi05Outputs(vision_hidden=vision_hidden, prefix=prefix, actions=noisy,
                           suffix_cache=suffix_cache)


def make_reference(*, device: str | torch.device = "meta") -> Pi05Reference:
    """The model without weights; on `meta` it allocates nothing, and its
    `official_schema` is the official checkpoint's."""
    with torch.device(device):
        return Pi05Reference().eval().requires_grad_(False)


def load(weights: Mapping[str, torch.Tensor], *, precision: Precision = "bfloat16") -> Pi05Reference:
    """The model bound to official tensors, in upstream's inference dtypes
    (`bfloat16`) or entirely in float32."""
    reference = make_reference()
    bind_upstream(reference.parts(), prefixes=PREFIXES, weights=weights, precision=precision)
    return reference


def random_weights(seed: int, *, device: str | torch.device = "cuda",
                   scale: float = 0.05) -> dict[str, torch.Tensor]:
    """Seeded official-layout weights (`official.random_official`); the small
    scale keeps deep random residual streams numerically useful."""
    return random_official(make_reference().parts(), prefixes=PREFIXES, seed=seed, device=device,
                           scale=scale)


__all__ = ["PREFIXES", "Pi05Outputs", "Pi05Reference", "load", "make_reference", "random_weights"]
