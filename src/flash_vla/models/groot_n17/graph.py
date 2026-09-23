"""Explicit vision, language and flow-matching stages for GR00T N1.7."""
from __future__ import annotations

from typing import Mapping

import torch

from flash_vla.runtime.graph import Graph

from .ops import PARAMS, WEIGHTS
from .spec import (
    ACTION_DIM,
    BACKBONE_DIM,
    CHUNK,
    DEEPSTACK_LAYERS,
    PATCH_WIDTH,
    PATCHES,
    STATE_DIM,
    VISUAL_TOKENS,
)


def build(g: Graph, shape: Mapping[str, int]) -> None:
    length = shape["sequence_length"]
    # The runtime warms up before the first observation is staged. Integer
    # gather/category indices must already be valid on a reused allocator.
    pixels = g.buf("pixel_values", (PATCHES, PATCH_WIDTH), init="zero")
    ids = g.buf("input_ids", (1, length), dtype=torch.int64, init="zero")
    mask = g.buf("attention_mask", (1, length), dtype=torch.int64, init="zero")
    positions = g.buf("position_ids", (3, 1, length), dtype=torch.int64, init="zero")
    image_indices = g.buf("image_indices", (VISUAL_TOKENS,), dtype=torch.int64, init="zero")
    state = g.buf("state", (1, 1, STATE_DIM), init="zero")
    embodiment = g.buf("embodiment_id", (1,), dtype=torch.int64, init="zero")
    noise = g.buf("noise", (1, CHUNK, ACTION_DIM), init="zero")
    weights = {part: {param: g.w(name) for param, name in zip(PARAMS[part], names)}
               for part, names in WEIGHTS.items()}

    g.stage("vision_encoder")
    vision = g.buf("vision_embeddings", (VISUAL_TOKENS, BACKBONE_DIM))
    deepstack = g.buf("deepstack", (DEEPSTACK_LAYERS, VISUAL_TOKENS, BACKBONE_DIM))
    g.op("groot_vision", pixels=pixels, out=vision, deepstack=deepstack, **weights["vision"])

    g.stage("llm_backbone")
    backbone = g.buf("backbone_features", (1, length, BACKBONE_DIM))
    g.op("groot_backbone", input_ids=ids, attention_mask=mask, position_ids=positions,
         image_indices=image_indices, vision=vision, deepstack=deepstack, out=backbone,
         **weights["backbone"])

    g.stage("action_expert")
    actions = g.buf("actions", (1, CHUNK, ACTION_DIM))
    velocity = g.buf("velocity_step_0", (1, CHUNK, ACTION_DIM))
    g.op("groot_action", backbone=backbone, state=state, noise=noise, embodiment=embodiment,
         input_ids=ids, attention_mask=mask, out=actions, velocity=velocity, **weights["action"])


__all__ = ["build"]
