"""Three-stage fixed-shape LingBot graph."""
from __future__ import annotations

from typing import Mapping

import torch

from flash_vla.models.lingbot.spec import (
    ACTION_DIM,
    BACKBONE_DIM,
    BACKBONE_WEIGHT_NAMES,
    CHUNK,
    EXPERT_WEIGHT_NAMES,
    HEAD_DIM,
    KV_HEADS,
    LANGUAGE_SLOTS,
    LAYERS,
    PATCH_ROWS_PER_VIEW,
    PATCH_WIDTH,
    PREFIX_LEN,
    STATE_DIM,
    VIEWS,
    VISUAL_TOKENS_PER_VIEW,
    WEIGHT_NAMES,
)
from flash_vla.runtime.graph import Graph

from .backends.upstream import ACTION_WEIGHT_PARAMS, BACKBONE_WEIGHT_PARAMS, WEIGHT_PARAMS


def build(g: Graph, shape: Mapping[str, int]) -> None:
    pixel_values = g.buf("pixel_values", (VIEWS, PATCH_ROWS_PER_VIEW, PATCH_WIDTH))
    image_masks = g.buf("image_masks", (VIEWS,), dtype=torch.bool)
    language_tokens = g.buf("language_tokens", (1, LANGUAGE_SLOTS), dtype=torch.int64)
    language_masks = g.buf("language_masks", (1, LANGUAGE_SLOTS), dtype=torch.bool)
    state = g.buf("state", (1, STATE_DIM))
    noise = g.buf("noise", (1, CHUNK, ACTION_DIM))

    g.stage("vision_encoder")
    vision = g.buf("vision_embeddings", (VIEWS, VISUAL_TOKENS_PER_VIEW, BACKBONE_DIM))
    vision_args = {
        param: g.w(name) for param, name in zip(WEIGHT_PARAMS, WEIGHT_NAMES)
    }
    g.op("lingbot_vision", pixel_values=pixel_values, out=vision,
         layers=shape["layers"], **vision_args)

    g.stage("llm_backbone")
    prefix_masks = g.buf("prefix_masks", (1, PREFIX_LEN), dtype=torch.bool)
    prefix_k = g.buf("prefix_k", (LAYERS, PREFIX_LEN, KV_HEADS, HEAD_DIM),
                     dtype=torch.float32, init="zero")
    prefix_v = g.buf("prefix_v", (LAYERS, PREFIX_LEN, KV_HEADS, HEAD_DIM),
                     dtype=torch.float32, init="zero")
    backbone_args = {
        param: g.w(name)
        for param, name in zip(BACKBONE_WEIGHT_PARAMS, BACKBONE_WEIGHT_NAMES)
    }
    g.op("lingbot_prefix", vision=vision, image_masks=image_masks,
         language_tokens=language_tokens, language_masks=language_masks,
         prefix_masks=prefix_masks, prefix_k=prefix_k, prefix_v=prefix_v,
         layers=shape["layers"], **backbone_args)

    g.stage("action_expert")
    actions = g.buf("actions", (1, CHUNK, ACTION_DIM))
    velocity = g.buf("velocity_step_0", (1, CHUNK, ACTION_DIM))
    action_args = {
        param: g.w(name)
        for param, name in zip(ACTION_WEIGHT_PARAMS, EXPERT_WEIGHT_NAMES)
    }
    g.op("lingbot_action", state=state, noise=noise, prefix_masks=prefix_masks,
         prefix_k=prefix_k, prefix_v=prefix_v, actions=actions,
         velocity_step_0=velocity, steps=shape["steps"], layers=shape["layers"],
         **action_args)


__all__ = ["build"]
