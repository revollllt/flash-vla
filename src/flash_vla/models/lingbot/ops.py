"""The LingBot graph's call sites: three monolithic stages over the frozen upstream model.

Each stage takes every weight of its tower as a numbered parameter, so the op
specs name those parameters (`WEIGHT_PARAMS` and its backbone and action
halves) and the graph binds them to the checkpoint names of `spec`. The FLOP
formulas are the dense matmul and attention work of the fixed shapes; they are
what the floor model prices, not a claim about any implementation.
"""
from __future__ import annotations

from math import prod
from typing import Mapping

from flash_vla.runtime.ops import OpSpec, Shapes

from .spec import (
    ACTION_DIM,
    BACKBONE_DIM,
    BACKBONE_FFN,
    BACKBONE_WEIGHT_NAMES,
    CHUNK,
    EXPERT_DIM,
    EXPERT_FFN,
    EXPERT_WEIGHT_NAMES,
    HEAD_DIM,
    KV_HEADS,
    LAYERS,
    PATCH_ROWS_PER_VIEW,
    PATCH_WIDTH,
    PREFIX_LEN,
    QUERY_HEADS,
    VIEWS,
    VISION_DIM,
    VISION_FFN,
    VISION_LAYERS,
    VISION_WEIGHT_NAMES,
    VISUAL_TOKENS_PER_VIEW,
    WEIGHT_NAMES,
)

#: Denoising steps the cost model counts: the full schedule.
FLOW_STEPS = 10
#: Vision blocks with full per-view attention; the rest attend within windows.
FULL_ATTENTION_LAYERS = 4
#: Patches per attention window of the windowed vision blocks.
WINDOW_TOKENS = 64

#: The three monolithic call sites of the LingBot graph.
CALL_SITES = frozenset({"lingbot_vision", "lingbot_prefix", "lingbot_action"})
WEIGHT_PARAMS = tuple(f"weight_{index:04d}" for index in range(len(WEIGHT_NAMES)))
BACKBONE_WEIGHT_PARAMS = tuple(
    f"backbone_weight_{index:04d}" for index in range(len(BACKBONE_WEIGHT_NAMES))
)
ACTION_WEIGHT_PARAMS = tuple(
    f"action_weight_{index:04d}" for index in range(len(EXPERT_WEIGHT_NAMES))
)

VISION_PARAMS = tuple(
    param for param, name in zip(WEIGHT_PARAMS, WEIGHT_NAMES)
    if name in set(VISION_WEIGHT_NAMES)
)


def vision_bytes(shapes: Shapes, itemsizes: Mapping[str, int]) -> tuple[int, int]:
    """Pixels and the vision weights read, the embeddings written."""
    read = sum(prod(shapes[name]) * itemsizes[name] for name in ("pixel_values", *VISION_PARAMS))
    return read, prod(shapes["out"]) * itemsizes["out"]


def vision_flops(shapes: Shapes) -> int:
    rows = VIEWS * PATCH_ROWS_PER_VIEW
    linear = 2 * rows * PATCH_WIDTH * VISION_DIM
    per_layer = (
        2 * rows * VISION_DIM * (3 * VISION_DIM)
        + 2 * rows * VISION_DIM * VISION_DIM
        + 6 * rows * VISION_DIM * VISION_FFN
    )
    full_attention = 4 * FULL_ATTENTION_LAYERS * VIEWS * PATCH_ROWS_PER_VIEW**2 * VISION_DIM
    windows = rows // WINDOW_TOKENS
    window_attention = (4 * (VISION_LAYERS - FULL_ATTENTION_LAYERS) * windows
                        * WINDOW_TOKENS**2 * VISION_DIM)
    merger = 2 * VIEWS * VISUAL_TOKENS_PER_VIEW * (4 * VISION_DIM) * (4 * VISION_DIM + BACKBONE_DIM)
    return linear + VISION_LAYERS * per_layer + full_attention + window_attention + merger


def prefix_flops(shapes: Shapes) -> int:
    rows = PREFIX_LEN
    projections = 2 * rows * BACKBONE_DIM * (
        QUERY_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM
    )
    attention = 4 * QUERY_HEADS * rows * rows * HEAD_DIM
    output = 2 * rows * QUERY_HEADS * HEAD_DIM * BACKBONE_DIM
    mlp = 6 * rows * BACKBONE_DIM * BACKBONE_FFN
    return LAYERS * (projections + attention + output + mlp)


def action_flops(shapes: Shapes) -> int:
    rows = CHUNK + 1
    projections = 2 * rows * EXPERT_DIM * (
        QUERY_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM
    )
    attention = 4 * QUERY_HEADS * rows * (PREFIX_LEN + rows) * HEAD_DIM
    output = 2 * rows * QUERY_HEADS * HEAD_DIM * EXPERT_DIM
    mlp = 6 * rows * EXPERT_DIM * EXPERT_FFN
    layer = projections + attention + output + mlp
    outer = (
        2 * EXPERT_DIM * ACTION_DIM
        + 2 * CHUNK * ACTION_DIM * EXPERT_DIM
        + 2 * CHUNK * 2 * EXPERT_DIM * EXPERT_DIM
        + 2 * CHUNK * EXPERT_DIM * ACTION_DIM
    )
    return FLOW_STEPS * (LAYERS * layer + outer)


OPS = (
    OpSpec(
        "lingbot_vision",
        ("pixel_values", "out", "layers") + WEIGHT_PARAMS,
        outputs=("out",),
        weights=WEIGHT_PARAMS,
        flops=vision_flops,
        bytes_override=vision_bytes,
    ),
    OpSpec(
        "lingbot_prefix",
        (
            "vision", "image_masks", "language_tokens", "language_masks",
            "prefix_masks", "prefix_k", "prefix_v", "layers",
        ) + BACKBONE_WEIGHT_PARAMS,
        outputs=("prefix_masks", "prefix_k", "prefix_v"),
        weights=BACKBONE_WEIGHT_PARAMS,
        flops=prefix_flops,
    ),
    OpSpec(
        "lingbot_action",
        (
            "state", "noise", "prefix_masks", "prefix_k", "prefix_v", "actions",
            "velocity_step_0", "steps", "layers",
        ) + ACTION_WEIGHT_PARAMS,
        outputs=("actions", "velocity_step_0"),
        weights=ACTION_WEIGHT_PARAMS,
        flops=action_flops,
    ),
)


__all__ = ["ACTION_WEIGHT_PARAMS", "BACKBONE_WEIGHT_PARAMS", "CALL_SITES", "FLOW_STEPS", "OPS",
           "WEIGHT_PARAMS"]
