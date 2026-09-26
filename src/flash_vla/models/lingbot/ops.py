"""The LingBot graph's call sites: vision, prefix and action as one op each.

Each stage takes its weights as numbered parameters -- the vision op every
weight of the model (`WEIGHT_PARAMS`), which the upstream route loads as one
policy; the prefix and action ops their towers' (`BACKBONE_WEIGHT_PARAMS`,
`ACTION_WEIGHT_PARAMS`) -- and the graph binds them to the checkpoint names
of `spec`.
"""
from __future__ import annotations

from flash_vla.runtime.ops import OpSpec

from .spec import (
    BACKBONE_WEIGHT_NAMES,
    EXPERT_WEIGHT_NAMES,
    WEIGHT_NAMES,
)

#: The three monolithic call sites of the LingBot graph.
CALL_SITES = frozenset({"lingbot_vision", "lingbot_prefix", "lingbot_action"})
WEIGHT_PARAMS = tuple(f"weight_{index:04d}" for index in range(len(WEIGHT_NAMES)))
BACKBONE_WEIGHT_PARAMS = tuple(
    f"backbone_weight_{index:04d}" for index in range(len(BACKBONE_WEIGHT_NAMES))
)
ACTION_WEIGHT_PARAMS = tuple(
    f"action_weight_{index:04d}" for index in range(len(EXPERT_WEIGHT_NAMES))
)


OPS = (
    OpSpec(
        "lingbot_vision",
        ("pixel_values", "out", "layers") + WEIGHT_PARAMS,
        outputs=("out",),
        weights=WEIGHT_PARAMS,
    ),
    OpSpec(
        "lingbot_prefix",
        (
            "vision", "image_masks", "language_tokens", "language_masks",
            "prefix_masks", "prefix_k", "prefix_v", "layers",
        ) + BACKBONE_WEIGHT_PARAMS,
        outputs=("prefix_masks", "prefix_k", "prefix_v"),
        weights=BACKBONE_WEIGHT_PARAMS,
    ),
    OpSpec(
        "lingbot_action",
        (
            "state", "noise", "prefix_masks", "prefix_k", "prefix_v", "actions",
            "velocity_step_0", "steps", "layers",
        ) + ACTION_WEIGHT_PARAMS,
        outputs=("actions", "velocity_step_0"),
        weights=ACTION_WEIGHT_PARAMS,
    ),
)


__all__ = ["ACTION_WEIGHT_PARAMS", "BACKBONE_WEIGHT_PARAMS", "CALL_SITES", "OPS", "WEIGHT_PARAMS"]
