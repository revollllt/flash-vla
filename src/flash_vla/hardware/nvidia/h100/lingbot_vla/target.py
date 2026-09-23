"""LingBot-VLA-4B on H100, BF16: the upstream model with the measured replacement ladder.

The model is `flash_vla.models.lingbot`; this Target owns the backends (the
upstream composition and its `UpstreamRoute` ladder), the shipped plan and the
logical IDs of the assets a machine resolves to files.
"""
from __future__ import annotations

from flash_vla.models.lingbot.definition import LingBotModel
from flash_vla.models.lingbot.spec import CHECKPOINT_REVISION
from flash_vla.runtime.vla import Target

from .backends import REGISTRY

TARGET = Target(
    name="hardware/nvidia/h100/lingbot_vla",
    hardware="h100-sxm5-80gb",
    model=LingBotModel(),
    registry=REGISTRY,
    plan={
        "lingbot_vision": "vision-attention",
        "lingbot_prefix": "vision-attention",
        "lingbot_action": "vision-attention",
    },
    reference_plan={},
    assets={
        "checkpoint": CHECKPOINT_REVISION,
        "fixture": "lingbot-robotwin-canonical-v1/seed-42",
        "upstream": "lingbot-vla@4eb34b7693a0565c67433f8fac9c59a2e67eb60b",
        "qwen": "qwen2.5-vl-3b@66285546d2b821cf421d4f5eb2576359d3770cd3",
    },
)

__all__ = ["TARGET"]
