"""LingBot-VLA BF16 bring-up on Thor using the model reference stages."""
from flash_vla.hardware.nvidia.torch.lingbot import BACKEND
from flash_vla.models.lingbot.definition import LingBotModel
from flash_vla.models.lingbot.spec import CHECKPOINT_REVISION
from flash_vla.runtime.registry import Registry
from flash_vla.runtime.vla import Target

TARGET = Target(
    name="hardware/nvidia/thor/lingbot_vla",
    hardware="jetson-agx-thor",
    model=LingBotModel(),
    registry=Registry({"reference": BACKEND}, default="reference"),
    plan={},
    reference_plan={},
    workloads=("robodojo",),
    assets={
        "checkpoint": CHECKPOINT_REVISION,
        "fixture": "lingbot-robotwin-canonical-v1/seed-42",
    },
)

__all__ = ["TARGET"]
