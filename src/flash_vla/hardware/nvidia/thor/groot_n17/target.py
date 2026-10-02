"""GR00T N1.7 BF16 bring-up on Thor using model reference stages."""
from flash_vla.hardware.nvidia.torch.groot_n17 import BACKEND
from flash_vla.models.groot_n17.definition import GrootModel
from flash_vla.models.groot_n17.weights import CHECKPOINT_ID, FIXTURE_ID
from flash_vla.runtime.registry import Registry
from flash_vla.runtime.vla import Target

TARGET = Target(
    name="hardware/nvidia/thor/groot_n17",
    hardware="jetson-agx-thor",
    model=GrootModel(),
    registry=Registry({"torch": BACKEND}, default="torch"),
    plan={},
    reference_plan={},
    workloads=("libero", "robodojo"),
    assets={"checkpoint": CHECKPOINT_ID, "fixture": FIXTURE_ID},
)

__all__ = ["TARGET"]
