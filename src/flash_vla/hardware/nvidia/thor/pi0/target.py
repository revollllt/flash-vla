"""Pi0 BF16 bring-up on Thor using the shared Torch backend."""
from flash_vla.hardware.nvidia.torch.pi0 import BACKEND
from flash_vla.models.pi0.definition import Pi0Model
from flash_vla.runtime.registry import Registry
from flash_vla.runtime.vla import Target

TARGET = Target(
    name="hardware/nvidia/thor/pi0",
    hardware="jetson-agx-thor",
    model=Pi0Model(),
    registry=Registry({"torch": BACKEND}, default="torch"),
    plan={},
    reference_plan={},
    workloads=("robodojo",),
)

__all__ = ["TARGET"]
