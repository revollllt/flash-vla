"""Pi0.5 BF16 bring-up on Thor using the shared Torch backend.

Both plans use the same route. No device-specific kernel tuning or quantization
recipe has been validated on Thor yet.
"""
from flash_vla.hardware.nvidia.torch.pi05 import BACKEND
from flash_vla.models.pi05.definition import Pi05Layout, Pi05Model
from flash_vla.runtime.registry import Registry
from flash_vla.runtime.vla import Target

TARGET = Target(
    name="hardware/nvidia/thor/pi05",
    hardware="jetson-agx-thor",
    model=Pi05Model(Pi05Layout(row_pad=64)),
    registry=Registry({"torch": BACKEND}, default="torch"),
    plan={},
    reference_plan={},
    workloads=("robodojo", "libero"),
)

__all__ = ["TARGET"]
