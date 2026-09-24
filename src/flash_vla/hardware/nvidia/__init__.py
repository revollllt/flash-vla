"""NVIDIA deployment targets."""

from types import MappingProxyType
from typing import Mapping

from flash_vla.hardware.roofline import Roofline

from .h100 import spec as h100_spec
from .rtx5090 import spec as rtx5090_spec

#: Identity hardware axis (`Target.hardware`) -> what the floor model reads of
#: that device, its spec's `ROOFLINE`.
HARDWARE_ROOFLINES: Mapping[str, Roofline] = MappingProxyType({
    "h100-sxm5-80gb": h100_spec.ROOFLINE,
    "rtx5090-32gb": rtx5090_spec.ROOFLINE,
})

__all__ = ["HARDWARE_ROOFLINES"]
