"""GR00T reference backend shared across NVIDIA devices."""
from flash_vla.hardware.nvidia.torch.groot_n17 import BACKEND, make_wrappers

__all__ = ["BACKEND", "make_wrappers"]
