"""LingBot reference backend shared across NVIDIA devices."""
from flash_vla.hardware.nvidia.torch.lingbot import BACKEND, make_wrappers

__all__ = ["BACKEND", "make_wrappers"]
