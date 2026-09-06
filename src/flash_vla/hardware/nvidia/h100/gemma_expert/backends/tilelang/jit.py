"""This package's TileLang JIT namespace and its own raw-builder registry.

One `KernelSet` per declaring module keeps `RAW_KERNELS` private, so the
expert's `tl_rms_factor` never collides with the same-named, differently-bodied
kernel in a Target's own `kernels/base.py`
(`flash_vla.hardware.nvidia.tilelang.jit`).
"""
from __future__ import annotations

from flash_vla.hardware.nvidia.tilelang.jit import (
    FAST_MATH,
    NO_WARP_SPEC,
    KernelSet,
)

_KERNELS = KernelSet()
RAW_KERNELS = _KERNELS.RAW_KERNELS
variant = _KERNELS.variant
kernel = _KERNELS.kernel

__all__ = ["FAST_MATH", "NO_WARP_SPEC", "RAW_KERNELS", "kernel", "variant"]
