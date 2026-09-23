"""Pi0.5 backbone FFN with CUDA RMSNorm and a fused GELU/product.

The two torch BF16 GEMMs retain the torch route's rounding boundaries.
The wrapper writes its declared output and x_norm buffers; the runner owns
its gate workspace. Native launch state belongs to each wrapper factory.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from flash_vla.hardware.nvidia.native import NativeLibrary

SOURCE = Path(__file__).with_suffix(".cu")


#: The backbone's RMSNorm and gated-activation kernels.
LIBRARY = NativeLibrary(
    name="rtx5090_pi05_fused_backbone",
    sources=(SOURCE,),
    arch=("-gencode", "arch=compute_120,code=sm_120"),
    flags=())


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The loaded library with its C ABI declared; build it before graph capture."""
    kernels = LIBRARY.load()
    kernels.backbone_rms_norm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
    ]
    kernels.backbone_rms_norm.restype = ctypes.c_int32
    kernels.backbone_gelu_mul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
    ]
    kernels.backbone_gelu_mul.restype = ctypes.c_int32
    return kernels


