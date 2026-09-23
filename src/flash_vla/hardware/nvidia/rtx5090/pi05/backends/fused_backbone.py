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
from flash_vla.runtime.registry import Backend

NAMES = frozenset({"llm_backbone_norm_gated_ffn"})
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


def make_wrappers(scratch, selected_names=None) -> dict:
    """Build capture-safe wrappers for this Target's contiguous CUDA BF16 FFN.

    x/x_norm are (M, 2048), weights (2048, 16384), and out (M, 16384).
    x_norm and out are distinct writable buffers; out temporarily holds up.
    The runner must warm each shape before freezing the supplied allocator.
    """
    native = None

    def llm_backbone_norm_gated_ffn(x, gate_w, up_w, out, x_norm):
        nonlocal native
        if native is None:
            native = library()
        rows = x.shape[0]
        normed, result = x_norm[:rows], out[:rows]
        gate = scratch("backbone_ffn_gate", result.shape, x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        status = native.backbone_rms_norm(x.data_ptr(), normed.data_ptr(), rows, stream)
        if status:
            raise RuntimeError(f"backbone_rms_norm M={rows} K=2048: cudaError {status}")
        torch.mm(normed, gate_w, out=gate)
        torch.mm(normed, up_w, out=result)
        status = native.backbone_gelu_mul(
            gate.data_ptr(), result.data_ptr(), result.numel(), stream)
        if status:
            raise RuntimeError(
                f"backbone_gelu_mul elements={result.numel()}: cudaError {status}")
        return out

    return {"llm_backbone_norm_gated_ffn": llm_backbone_norm_gated_ffn}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)
