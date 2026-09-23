"""Pi0.5 vision LayerNorm and post-addmm GELU on handwritten CUDA.

BF16 CUDA x(views,256,1152), norm weight/bias(1152) normalize in FP32 and
round to BF16 before unchanged torch.addmm. QKV writes out(views,256,3456).
FFN writes out(views,256,4304), applying tanh GELU in place only after addmm
has rounded to BF16. Scratch is per runner; warmup compiles and allocates it
before CUDA Graph capture. Rows and affine parameters are contiguous.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache, partial
from pathlib import Path

import torch

from flash_vla.hardware.nvidia.native import NativeLibrary
from flash_vla.runtime.registry import Backend

NAMES = frozenset({"vision_encoder_norm_qkv", "vision_encoder_norm_ffn_up"})


SOURCE = Path(__file__).with_suffix(".cu")

#: The vision encoder's LayerNorm and GELU kernels.
LIBRARY = NativeLibrary(
    name="rtx5090_pi05_fused_vision",
    sources=(SOURCE,),
    arch=("-arch=sm_120a",),
    flags=("--fmad=false",))


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The loaded library with its C ABI declared; build it before graph capture."""
    kernels = LIBRARY.load()
    kernels.pi05_vision_layer_norm_launch.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int32, ctypes.c_void_p])
    kernels.pi05_vision_layer_norm_launch.restype = ctypes.c_int32
    kernels.pi05_vision_gelu_launch.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    kernels.pi05_vision_gelu_launch.restype = ctypes.c_int32
    return kernels


def _norm(x, weight, bias, scratch):
    rows = x.numel() // 1152
    normalized = scratch("pi05_vision_normalized", (rows, 1152), x.dtype, x.device)
    rc = library().pi05_vision_layer_norm_launch(
        x.data_ptr(), weight.data_ptr(), bias.data_ptr(), normalized.data_ptr(),
        rows, torch.cuda.current_stream(x.device).cuda_stream)
    if rc:
        raise RuntimeError(f"pi05_vision_layer_norm rows={rows}, threads=256: CUDA error {rc}")
    return normalized


def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out, *, scratch):
    """Normalize in FP32 then write the existing BF16 bias-fused QKV projection."""
    normalized = _norm(x, norm_w, norm_b, scratch)
    torch.addmm(qkv_b, normalized, qkv_w, out=out.view(-1, 3456))
    return out


def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out, *, scratch):
    """Normalize and project before in-place GELU of the BF16-rounded projection."""
    normalized = _norm(x, norm_w, norm_b, scratch)
    torch.addmm(bias, normalized, weight, out=out.view(-1, 4304))
    rc = library().pi05_vision_gelu_launch(
        out.data_ptr(), out.numel(), torch.cuda.current_stream(out.device).cuda_stream)
    if rc:
        raise RuntimeError(f"pi05_vision_gelu elements={out.numel()}, threads=256: CUDA error {rc}")
    return out


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind only the selected vision wrappers to the runner's scratch."""
    names = NAMES if selected_names is None else selected_names
    wrappers = {"vision_encoder_norm_qkv": vision_encoder_norm_qkv,
                "vision_encoder_norm_ffn_up": vision_encoder_norm_ffn_up}
    return {name: partial(wrappers[name], scratch=scratch) for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
