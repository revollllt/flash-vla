"""Pi0.5 RTX 5090 backbone RMSNorm, BF16 QKV GEMM and fused RoPE/scatter."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from flash_vla.hardware.nvidia.native import NativeLibrary
from flash_vla.runtime.registry import Backend

from . import fused_backbone

NAMES = ("llm_backbone_norm_qkv_rope",)
SOURCE = Path(__file__).with_suffix(".cu")


#: The backbone's RoPE scatter into the prefix KV cache.
LIBRARY = NativeLibrary(
    name="rtx5090_pi05_fused_prefix_qkv",
    sources=(SOURCE,),
    arch=("-arch=sm_120",),
    flags=("--fmad=false",))


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The loaded library with its C ABI declared; build it before graph capture."""
    kernels = LIBRARY.load()
    kernels.prefix_rope_scatter.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_int32, ctypes.c_void_p]
    kernels.prefix_rope_scatter.restype = ctypes.c_int32
    return kernels


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind contiguous CUDA BF16 prefix QKV: Mx2048 @ 2048x2560.

    RoPE is (M, 256); Q is (M*8, 256), K/V are (M, 256). The wrapper writes
    Q/K/V and x_norm (M, 2048). RMSNorm rounds its normalized output to BF16
    before the GEMM; the GEMM rounds to BF16 before FP32 RoPE arithmetic.
    Warmup builds native libraries and allocates scratch before the runner
    freezes it for capture. Libraries and scratch ownership stay within this
    wrapper factory; constructing the op table requires no CUDA toolchain.
    """
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"prefix QKV backend does not implement {sorted(unknown)}")
    libraries = []
    role = f"pi05_prefix_qkv_{id(libraries)}"

    def llm_backbone_norm_qkv_rope(x, weight_qkv, rope, Q, K, V, x_norm):
        if not libraries:
            libraries.extend((fused_backbone.library(), library()))
        norm_library, rope_library = libraries
        rows = x.shape[0]
        normed = x_norm[:rows]
        projected = scratch(role, (rows, 2560), x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        status = norm_library.backbone_rms_norm(x.data_ptr(), normed.data_ptr(), rows, stream)
        if status:
            raise RuntimeError(f"backbone_rms_norm M={rows} K=2048: cudaError {status}")
        torch.mm(normed, weight_qkv, out=projected)
        status = rope_library.prefix_rope_scatter(
            projected.data_ptr(), rope.data_ptr(), Q.data_ptr(), K.data_ptr(),
            V.data_ptr(), rows, stream)
        if status:
            raise RuntimeError(f"prefix_rope_scatter M={rows} width=2560: cudaError {status}")

    return {name: llm_backbone_norm_qkv_rope for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
