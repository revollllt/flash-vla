"""Pi0.5 RTX 5090 backbone RMSNorm, BF16 QKV GEMM and fused RoPE/scatter.

The GEMM is cuBLAS, which takes about 50 us at both 712 and 968 rows, or
`cutlass_backbone`'s stream-K config 0, which scales with the rows: at the
geometries of `CUTLASS_GEOMETRIES` it is more than 5% faster
(`lab/pi05/geometry_screen.py`).
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from flash_vla.hardware.nvidia.native import NativeLibrary
from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch

from . import cutlass_backbone, fused_backbone

NAMES = ("llm_backbone_norm_qkv_rope",)
#: (M, K, N): two views' 712-row prefix, 41 against 50 us per layer
#: (`results/pi05-rtx5090/workload-generalization/g1/geometry-screen.json`).
CUTLASS_GEOMETRIES = frozenset({(712, 2048, 2560)})
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


def make_wrappers(scratch: Scratch, selected_names: frozenset[str] | None = None
                  ) -> dict[str, Wrapper]:
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
    plans: dict[tuple[int, int, int, int, int, int, int, float], cutlass_backbone.GemmPlan] = {}
    role = f"pi05_prefix_qkv_{id(plans)}"

    def llm_backbone_norm_qkv_rope(x: torch.Tensor, weight_qkv: torch.Tensor,
                                   rope: torch.Tensor, Q: torch.Tensor, K: torch.Tensor,
                                   V: torch.Tensor, x_norm: torch.Tensor) -> None:
        rows = x.shape[0]
        normed = x_norm[:rows]
        projected = scratch(role, (rows, 2560), x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        status = fused_backbone.library().backbone_rms_norm(
            x.data_ptr(), normed.data_ptr(), rows, stream)
        if status:
            raise RuntimeError(f"backbone_rms_norm M={rows} K=2048: cudaError {status}")
        if (rows, *weight_qkv.shape) in CUTLASS_GEOMETRIES:
            cutlass_backbone.run_gemm(plans, scratch, normed, weight_qkv, projected, beta=0.0,
                                      stream=stream, config=0)
        else:
            torch.mm(normed, weight_qkv, out=projected)
        status = library().prefix_rope_scatter(
            projected.data_ptr(), rope.data_ptr(), Q.data_ptr(), K.data_ptr(),
            V.data_ptr(), rows, stream)
        if status:
            raise RuntimeError(f"prefix_rope_scatter M={rows} width=2560: cudaError {status}")

    return {name: llm_backbone_norm_qkv_rope for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
