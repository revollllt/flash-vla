"""Pi0.5 vision projections on CUTLASS bias-GEMM tiles and pointwise stages.

Each tile preserves FP32 accumulation plus BF16 bias before its BF16 output.
Up then applies the existing GELU; residual sites add the BF16 projection to out.
The tile is chosen per GEMM geometry at plan time (`TILES`). Plans and device
workspaces belong to the wrapper instance and runner scratch.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import weakref

import torch

from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch

from . import cutlass_backbone, fused_vision

NAMES = frozenset({"vision_encoder_norm_qkv", "vision_encoder_norm_ffn_up",
                   "vision_encoder_ffn_down_residual", "vision_encoder_out_proj_residual"})
#: The stream-K family config (`cutlass_backbone.cu`) of each GEMM geometry
#: (M, K, N) where it beats config 10 by more than 5% in
#: `lab/pi05/geometry_screen.py`: two views' attention output projection.
TILES = {(512, 1152, 1152): 8}
DEFAULT_TILE = 10


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The CUTLASS library with this module's entry points declared."""
    kernels = cutlass_backbone.library()
    kernels.vision_bias_gemm_workspace.argtypes = [ctypes.c_int32] * 4
    kernels.vision_bias_gemm_workspace.restype = ctypes.c_int64
    kernels.vision_bias_gemm_plan.argtypes = (
        [ctypes.c_int32] * 4 + [ctypes.c_void_p] * 6
        + [ctypes.POINTER(ctypes.c_void_p)])
    kernels.vision_bias_gemm_plan.restype = ctypes.c_int32
    kernels.vision_bias_gemm_run.argtypes = [ctypes.c_void_p] * 2
    kernels.vision_bias_gemm_run.restype = ctypes.c_int32
    kernels.vision_bias_gemm_destroy.argtypes = [ctypes.c_void_p]
    kernels.vision_bias_gemm_destroy.restype = None
    return kernels


class VisionPlan:
    """output = a @ weight + bias on the family tile `TILES` names for this
    geometry, bound to these tensors' addresses. Planning allocates, so it
    happens in warmup, never during capture."""

    def __init__(self, scratch: Scratch, role: str, a: torch.Tensor, weight: torch.Tensor,
                 bias: torch.Tensor, output: torch.Tensor, stream: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("vision GEMM pointer set was not warmed before capture")
        native = library()
        m, k = a.shape
        n = weight.shape[1]
        tile = TILES.get((m, k, n), DEFAULT_TILE)
        size = native.vision_bias_gemm_workspace(tile, m, k, n)
        self.workspace = scratch(role, (max(size, 1),), torch.uint8, a.device)
        self.tensors = (a, weight, bias, output)
        self.handle = ctypes.c_void_p()
        cutlass_backbone.check(native.vision_bias_gemm_plan(
            tile, m, k, n, a.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
            self.workspace.data_ptr(), stream, ctypes.byref(self.handle)),
            f"vision_bias_gemm_plan config={tile} M={m} K={k} N={n}")
        self.destroy = weakref.finalize(self, native.vision_bias_gemm_destroy, self.handle)


def make_wrappers(scratch: Scratch, selected_names: frozenset[str] | None = None
                  ) -> dict[str, Wrapper]:
    """Bind contiguous CUDA BF16 vision tensors without loading native code.

    QKV: x(views,256,1152), norm vectors(1152), weight(1152,3456), bias(3456),
    out(views,256,3456). Up: same x/norm, weight(1152,4304), bias(4304),
    out(views,256,4304). Down: x(views,256,4304), weight(4304,1152), bias(1152),
    out(views,256,1152). Attention out-projection has K=1152 instead of 4304.
    Both residual sites read and update out in place, with res aliasing out. Warmup
    plans all pointer sets and scratch before capture; replay allocates nothing.
    """
    names = NAMES if selected_names is None else set(selected_names)
    plans: dict[tuple[int, int, int, int], VisionPlan] = {}
    role = f"pi05_vision_streamk_{id(plans)}"

    def gemm(a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, output: torch.Tensor,
             stream: int) -> None:
        key = (a.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr())
        plan = plans[key] if key in plans else plans.setdefault(
            key, VisionPlan(scratch, role, a, weight, bias, output, stream))
        cutlass_backbone.check(library().vision_bias_gemm_run(plan.handle, stream),
                               f"vision_bias_gemm_run M={a.shape[0]} K={a.shape[1]}")

    def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out):
        normalized = fused_vision._norm(x, norm_w, norm_b, scratch)
        gemm(normalized, qkv_w, qkv_b, out.view(-1, 3456),
             torch.cuda.current_stream().cuda_stream)
        return out

    def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out):
        normalized = fused_vision._norm(x, norm_w, norm_b, scratch)
        stream = torch.cuda.current_stream().cuda_stream
        gemm(normalized, weight, bias, out.view(-1, 4304), stream)
        cutlass_backbone.check(fused_vision.library().pi05_vision_gelu_launch(
            out.data_ptr(), out.numel(), stream), "pi05_vision_gelu")
        return out

    def projection_residual(x, weight, bias, res, out):
        k = weight.shape[0]
        projected = scratch(f"{role}_projection", (x.numel() // k, 1152),
                            x.dtype, x.device)
        gemm(x.view(-1, k), weight, bias, projected,
             torch.cuda.current_stream().cuda_stream)
        out.add_(projected.view_as(out))
        return out

    wrappers = {
        "vision_encoder_norm_qkv": vision_encoder_norm_qkv,
        "vision_encoder_norm_ffn_up": vision_encoder_norm_ffn_up,
        "vision_encoder_ffn_down_residual": projection_residual,
        "vision_encoder_out_proj_residual": projection_residual,
    }
    return {name: wrappers[name] for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
