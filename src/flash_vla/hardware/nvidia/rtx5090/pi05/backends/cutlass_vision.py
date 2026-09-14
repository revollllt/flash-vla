"""Pi0.5 vision projections on one CUTLASS bias-GEMM tile and pointwise stages.

Config 10 preserves FP32 accumulation plus BF16 bias before its BF16 output.
Up then applies the existing GELU; residual sites add the BF16 projection to out.
Plans and device workspaces belong to the wrapper instance and runner scratch.
"""
from __future__ import annotations

import ctypes
import weakref

import torch

from . import cutlass_backbone, fused_vision

NAMES = frozenset({"vision_encoder_norm_qkv", "vision_encoder_norm_ffn_up",
                   "vision_encoder_ffn_down_residual", "vision_encoder_out_proj_residual"})


def _library():
    library = cutlass_backbone._library()
    library.vision_bias_gemm_workspace.argtypes = [ctypes.c_int32] * 3
    library.vision_bias_gemm_workspace.restype = ctypes.c_int64
    library.vision_bias_gemm_plan.argtypes = (
        [ctypes.c_int32] * 3 + [ctypes.c_void_p] * 6
        + [ctypes.POINTER(ctypes.c_void_p)])
    library.vision_bias_gemm_plan.restype = ctypes.c_int32
    library.vision_bias_gemm_run.argtypes = [ctypes.c_void_p] * 2
    library.vision_bias_gemm_run.restype = ctypes.c_int32
    library.vision_bias_gemm_destroy.argtypes = [ctypes.c_void_p]
    library.vision_bias_gemm_destroy.restype = None
    return library


class _Plan:
    def __init__(self, library, scratch, role, a, weight, bias, output, stream):
        m, k = a.shape
        n = weight.shape[1]
        size = library.vision_bias_gemm_workspace(m, k, n)
        self.workspace = scratch(role, (max(size, 1),), torch.uint8, a.device)
        self.tensors = (a, weight, bias, output)
        self.handle = ctypes.c_void_p()
        cutlass_backbone._check(library.vision_bias_gemm_plan(
            m, k, n, a.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
            self.workspace.data_ptr(), stream, ctypes.byref(self.handle)),
            f"vision_bias_gemm_plan M={m} K={k} N={n}")
        self.destroy = weakref.finalize(self, library.vision_bias_gemm_destroy, self.handle)


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind contiguous CUDA BF16 vision tensors without loading native code.

    QKV: x(views,256,1152), norm vectors(1152), weight(1152,3456), bias(3456),
    out(views,256,3456). Up: same x/norm, weight(1152,4304), bias(4304),
    out(views,256,4304). Down: x(views,256,4304), weight(4304,1152), bias(1152),
    out(views,256,1152). Attention out-projection has K=1152 instead of 4304.
    Both residual sites read and update out in place, with res aliasing out. Warmup
    plans all pointer sets and scratch before capture; replay allocates nothing.
    """
    names = NAMES if selected_names is None else set(selected_names)
    library = pointwise = None
    plans = {}
    role = f"pi05_vision_streamk_{id(plans)}"

    def gemm(a, weight, bias, output, stream):
        nonlocal library
        key = (a.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr())
        plan = plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("vision GEMM pointer set was not warmed before capture")
            if library is None:
                library = _library()
            plan = _Plan(library, scratch, role, a, weight, bias, output, stream)
            plans[key] = plan
        cutlass_backbone._check(library.vision_bias_gemm_run(plan.handle, stream),
                               f"vision_bias_gemm_run M={a.shape[0]} K={a.shape[1]}")

    def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out):
        normalized = fused_vision._norm(x, norm_w, norm_b, scratch)
        gemm(normalized, qkv_w, qkv_b, out.view(-1, 3456),
             torch.cuda.current_stream().cuda_stream)
        return out

    def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out):
        nonlocal pointwise
        normalized = fused_vision._norm(x, norm_w, norm_b, scratch)
        stream = torch.cuda.current_stream().cuda_stream
        gemm(normalized, weight, bias, out.view(-1, 4304), stream)
        if pointwise is None:
            pointwise = fused_vision._library()
        cutlass_backbone._check(pointwise.pi05_vision_gelu_launch(
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


__all__ = ["NAMES", "make_wrappers"]
