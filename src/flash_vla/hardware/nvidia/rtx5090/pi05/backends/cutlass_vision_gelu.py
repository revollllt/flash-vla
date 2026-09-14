"""Pi0.5 vision FFN up with cfg10 GEMM and rounded tanh GELU.

FP32 accumulation plus BF16 bias rounds to BF16 before the existing GELU.
The original norm and other vision sites retain their current implementations.
"""
from __future__ import annotations

import ctypes
import weakref

import torch

from . import cutlass_backbone, fused_vision

NAMES = frozenset({"vision_encoder_norm_ffn_up"})


def _library():
    library = cutlass_backbone._library()
    library.vision_gelu_gemm_workspace.argtypes = [ctypes.c_int32] * 3
    library.vision_gelu_gemm_workspace.restype = ctypes.c_int64
    library.vision_gelu_gemm_plan.argtypes = (
        [ctypes.c_int32] * 3 + [ctypes.c_void_p] * 6
        + [ctypes.POINTER(ctypes.c_void_p)])
    library.vision_gelu_gemm_plan.restype = ctypes.c_int32
    library.vision_gelu_gemm_run.argtypes = [ctypes.c_void_p] * 2
    library.vision_gelu_gemm_run.restype = ctypes.c_int32
    library.vision_gelu_gemm_destroy.argtypes = [ctypes.c_void_p]
    library.vision_gelu_gemm_destroy.restype = None
    return library


class _Plan:
    def __init__(self, library, scratch, role, a, weight, bias, output, stream):
        m, k = a.shape
        n = weight.shape[1]
        size = library.vision_gelu_gemm_workspace(m, k, n)
        self.workspace = scratch(role, (max(size, 1),), torch.uint8, a.device)
        self.tensors = (a, weight, bias, output)
        self.handle = ctypes.c_void_p()
        cutlass_backbone._check(library.vision_gelu_gemm_plan(
            m, k, n, a.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
            self.workspace.data_ptr(), stream, ctypes.byref(self.handle)),
            f"vision_gelu_gemm_plan M={m} K={k} N={n}")
        self.destroy = weakref.finalize(self, library.vision_gelu_gemm_destroy, self.handle)


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind CUDA BF16 x(views,256,1152), norm vectors(1152), weight(1152,4304),
    bias(4304), and out(views,256,4304), without loading native code.
    Warmup owns plans and scratch for each pointer set before graph capture.
    """
    names = NAMES if selected_names is None else set(selected_names)
    library = None
    plans = {}
    role = f"pi05_vision_rounded_gelu_{id(plans)}"

    def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out):
        nonlocal library
        normalized = fused_vision._norm(x, norm_w, norm_b, scratch)
        output = out.view(-1, 4304)
        stream = torch.cuda.current_stream().cuda_stream
        key = (normalized.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr())
        plan = plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("vision GELU GEMM pointer set was not warmed before capture")
            if library is None:
                library = _library()
            plan = _Plan(library, scratch, role, normalized, weight, bias, output, stream)
            plans[key] = plan
        cutlass_backbone._check(library.vision_gelu_gemm_run(plan.handle, stream),
                               "vision_gelu_gemm_run")
        return out

    wrappers = {"vision_encoder_norm_ffn_up": vision_encoder_norm_ffn_up}
    return {name: wrappers[name] for name in names}


__all__ = ["NAMES", "make_wrappers"]
