"""Pi0.5 backbone FFN on one CUTLASS Stream-K tile and CUDA pointwise stages.

The measured 128x128x64 tile serves both gate/up and down GEMMs. All nonlinear
rounding matches fused_backbone; only its two torch GEMMs are replaced.
Native plans are instance-owned and bind the runner's stable tensor pointers.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import weakref

import torch

from flash_vla.hardware.nvidia.native import CUTLASS_DIR, CUTLASS_VERSION_HEADER, NativeLibrary
from flash_vla.runtime.registry import Backend

from . import fused_backbone

NAMES = frozenset({
    "llm_backbone_norm_gated_ffn", "llm_backbone_ffn_down_residual",
    "llm_backbone_norm_gated_ffn_masked", "llm_backbone_ffn_down_residual_masked",
})
SOURCE = Path(__file__).with_suffix(".cu")


#: The CUTLASS backbone and expert GEMMs, stream-K and row-bucketed.
LIBRARY = NativeLibrary(
    name="rtx5090_pi05_cutlass_backbone",
    sources=(SOURCE,),
    arch=("-gencode", "arch=compute_120a,code=sm_120a"),
    flags=("--expt-relaxed-constexpr", "-Xptxas=-v"),
    include_dirs=(CUTLASS_DIR / "include",),
    headers=(CUTLASS_VERSION_HEADER,))


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The loaded library with its C ABI declared; build it before graph capture."""
    kernels = LIBRARY.load()
    kernels.backbone_gemm_workspace.argtypes = [ctypes.c_int32] * 3
    kernels.backbone_gemm_workspace.restype = ctypes.c_int64
    kernels.backbone_gemm_plan.argtypes = (
        [ctypes.c_int32] * 3 + [ctypes.c_float] + [ctypes.c_void_p] * 5
        + [ctypes.POINTER(ctypes.c_void_p)])
    kernels.backbone_gemm_plan.restype = ctypes.c_int32
    kernels.backbone_gemm_run.argtypes = [ctypes.c_void_p] * 2
    kernels.backbone_gemm_run.restype = ctypes.c_int32
    kernels.backbone_gemm_destroy.argtypes = [ctypes.c_void_p]
    kernels.backbone_gemm_destroy.restype = None
    kernels.expert_down_workspace.argtypes = [ctypes.c_int32] * 2
    kernels.expert_down_workspace.restype = ctypes.c_int64
    kernels.expert_down_plan.argtypes = (
        [ctypes.c_int32] * 2 + [ctypes.c_void_p] * 6 + [ctypes.POINTER(ctypes.c_void_p)])
    kernels.expert_down_plan.restype = ctypes.c_int32
    kernels.expert_down_run.argtypes = [ctypes.c_void_p] * 2
    kernels.expert_down_run.restype = ctypes.c_int32
    kernels.expert_down_destroy.argtypes = [ctypes.c_void_p]
    kernels.expert_down_destroy.restype = None
    return kernels


def check(status, operation):
    if status >= 1000:
        raise RuntimeError(f"{operation}: CUTLASS status {status - 1000}")
    if status:
        raise RuntimeError(f"{operation}: cudaError {status}")


class _Plan:
    def __init__(self, native, scratch, a, b, output, beta, stream):
        m, k = a.shape
        n = b.shape[1]
        size = native.backbone_gemm_workspace(m, k, n)
        # These backbone GEMMs run serially on one stream. Stream-K resets its
        # barriers after each launch, so plans can share the same scratch role.
        self.workspace = scratch("cutlass_backbone_workspace", (max(size, 1),),
                                 torch.uint8, a.device)
        self.tensors = (a, b, output)
        self.handle = ctypes.c_void_p()
        check(native.backbone_gemm_plan(
            m, k, n, beta, a.data_ptr(), b.data_ptr(), output.data_ptr(),
            self.workspace.data_ptr(), stream, ctypes.byref(self.handle)),
            f"backbone_gemm_plan M={m} K={k} N={n} beta={beta}")
        self.destroy = weakref.finalize(self, native.backbone_gemm_destroy, self.handle)


def make_wrappers(scratch, selected_names=None) -> dict:
    """Build capture-safe BF16 wrappers for contiguous CUDA backbone tensors.

    FFN: x/x_norm (M, 2048), weights (2048, 16384), output (M, 16384).
    Down: x (M, 16384), weight (16384, 2048), output (M, 2048), with C=D.
    The runner warms each pointer set before capture and owns all scratch.
    """
    names = NAMES if selected_names is None else set(selected_names)
    # Declaration builds the op table without a compiler or CUDA context.
    native = pointwise = None
    plans = {}

    def gemm(a, b, output, *, beta, stream):
        nonlocal native
        key = (a.data_ptr(), b.data_ptr(), output.data_ptr(), beta)
        plan = plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("backbone GEMM pointer set was not warmed before capture")
            if native is None:
                native = library()
            plan = _Plan(native, scratch, a, b, output, beta, stream)
            plans[key] = plan
        check(native.backbone_gemm_run(plan.handle, stream),
               f"backbone_gemm_run M={a.shape[0]} K={a.shape[1]} N={b.shape[1]}")

    def llm_backbone_norm_gated_ffn(x, gate_w, up_w, out, x_norm):
        nonlocal pointwise
        if pointwise is None:
            pointwise = fused_backbone.library()
        rows = x.shape[0]
        normed, result = x_norm[:rows], out[:rows]
        gate = scratch("backbone_ffn_gate", result.shape, x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        check(pointwise.backbone_rms_norm(x.data_ptr(), normed.data_ptr(), rows, stream),
               f"backbone_rms_norm M={rows} K=2048")
        gemm(normed, gate_w, gate, beta=0.0, stream=stream)
        gemm(normed, up_w, result, beta=0.0, stream=stream)
        check(pointwise.backbone_gelu_mul(
            gate.data_ptr(), result.data_ptr(), result.numel(), stream),
            f"backbone_gelu_mul elements={result.numel()}")
        return out

    def llm_backbone_ffn_down_residual(x, weight, out):
        gemm(x, weight, out, beta=1.0, stream=torch.cuda.current_stream().cuda_stream)
        return out

    def llm_backbone_norm_gated_ffn_masked(x, gate_w, up_w, out, x_norm, mask):
        return llm_backbone_norm_gated_ffn(x, gate_w, up_w, out, x_norm)

    def llm_backbone_ffn_down_residual_masked(x, weight, out, mask):
        return llm_backbone_ffn_down_residual(x, weight, out)

    wrappers = {
        "llm_backbone_norm_gated_ffn": llm_backbone_norm_gated_ffn,
        "llm_backbone_ffn_down_residual": llm_backbone_ffn_down_residual,
        "llm_backbone_norm_gated_ffn_masked": llm_backbone_norm_gated_ffn_masked,
        "llm_backbone_ffn_down_residual_masked": llm_backbone_ffn_down_residual_masked,
    }
    return {name: wrappers[name] for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)
