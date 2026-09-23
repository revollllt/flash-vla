"""Pi0.5 backbone GEMMs with device-selected M896/M968 and fixed graph addresses.

The mask is an explicit op argument. Both static plans are warmed and captured;
their native entries select the active bucket from mask[896] on every replay.
The bucketed tensors remain BF16 M968. Other row counts use the existing
full-row CUTLASS plan without bucket masking. This module only loads native libraries
on the first real invocation, so declaration needs no CUDA context or compiler.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import weakref

import torch

from flash_vla.models.pi05.ops import MASKED_CALL_SITES
from flash_vla.runtime.registry import Backend

from . import cutlass_backbone, fused_backbone

NAMES = frozenset(MASKED_CALL_SITES.values())
# Adjacent M128 tile boundaries for this Target's 968 physical prefix rows.
_BUCKET_ROWS = (896, 968)


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The CUTLASS library with this module's entry points declared."""
    kernels = cutlass_backbone.library()
    kernels.backbone_bucket_workspace.argtypes = [ctypes.c_int32] * 3
    kernels.backbone_bucket_workspace.restype = ctypes.c_int64
    kernels.backbone_bucket_plan.argtypes = (
        [ctypes.c_int32] * 3 + [ctypes.c_float] + [ctypes.c_void_p] * 5
        + [ctypes.POINTER(ctypes.c_void_p)])
    kernels.backbone_bucket_plan.restype = ctypes.c_int32
    kernels.backbone_bucket_run.argtypes = [ctypes.c_void_p] * 3
    kernels.backbone_bucket_run.restype = ctypes.c_int32
    kernels.backbone_bucket_destroy.argtypes = [ctypes.c_void_p]
    kernels.backbone_bucket_destroy.restype = None
    return kernels


class _Plan:
    """Own both static native plans and scratch workspaces for one pointer set."""

    def __init__(self, native, scratch, a, b, output, beta, stream):
        k, n = a.shape[1], b.shape[1]
        self.tensors = (a, b, output)
        self.handles, self.workspaces, self.destroy = [], [], []
        for m in _BUCKET_ROWS:
            size = native.backbone_bucket_workspace(m, k, n)
            if size < 0:
                cutlass_backbone.check(-size, "backbone_bucket_workspace")
            # Calls are serial; separate roles keep each bucket's barriers distinct.
            workspace = scratch(f"bucketed_backbone_workspace_{m}", (max(size, 1),),
                                torch.uint8, a.device)
            handle = ctypes.c_void_p()
            cutlass_backbone.check(native.backbone_bucket_plan(
                m, k, n, beta, a.data_ptr(), b.data_ptr(), output.data_ptr(),
                workspace.data_ptr(), stream, ctypes.byref(handle)),
                f"backbone_bucket_plan M={m} K={k} N={n} beta={beta}")
            self.handles.append(handle)
            self.workspaces.append(workspace)
            self.destroy.append(weakref.finalize(self, native.backbone_bucket_destroy, handle))


def make_wrappers(scratch, selected_names=None) -> dict:
    """Build M968 BF16 backbone wrappers with a BF16 prefix mask of length 968.

    Up writes out (968,16384) and x_norm (968,2048); down/out-projection add to
    out (968,2048). The explicit mask stays at one address but may change
    between graph replays. Both plans and all workspace are created in warmup.
    """
    names = NAMES if selected_names is None else set(selected_names)
    native = pointwise = None
    plans = {}

    def run_gemm(a, b, output, mask, *, beta, stream):
        nonlocal native
        key = (a.data_ptr(), b.data_ptr(), output.data_ptr(), beta)
        plan = plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("bucketed backbone pointer set was not warmed before capture")
            if native is None:
                native = library()
            plan_type = _Plan if a.shape[0] == 968 else cutlass_backbone._Plan
            plan = plan_type(native, scratch, a, b, output, beta, stream)
            plans[key] = plan
        if a.shape[0] == 968:
            for handle in plan.handles:
                cutlass_backbone.check(
                    native.backbone_bucket_run(handle, mask.data_ptr(), stream),
                    "backbone_bucket_run")
        else:
            cutlass_backbone.check(native.backbone_gemm_run(plan.handle, stream),
                                    "backbone_gemm_run")

    def llm_backbone_norm_gated_ffn_masked(x, gate_w, up_w, out, x_norm, mask):
        nonlocal pointwise
        if pointwise is None:
            pointwise = fused_backbone.library()
            pointwise.backbone_masked_gelu_mul.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            pointwise.backbone_masked_gelu_mul.restype = ctypes.c_int32
        rows = x.shape[0]
        normed, result = x_norm[:rows], out[:rows]
        gate = scratch("backbone_ffn_gate", result.shape, x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        cutlass_backbone.check(pointwise.backbone_rms_norm(
            x.data_ptr(), normed.data_ptr(), rows, stream), "backbone_rms_norm")
        run_gemm(normed, gate_w, gate, mask, beta=0.0, stream=stream)
        run_gemm(normed, up_w, result, mask, beta=0.0, stream=stream)
        # The short bucket zeroes the tail before any stale gate/up load.
        if rows == 968:
            status = pointwise.backbone_masked_gelu_mul(
                gate.data_ptr(), result.data_ptr(), result.numel(), mask.data_ptr(), stream)
        else:
            status = pointwise.backbone_gelu_mul(
                gate.data_ptr(), result.data_ptr(), result.numel(), stream)
        cutlass_backbone.check(status, "backbone_gelu_mul")
        return out

    def llm_backbone_ffn_down_residual_masked(x, weight, out, mask):
        run_gemm(x, weight, out, mask, beta=1.0,
                 stream=torch.cuda.current_stream().cuda_stream)
        return out

    wrappers = {
        "llm_backbone_norm_gated_ffn_masked": llm_backbone_norm_gated_ffn_masked,
        "llm_backbone_ffn_down_residual_masked": llm_backbone_ffn_down_residual_masked,
        "llm_backbone_out_proj_residual_masked": llm_backbone_ffn_down_residual_masked,
    }
    return {name: wrappers[name] for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)
