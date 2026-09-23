"""Pi0.5 backbone GEMMs with device-selected M896/M968 and fixed graph addresses.

The mask is an explicit op argument. Both static plans are warmed and captured;
their native entries select the active bucket from mask[896] on every replay.
The bucketed tensors remain BF16 M968. Other row counts use the existing
full-row CUTLASS plan without bucket masking. This module only loads native libraries
on the first real invocation, so declaration needs no CUDA context or compiler.
"""
from __future__ import annotations

import ctypes
import weakref

import torch

from flash_vla.runtime.ops import OpSpec, dual_gemm, gemm

from . import cutlass_backbone, fused_backbone

MASKED_CALL_SITES = {
    "llm_backbone_norm_gated_ffn": "llm_backbone_norm_gated_ffn_masked",
    "llm_backbone_ffn_down_residual": "llm_backbone_ffn_down_residual_masked",
    "llm_backbone_out_proj_residual": "llm_backbone_out_proj_residual_masked",
}
OPS = (
    OpSpec("llm_backbone_norm_gated_ffn_masked",
           ("x", "gate_w", "up_w", "out", "x_norm", "mask"),
           outputs=("out", "x_norm"), weights=("gate_w", "up_w"), aux=("x_norm",),
           flops=dual_gemm("x", "gate_w")),
    OpSpec("llm_backbone_ffn_down_residual_masked",
           ("x", "weight", "out", "mask"), outputs=("out",), inout=("out",),
           weights=("weight",), flops=gemm("x", "weight")),
    OpSpec("llm_backbone_out_proj_residual_masked",
           ("x", "weight", "out", "mask"), outputs=("out",), inout=("out",),
           weights=("weight",), flops=gemm("x", "weight")),
)
NAMES = frozenset(MASKED_CALL_SITES.values())
# Adjacent M128 tile boundaries for this Target's 968 physical prefix rows.
_BUCKET_ROWS = (896, 968)


def _library():
    library = cutlass_backbone._library()
    library.backbone_bucket_workspace.argtypes = [ctypes.c_int32] * 3
    library.backbone_bucket_workspace.restype = ctypes.c_int64
    library.backbone_bucket_plan.argtypes = (
        [ctypes.c_int32] * 3 + [ctypes.c_float] + [ctypes.c_void_p] * 5
        + [ctypes.POINTER(ctypes.c_void_p)])
    library.backbone_bucket_plan.restype = ctypes.c_int32
    library.backbone_bucket_run.argtypes = [ctypes.c_void_p] * 3
    library.backbone_bucket_run.restype = ctypes.c_int32
    library.backbone_bucket_destroy.argtypes = [ctypes.c_void_p]
    library.backbone_bucket_destroy.restype = None
    return library


class _Plan:
    """Own both static native plans and scratch workspaces for one pointer set."""

    def __init__(self, library, scratch, a, b, output, beta, stream):
        k, n = a.shape[1], b.shape[1]
        self.tensors = (a, b, output)
        self.handles, self.workspaces, self.destroy = [], [], []
        for m in _BUCKET_ROWS:
            size = library.backbone_bucket_workspace(m, k, n)
            if size < 0:
                cutlass_backbone._check(-size, "backbone_bucket_workspace")
            # Calls are serial; separate roles keep each bucket's barriers distinct.
            workspace = scratch(f"bucketed_backbone_workspace_{m}", (max(size, 1),),
                                torch.uint8, a.device)
            handle = ctypes.c_void_p()
            cutlass_backbone._check(library.backbone_bucket_plan(
                m, k, n, beta, a.data_ptr(), b.data_ptr(), output.data_ptr(),
                workspace.data_ptr(), stream, ctypes.byref(handle)),
                f"backbone_bucket_plan M={m} K={k} N={n} beta={beta}")
            self.handles.append(handle)
            self.workspaces.append(workspace)
            self.destroy.append(weakref.finalize(self, library.backbone_bucket_destroy, handle))


def make_wrappers(scratch, selected_names=None) -> dict:
    """Build M968 BF16 backbone wrappers with a BF16 prefix mask of length 968.

    Up writes out (968,16384) and x_norm (968,2048); down/out-projection add to
    out (968,2048). The explicit mask stays at one address but may change
    between graph replays. Both plans and all workspace are created in warmup.
    """
    names = NAMES if selected_names is None else set(selected_names)
    library = pointwise = None
    plans = {}

    def run_gemm(a, b, output, mask, *, beta, stream):
        nonlocal library
        key = (a.data_ptr(), b.data_ptr(), output.data_ptr(), beta)
        plan = plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("bucketed backbone pointer set was not warmed before capture")
            if library is None:
                library = _library()
            plan_type = _Plan if a.shape[0] == 968 else cutlass_backbone._Plan
            plan = plan_type(library, scratch, a, b, output, beta, stream)
            plans[key] = plan
        if a.shape[0] == 968:
            for handle in plan.handles:
                cutlass_backbone._check(
                    library.backbone_bucket_run(handle, mask.data_ptr(), stream),
                    "backbone_bucket_run")
        else:
            cutlass_backbone._check(library.backbone_gemm_run(plan.handle, stream),
                                    "backbone_gemm_run")

    def llm_backbone_norm_gated_ffn_masked(x, gate_w, up_w, out, x_norm, mask):
        nonlocal pointwise
        if pointwise is None:
            pointwise = fused_backbone._library()
            pointwise.backbone_masked_gelu_mul.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            pointwise.backbone_masked_gelu_mul.restype = ctypes.c_int32
        rows = x.shape[0]
        normed, result = x_norm[:rows], out[:rows]
        gate = scratch("backbone_ffn_gate", result.shape, x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        cutlass_backbone._check(pointwise.backbone_rms_norm(
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
        cutlass_backbone._check(status, "backbone_gelu_mul")
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
