"""Pi0.5 backbone GEMMs with a device-selected short or full row bucket.

The buckets are `row_buckets.bucket_rows` of the runner's shape. Both static
plans are warmed and captured, and their native entries select the active
bucket from the mask at the short bucket's row on every replay. A prefix with
no tile to skip (`row_buckets.has_short_bucket`) is not supported; a Target
routes it to `cutlass_backbone`'s full-row plan. This module only loads native
libraries on the first real invocation, so declaration needs no CUDA context or
compiler.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import weakref

import torch

from flash_vla.models.pi05.ops import MASKED_CALL_SITES
from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch

from . import cutlass_backbone, fused_backbone
from .row_buckets import bucket_rows, has_short_bucket

NAMES = frozenset(MASKED_CALL_SITES.values())


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The CUTLASS library with this module's entry points declared."""
    kernels = cutlass_backbone.library()
    kernels.backbone_bucket_workspace.argtypes = [ctypes.c_int32] * 3
    kernels.backbone_bucket_workspace.restype = ctypes.c_int64
    kernels.backbone_bucket_plan.argtypes = (
        [ctypes.c_int32] * 4 + [ctypes.c_float] + [ctypes.c_void_p] * 5
        + [ctypes.POINTER(ctypes.c_void_p)])
    kernels.backbone_bucket_plan.restype = ctypes.c_int32
    kernels.backbone_bucket_run.argtypes = [ctypes.c_void_p] * 3
    kernels.backbone_bucket_run.restype = ctypes.c_int32
    kernels.backbone_bucket_destroy.argtypes = [ctypes.c_void_p]
    kernels.backbone_bucket_destroy.restype = None
    return kernels


class BucketPlan:
    """Both static native plans of output = a @ b + beta * output, one per row
    bucket, and their scratch workspaces, bound to one pointer set. Planning
    allocates, so it happens in warmup, never during capture."""

    def __init__(self, scratch: Scratch, a: torch.Tensor, b: torch.Tensor,
                 output: torch.Tensor, beta: float, stream: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("bucketed backbone pointer set was not warmed before capture")
        native = library()
        k, n = a.shape[1], b.shape[1]
        buckets = bucket_rows(scratch.shape)
        self.tensors = (a, b, output)
        self.handles: list[ctypes.c_void_p] = []
        self.workspaces: list[torch.Tensor] = []
        self.destroy: list[weakref.finalize] = []
        for m in buckets:
            size = native.backbone_bucket_workspace(m, k, n)
            if size < 0:
                cutlass_backbone.check(-size, "backbone_bucket_workspace")
            # Calls are serial; separate roles keep each bucket's barriers distinct.
            workspace = scratch(f"bucketed_backbone_workspace_{m}", (max(size, 1),),
                                torch.uint8, a.device)
            handle = ctypes.c_void_p()
            cutlass_backbone.check(native.backbone_bucket_plan(
                m, k, n, buckets[0], beta, a.data_ptr(), b.data_ptr(), output.data_ptr(),
                workspace.data_ptr(), stream, ctypes.byref(handle)),
                f"backbone_bucket_plan M={m} K={k} N={n} beta={beta}")
            self.handles.append(handle)
            self.workspaces.append(workspace)
            self.destroy.append(weakref.finalize(self, native.backbone_bucket_destroy, handle))


def make_wrappers(scratch: Scratch, selected_names: frozenset[str] | None = None
                  ) -> dict[str, Wrapper]:
    """Build BF16 backbone wrappers over the prefix rows with a BF16 prefix mask.

    Up writes out (prefix,16384) and x_norm (prefix,2048); down/out-projection
    add to out (prefix,2048). The explicit mask stays at one address but may
    change between graph replays. Both plans and all workspace are created in
    warmup.
    """
    names = NAMES if selected_names is None else selected_names
    short_rows = bucket_rows(scratch.shape)[0]
    plans: dict[tuple[int, int, int, float], BucketPlan] = {}

    def run_gemm(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor, mask: torch.Tensor,
                 *, beta: float, stream: int) -> None:
        key = (a.data_ptr(), b.data_ptr(), output.data_ptr(), beta)
        plan = plans[key] if key in plans else plans.setdefault(
            key, BucketPlan(scratch, a, b, output, beta, stream))
        for handle in plan.handles:
            cutlass_backbone.check(
                library().backbone_bucket_run(handle, mask.data_ptr(), stream),
                "backbone_bucket_run")

    def llm_backbone_norm_gated_ffn_masked(x: torch.Tensor, gate_w: torch.Tensor,
                                           up_w: torch.Tensor, out: torch.Tensor,
                                           x_norm: torch.Tensor, mask: torch.Tensor
                                           ) -> torch.Tensor:
        pointwise = fused_backbone.library()
        rows = x.shape[0]
        normed, result = x_norm[:rows], out[:rows]
        gate = scratch("backbone_ffn_gate", result.shape, x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        cutlass_backbone.check(pointwise.backbone_rms_norm(
            x.data_ptr(), normed.data_ptr(), rows, stream), "backbone_rms_norm")
        run_gemm(normed, gate_w, gate, mask, beta=0.0, stream=stream)
        run_gemm(normed, up_w, result, mask, beta=0.0, stream=stream)
        # The short bucket zeroes the tail before any stale gate/up load.
        cutlass_backbone.check(pointwise.backbone_masked_gelu_mul(
            gate.data_ptr(), result.data_ptr(), result.numel(), mask.data_ptr(), short_rows,
            stream), "backbone_masked_gelu_mul")
        return out

    def llm_backbone_ffn_down_residual_masked(x: torch.Tensor, weight: torch.Tensor,
                                              out: torch.Tensor, mask: torch.Tensor
                                              ) -> torch.Tensor:
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
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers, supports=has_short_bucket)
