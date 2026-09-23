"""Pi0.5 expert projections with a BF16-rounded CUTLASS gated residual epilogue."""
from __future__ import annotations

import ctypes
import weakref

import torch

from flash_vla.runtime.registry import Backend

from . import cutlass_backbone
from .cutlass_backbone import check

NAMES = ("action_expert_ffn_down_residual", "action_expert_out_proj_residual")


class _Plan:
    def __init__(self, native, scratch, role, x, weight, gate, out, stream):
        m, k = x.shape
        size = native.expert_down_workspace(m, k)
        self.workspace = scratch(role, (max(size, 1),), torch.uint8, x.device)
        self.tensors = (x, weight, gate, out)
        self.handle = ctypes.c_void_p()
        check(native.expert_down_plan(
            m, k, x.data_ptr(), weight.data_ptr(), gate.data_ptr(), out.data_ptr(),
            self.workspace.data_ptr(), stream, ctypes.byref(self.handle)),
            f"expert_down_plan M={m} K={k} N=1024")
        self.destroy = weakref.finalize(self, native.expert_down_destroy, self.handle)


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind contiguous CUDA BF16 x(M,K), weight(K,1024), gate(1024,).

    K is 2048 for attention out-projection and 4096 for FFN down.
    out(M,1024) is read and written in place, with a BF16-rounded GEMM result
    before the separate FP32 multiply/add. Each closure owns native plans and
    source references. Warmup must visit all pointer sets before graph capture;
    its Stream-K scratch is shared serially and no projection buffer is needed.
    Op-table construction needs no compiler or CUDA context.
    """
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"expert residual backend does not implement {sorted(unknown)}")
    native = None
    plans = {}
    role = f"pi05_expert_residual_streamk_{id(plans)}"

    def projection_residual(x, weight, gate, out):
        nonlocal native
        m, k = x.shape
        key = (m, k, x.data_ptr(), weight.data_ptr(), gate.data_ptr(), out.data_ptr())
        plan = plans.get(key)
        stream = torch.cuda.current_stream().cuda_stream
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("expert residual pointer set was not warmed before capture")
            if native is None:
                native = cutlass_backbone.library()
            plan = _Plan(native, scratch, role, x, weight, gate, out, stream)
            plans[key] = plan
        check(native.expert_down_run(plan.handle, stream),
               f"expert_down_run M={m} K={k} N=1024")
        return out

    return {name: projection_residual for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
