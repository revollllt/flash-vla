"""Pi0.5 expert FFN down with a BF16-rounded CUTLASS gated residual epilogue."""
from __future__ import annotations

import ctypes
import weakref

import torch

from .cutlass_backbone import _check, _library

NAMES = ("action_expert_ffn_down_residual",)


class _Plan:
    def __init__(self, library, scratch, role, x, weight, gate, out, stream):
        size = library.expert_down_workspace()
        self.workspace = scratch(role, (max(size, 1),), torch.uint8, x.device)
        self.tensors = (x, weight, gate, out)
        self.handle = ctypes.c_void_p()
        _check(library.expert_down_plan(
            x.data_ptr(), weight.data_ptr(), gate.data_ptr(), out.data_ptr(),
            self.workspace.data_ptr(), stream, ctypes.byref(self.handle)),
            "expert_down_plan M=50 K=4096 N=1024")
        self.destroy = weakref.finalize(self, library.expert_down_destroy, self.handle)


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind contiguous CUDA BF16 x(50,4096), weight(4096,1024), gate(1024,).

    out(50,1024) is read and written in place, with a BF16-rounded GEMM result
    before the separate FP32 multiply/add. Each closure owns native plans and
    source references. Warmup must visit all pointer sets before graph capture;
    its Stream-K scratch is shared serially and no projection buffer is needed.
    Op-table construction needs no compiler or CUDA context.
    """
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"expert residual backend does not implement {sorted(unknown)}")
    library = None
    plans = {}
    role = f"pi05_expert_down_streamk_{id(plans)}"

    def action_expert_ffn_down_residual(x, weight, gate, out):
        nonlocal library
        key = (x.data_ptr(), weight.data_ptr(), gate.data_ptr(), out.data_ptr())
        plan = plans.get(key)
        stream = torch.cuda.current_stream().cuda_stream
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("expert down pointer set was not warmed before capture")
            if library is None:
                library = _library()
            plan = _Plan(library, scratch, role, x, weight, gate, out, stream)
            plans[key] = plan
        _check(library.expert_down_run(plan.handle, stream),
               "expert_down_run M=50 K=4096 N=1024")
        return out

    return {name: action_expert_ffn_down_residual for name in names}


__all__ = ["NAMES", "make_wrappers"]
