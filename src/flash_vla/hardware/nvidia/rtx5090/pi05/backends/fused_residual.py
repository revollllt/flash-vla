"""Pi0.5 RTX 5090 expert projections with a fused gated residual update."""
from __future__ import annotations

import torch

from flash_vla.runtime.registry import Backend

from . import fused_ffn
from .fused_ffn import check

NAMES = ("action_expert_out_proj_residual", "action_expert_ffn_down_residual")


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind bf16 CUDA MxK @ Kx1024, then out += projection * gate in fp32.

    K is 2048 for the attention projection and 4096 for FFN down; gate is
    (1024,) and out is contiguous (M, 1024), read and written in place. The
    bf16 GEMM output lives in scratch. Warmup builds CUDA and allocates this
    buffer before scratch freezes and the runner captures its graph.
    """
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"fused residual backend does not implement {sorted(unknown)}")
    workspaces = {}
    role = f"pi05_fused_residual_{id(workspaces)}"

    def projection_residual(x, weight, gate, out):
        rows = x.shape[0]
        key = (rows, x.dtype, x.device)
        if key not in workspaces:
            workspaces[key] = scratch(role, (rows, 1024), x.dtype, x.device)
        projected = workspaces[key]
        torch.mm(x, weight, out=projected)
        check(fused_ffn.library().gated_residual_launch(
            projected.data_ptr(), gate.data_ptr(), out.data_ptr(), rows,
            torch.cuda.current_stream().cuda_stream), "gated_residual", rows)
        return out

    return {name: projection_residual for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
