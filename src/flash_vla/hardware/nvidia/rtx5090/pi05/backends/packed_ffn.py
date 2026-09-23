"""One-GEMM variant of the Pi0.5 RTX 5090 fused expert FFN."""
from __future__ import annotations

import torch

from flash_vla.runtime.registry import Backend

from . import fused_ffn
from .fused_ffn import NAMES, check


def make_wrappers(scratch, selected_names=None) -> dict:
    """Pack immutable gate/up weights during warmup and bind scratch buffers.

    Shapes, bf16 roundings, mutations and capture requirements match fused_ffn.
    Each closure retains both source tensors and the packed buffer for every
    distinct weight pair. Packing uses scratch, so warmup must visit all pairs
    before it freezes; 18 layers add 288 MiB of packed weights.
    """
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"packed FFN backend does not implement {sorted(unknown)}")
    weights = {}
    workspaces = {}
    role = f"pi05_packed_ffn_{id(weights)}"

    def action_expert_norm_gated_ffn(x, scale, gate_w, up_w, gate_b, up_b,
                                     out, norm_factor):
        pair = (gate_w.data_ptr(), up_w.data_ptr())
        if pair not in weights:
            packed = scratch(role + f"_weight_{len(weights)}", (1024, 8192),
                             gate_w.dtype, gate_w.device)
            packed[:, :4096].copy_(gate_w)
            packed[:, 4096:].copy_(up_w)
            weights[pair] = (gate_w, up_w, packed)
        packed = weights[pair][2]
        rows = x.shape[0]
        key = (rows, x.dtype, x.device)
        if key not in workspaces:
            workspaces[key] = (
                scratch(role + "_a", (rows, 1024), x.dtype, x.device),
                scratch(role + "_both", (rows, 8192), x.dtype, x.device),
            )
        a, both = workspaces[key]
        lib = fused_ffn.library()
        stream = torch.cuda.current_stream().cuda_stream
        check(lib.ada_rms_launch(x.data_ptr(), scale.data_ptr(), a.data_ptr(),
                                 norm_factor.data_ptr(), rows, stream), "ada_rms", rows)
        torch.mm(a, packed, out=both)
        check(lib.packed_gated_activation_launch(
            both.data_ptr(), both.data_ptr() + 4096 * both.element_size(),
            gate_b.data_ptr(), up_b.data_ptr(), out.data_ptr(), rows, stream),
            "packed_gated_activation", rows)
        return out

    return {name: action_expert_norm_gated_ffn for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
