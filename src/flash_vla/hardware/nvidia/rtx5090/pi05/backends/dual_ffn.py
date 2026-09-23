"""Pi0.5 expert FFN: unchanged prepare followed by one rounded dual-dot kernel.

The measured 16x64x32 tile keeps both FP32 projections in registers, rounds
both to BF16 before FP32 bias/GELU/product, and stores only the final BF16
output. Packed weights and prepared activations use per-wrapper scratch.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from flash_vla.runtime.registry import Backend

from . import fused_ffn
from .fused_ffn import NAMES, check


@triton.jit
def _dual_dot(A, Packed, GateBias, UpBias, Out, M: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    kk = tl.arange(0, 32)
    gate_acc = tl.zeros((16, 64), tl.float32)
    up_acc = tl.zeros((16, 64), tl.float32)
    for start in range(0, 1024, 32):
        a = tl.load(A + rows[:, None] * 1024 + (start + kk[None, :]),
                    mask=rows[:, None] < M, other=0.0)
        offsets = (start + kk[:, None]) * 8192 + cols[None, :]
        gate_w = tl.load(Packed + offsets)
        up_w = tl.load(Packed + offsets + 4096)
        gate_acc = tl.dot(a, gate_w, gate_acc)
        up_acc = tl.dot(a, up_w, up_acc)
    # Preserve each BF16 GEMM store before either FP32 bias addition.
    gate = gate_acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    up = up_acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    gate = gate + tl.load(GateBias + cols)[None, :].to(tl.float32)
    up = up + tl.load(UpBias + cols)[None, :].to(tl.float32)
    cube = gate * gate * gate
    gelu = 0.5 * gate * (1.0 + libdevice.tanh(
        0.7978845608028654 * (gate + 0.044715 * cube)))
    result = (gelu * up).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Out + rows[:, None] * 4096 + cols[None, :], result,
             mask=rows[:, None] < M)


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind CUDA BF16 x(M,1024), gate/up weights(1024,4096), biases(4096).

    Prepare writes norm_factor(M) and the same rounded/scaled activation as
    packed_ffn. The suffix writes out(M,4096). Each closure retains the source
    weights and packed tensor; warmup packs all pairs and compiles the sole
    tile before graph capture. Op-table declaration does not initialize CUDA.
    """
    names = set(NAMES) if selected_names is None else set(selected_names)
    weights = {}
    workspaces = {}
    role = f"pi05_dual_ffn_{id(weights)}"

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
            workspaces[key] = scratch(role + "_a", (rows, 1024), x.dtype, x.device)
        a = workspaces[key]
        lib = fused_ffn.library()
        stream = torch.cuda.current_stream().cuda_stream
        check(lib.ada_rms_launch(x.data_ptr(), scale.data_ptr(), a.data_ptr(),
                                 norm_factor.data_ptr(), rows, stream), "ada_rms", rows)
        _dual_dot[(triton.cdiv(rows, 16), 64)](
            a, packed, gate_b, up_b, out, rows, num_warps=4, num_stages=3,
            enable_fp_fusion=False, enable_reflect_ftz=False)
        return out

    return {name: action_expert_norm_gated_ffn for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
