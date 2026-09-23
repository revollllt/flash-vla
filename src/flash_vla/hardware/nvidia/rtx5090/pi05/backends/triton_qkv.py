"""Pi0.5 expert QKV with the measured 16x32x32 BF16 Triton GEMM.

For contiguous BF16 x(M,1024), scale(1024), weight(1024,2560), bias(2560),
and rope(M,256), write Q(M*8,256), K/V(M,256), and norm_factor(M).
The existing prepare and finish retain their factor/bias/RoPE/scatter order.
Both temporary tensors belong to the runner's scratch allocator. Warmup loads
the native pointwise library and compiles the sole tile before graph capture.
"""
from __future__ import annotations

from functools import partial

import torch
import triton
import triton.language as tl

from flash_vla.runtime.registry import Backend

from . import fused_qkv

NAMES = frozenset({"action_expert_norm_qkv_rope"})


@triton.jit
def _qkv_mm(A, Weight, Projected, M: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for start in range(0, 1024, 32):
        a = tl.load(A + rows[:, None] * 1024 + start + kk[None, :],
                    mask=rows[:, None] < M, other=0.0)
        weight = tl.load(Weight + (start + kk[:, None]) * 2560 + cols[None, :])
        acc = tl.dot(a, weight, acc)
    # Finish must consume the rounded BF16 projection.
    projected = acc.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Projected + rows[:, None] * 2560 + cols[None, :], projected,
             mask=rows[:, None] < M)


def action_expert_norm_qkv_rope(
    x, scale, weight_qkv, bias, rope, Q, K, V, norm_factor, *, scratch,
):
    rows = x.shape[0]
    scaled = scratch("pi05_qkv_scaled", (rows, 1024), x.dtype, x.device)
    projected = scratch("pi05_qkv_projected", (rows, 2560), x.dtype, x.device)
    lib = fused_qkv.library()
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.pi05_qkv_prepare(
        x.data_ptr(), scale.data_ptr(), scaled.data_ptr(), norm_factor.data_ptr(), rows, stream)
    if rc:
        raise RuntimeError(f"pi05_qkv_prepare rows={rows}, threads=256: CUDA error {rc}")
    _qkv_mm[(triton.cdiv(rows, 16), 80)](
        scaled, weight_qkv, projected, rows, num_warps=4, num_stages=3,
        enable_fp_fusion=False, enable_reflect_ftz=False)
    rc = lib.pi05_qkv_finish(
        projected.data_ptr(), norm_factor.data_ptr(), bias.data_ptr(), rope.data_ptr(),
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), rows, stream)
    if rc:
        raise RuntimeError(f"pi05_qkv_finish blocks={rows * 5}, threads=256: CUDA error {rc}")


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind scratch ownership without loading native code or initializing CUDA."""
    names = NAMES if selected_names is None else selected_names
    return {name: partial(action_expert_norm_qkv_rope, scratch=scratch) for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
