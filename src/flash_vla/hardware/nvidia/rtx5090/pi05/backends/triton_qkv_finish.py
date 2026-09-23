"""Pi0.5 expert QKV with one rounded GEMM/factor/bias/RoPE/scatter kernel.

Contiguous BF16 x(M,1024), scale(1024), weight(1024,2560), bias(2560), and
rope(M,256) produce Q(M*8,256), K/V(M,256), and norm_factor(M). K/V are the
caller's suffix views, so their pointers already include the prefix offset.
The runner owns the scaled-activation scratch; no projected tensor is stored.
Warmup loads the existing prepare library and compiles the sole 16x32x32 tile.
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
def _qkv_mm_finish(A, Weight, Factor, Bias, Rope, Q, K, V, M: tl.constexpr):
    tile_n = tl.program_id(1)
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tile_n * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for start in range(0, 1024, 32):
        a = tl.load(A + rows[:, None] * 1024 + start + kk[None, :],
                    mask=rows[:, None] < M, other=0.0)
        weight = tl.load(Weight + (start + kk[:, None]) * 2560 + cols[None, :])
        acc = tl.dot(a, weight, acc)
    # Keep the store/reload rounding of the separate BF16 GEMM before its epilogue.
    projected = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    factor = tl.load(Factor + rows, mask=rows < M, other=0.0).to(tl.float32)
    values = projected * factor[:, None]
    values = values + tl.load(Bias + cols)[None, :].to(tl.float32)
    if tile_n < 72:
        even, odd = tl.split(tl.reshape(values, (16, 16, 2)))
        pairs = tile_n * 16 + tl.arange(0, 16)
        rope_cols = 2 * (pairs % 128)
        cos = tl.load(Rope + rows[:, None] * 256 + rope_cols[None, :],
                      mask=rows[:, None] < M, other=0.0).to(tl.float32)
        sin = tl.load(Rope + rows[:, None] * 256 + rope_cols[None, :] + 1,
                      mask=rows[:, None] < M, other=0.0).to(tl.float32)
        rotated_even = even * cos - odd * sin
        rotated_odd = odd * cos + even * sin
        values = tl.reshape(tl.join(rotated_even, rotated_odd), (16, 32))
    result = values.to(tl.bfloat16, fp_downcast_rounding="rtne")
    if tile_n < 64:
        tl.store(Q + rows[:, None] * 2048 + cols[None, :], result,
                 mask=rows[:, None] < M)
    elif tile_n < 72:
        # K/V are already suffix pointers from the caller, with prefix offset applied.
        tl.store(K + rows[:, None] * 256 + cols[None, :] - 2048, result,
                 mask=rows[:, None] < M)
    else:
        tl.store(V + rows[:, None] * 256 + cols[None, :] - 2304, result,
                 mask=rows[:, None] < M)


def action_expert_norm_qkv_rope(
    x, scale, weight_qkv, bias, rope, Q, K, V, norm_factor, *, scratch,
):
    rows = x.shape[0]
    scaled = scratch("pi05_qkv_scaled", (rows, 1024), x.dtype, x.device)
    lib = fused_qkv.library()
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.pi05_qkv_prepare(
        x.data_ptr(), scale.data_ptr(), scaled.data_ptr(), norm_factor.data_ptr(), rows, stream)
    if rc:
        raise RuntimeError(f"pi05_qkv_prepare rows={rows}, threads=256: CUDA error {rc}")
    _qkv_mm_finish[(triton.cdiv(rows, 16), 80)](
        scaled, weight_qkv, norm_factor, bias, rope, Q, K, V, rows,
        num_warps=4, num_stages=3, enable_fp_fusion=False, enable_reflect_ftz=False)


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind scratch ownership without native loading or CUDA initialization."""
    names = NAMES if selected_names is None else selected_names
    return {name: partial(action_expert_norm_qkv_rope, scratch=scratch) for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
