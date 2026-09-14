"""Pi0.5 expert attention with the unchanged score/softmax path and one PV launch.

The fixed Target has BF16 Q/out(400,256), K/V(1018,256), and mask(1018).
Out may alias Q. FP32 logits and materialized BF16 probabilities retain the
existing runner-owned scratch roles; only the final P@V implementation changes.
The 16x32x64 PV tile masks K's tail, accumulates FP32 and stores BF16.
Warmup compiles before graph capture; wrapper declaration does not use CUDA.
"""
from __future__ import annotations

from functools import partial

import torch
import triton
import triton.language as tl

from .fused_attention import NAMES, _library


@triton.jit
def _pv(P, V, Out):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 64)
    acc = tl.zeros((16, 32), tl.float32)
    for start in range(0, 1018, 64):
        k = start + kk
        p = tl.load(P + rows[:, None] * 1018 + k[None, :],
                    mask=k[None, :] < 1018, other=0.0)
        v = tl.load(V + k[:, None] * 256 + cols[None, :],
                    mask=k[:, None] < 1018, other=0.0)
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
    result = acc.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Out + rows[:, None] * 256 + cols[None, :], result)


def action_expert_attention(Q, K, V, mask, out, prefix_len=None, *, scratch):
    queries, head_dim = Q.shape
    keys = K.shape[0]
    logits = scratch("pi05_attention_logits", (queries, keys), torch.float32, Q.device)
    probabilities = scratch("pi05_attention_probabilities", (queries, keys), Q.dtype, Q.device)
    lib = _library()
    torch.mm(Q, K.T, out_dtype=torch.float32, out=logits)
    rc = lib.pi05_attention_softmax_launch(
        logits.data_ptr(), mask.data_ptr(), probabilities.data_ptr(), queries, keys,
        float(head_dim ** -0.5), torch.cuda.current_stream(Q.device).cuda_stream)
    if rc:
        raise RuntimeError(
            f"pi05_attention_softmax queries={queries}, keys={keys}, threads=256: CUDA error {rc}")
    _pv[(25, 8)](
        probabilities, V, out, num_warps=4, num_stages=3,
        enable_fp_fusion=False, enable_reflect_ftz=False)
    return out


def make_wrappers(scratch, selected_names=None) -> dict:
    names = NAMES if selected_names is None else selected_names
    wrappers = {"action_expert_attention": action_expert_attention}
    return {name: partial(wrappers[name], scratch=scratch) for name in names}


__all__ = ["NAMES", "make_wrappers"]
