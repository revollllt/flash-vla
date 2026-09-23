"""Pi0.5 expert attention with tiled QK and the original softmax/PV path.

BF16 Q/out(queries,256), K/V(keys,256), and additive mask(keys) use the existing
runner-owned FP32 logits and BF16 probability scratch. Out may alias Q.
Only QK changes: one 32x32x64 tile writes the same FP32 score layout.
Warmup compiles before graph capture; wrapper declaration does not use CUDA.
"""
from __future__ import annotations

from functools import partial

import torch
import triton
import triton.language as tl

from flash_vla.runtime.registry import Backend

from . import fused_attention
from .fused_attention import NAMES


@triton.jit
def _qk(Q, K, Logits, QUERIES: tl.constexpr, KEYS: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 64)
    acc = tl.zeros((32, 32), tl.float32)
    for start in range(0, 256, 64):
        q = tl.load(Q + rows[:, None] * 256 + start + kk[None, :],
                    mask=rows[:, None] < QUERIES, other=0.0)
        k = tl.load(K + cols[None, :] * 256 + start + kk[:, None],
                    mask=cols[None, :] < KEYS, other=0.0)
        acc = tl.dot(q, k, acc, out_dtype=tl.float32)
    tl.store(Logits + rows[:, None] * KEYS + cols[None, :], acc,
             mask=(rows[:, None] < QUERIES) & (cols[None, :] < KEYS))


def action_expert_attention(Q, K, V, mask, out, prefix_len=None, *, scratch):
    queries, head_dim = Q.shape
    keys = K.shape[0]
    logits = scratch("pi05_attention_logits", (queries, keys), torch.float32, Q.device)
    probabilities = scratch("pi05_attention_probabilities", (queries, keys), Q.dtype, Q.device)
    lib = fused_attention.library()
    _qk[(triton.cdiv(queries, 32), triton.cdiv(keys, 32))](
        Q, K, logits, queries, keys, num_warps=4, num_stages=3,
        enable_fp_fusion=False, enable_reflect_ftz=False)
    rc = lib.pi05_attention_softmax_launch(
        logits.data_ptr(), mask.data_ptr(), probabilities.data_ptr(), queries, keys,
        float(head_dim ** -0.5), torch.cuda.current_stream(Q.device).cuda_stream)
    if rc:
        raise RuntimeError(
            f"pi05_attention_softmax queries={queries}, keys={keys}, threads=256: CUDA error {rc}")
    torch.mm(probabilities, V, out=out)
    return out


def make_wrappers(scratch, selected_names=None) -> dict:
    names = NAMES if selected_names is None else selected_names
    wrappers = {"action_expert_attention": action_expert_attention}
    return {name: partial(wrappers[name], scratch=scratch) for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
