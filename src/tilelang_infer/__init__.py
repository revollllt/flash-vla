"""A TileLang implementation of Pi0 vision-language-action inference.

    from tilelang_infer import Pi0Inference, random_checkpoint

    engine = Pi0Inference(random_checkpoint(), num_views=3, chunk_size=50)
    actions = engine.forward(images, state, noise)

The whole forward pass -- 27 vision layers, 18 encoder layers, and 18 decoder
layers per diffusion step -- is captured into one CUDA graph at construction, so
`forward` is a buffer copy and a replay.

Layout:
    kernels.py             TileLang kernel definitions
    fused_norm_kernels.py  kernels that absorb the norm into the following GEMM
    wrappers.py            one wrapper per call site, owning its tile config
    fused_wrappers.py      the fused decoder alternatives
    ops.py                 the operation table the forward pass is written against
    pi0_infer.py           the forward pass itself
    inference.py           weights, buffers, graph capture
    attention.py           vision and encoder attention, in torch
    buffers.py             scratch pool for graph-safe temporaries
    bench/                 timing, profiling and parity harnesses

Requires an H100 (the kernels use Hopper WGMMA and TMA) and TileLang 0.1.11.
"""
from __future__ import annotations

from .inference import Pi0Inference, random_checkpoint
from .ops import op_table

__all__ = ["Pi0Inference", "random_checkpoint", "op_table"]
