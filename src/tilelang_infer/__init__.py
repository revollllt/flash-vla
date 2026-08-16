"""A TileLang implementation of Pi0 vision-language-action inference.

    from tilelang_infer import Pi0Inference, random_checkpoint

    engine = Pi0Inference(random_checkpoint(), num_views=3, chunk_size=50)
    actions = engine.forward(images, state, noise)

The whole forward pass -- 27 vision layers, 18 encoder layers, and 18 decoder
layers per diffusion step -- is captured into one CUDA graph at construction, so
`forward` is a buffer copy and a replay.

The implementation is a hardware/model target at
`hardware.nvidia.h100.pi0`; this module keeps the original public imports.

Requires an H100 (the kernels use Hopper WGMMA and TMA) and TileLang 0.1.11.
"""
from __future__ import annotations

__all__ = ["Pi0Inference", "random_checkpoint", "op_table"]


def __getattr__(name: str):
    """Preserve the public API without loading the H100 target for model-only imports."""
    if name == "random_checkpoint":
        from .models.pi0 import random_checkpoint

        globals()[name] = random_checkpoint
        return random_checkpoint
    if name in {"Pi0Inference", "op_table"}:
        from .hardware.nvidia.h100.pi0 import Pi0Inference, op_table

        globals().update(Pi0Inference=Pi0Inference, op_table=op_table)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
