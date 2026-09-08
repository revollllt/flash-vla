"""Extreme-latency VLA inference: one runner, declarative Targets.

    from flash_vla import ModelRunner
    from flash_vla.hardware.nvidia.h100.pi0 import TARGET
    from flash_vla.models.pi0 import random_checkpoint, random_checkpoint_revision

    runner = ModelRunner(TARGET, random_checkpoint(),
                         model_revision=random_checkpoint_revision(0),
                         num_views=3, chunk_size=50)
    actions = runner.forward(images=images, state=state, noise=noise)

A Target declares its computation graph against the framework's op vocabulary
(`flash_vla.runtime`); the runner allocates, binds the plan, captures each
stage into a CUDA graph, and replays. Requires an H100 (the kernels use Hopper
WGMMA and TMA) and TileLang 0.1.11.
"""
from __future__ import annotations

__all__ = ["ModelRunner", "VLA"]


def __getattr__(name: str):
    """Load the runtime lazily so model-only imports stay light."""
    if name in __all__:
        from . import runtime

        value = getattr(runtime, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
