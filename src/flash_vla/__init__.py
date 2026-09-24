"""Extreme-latency VLA inference: one runner, a model definition per model, a Target per device.

    from flash_vla import ModelRunner
    from flash_vla.hardware.nvidia.h100.pi0 import TARGET
    from flash_vla.models.pi0 import random_checkpoint, random_checkpoint_revision
    from flash_vla.provenance import WeightsProvenance

    revision = random_checkpoint_revision(0)
    runner = ModelRunner(TARGET, random_checkpoint(),
                         weights_provenance=WeightsProvenance(checkpoint_id=revision,
                                                              checkpoint_digest=revision),
                         num_views=3, chunk_size=50)
    actions = runner.forward(images=images, state=state, noise=noise)

A harness builds a Target by name instead (`flash_vla.inference.build`), with
the weights and fixture its model's `sources` module resolves.

A model (`flash_vla.models.<model>`) writes its computation graph against the
framework's op vocabulary; a Target (`flash_vla.hardware.<vendor>.<device>`)
composes it with that device's backends and plans; the runner
(`flash_vla.runtime`) allocates, binds the plan, captures each stage into a
CUDA graph, and replays.
"""
from __future__ import annotations

__all__ = ["ModelDefinition", "ModelRunner", "Target"]


def __getattr__(name: str) -> object:
    """Load the runtime lazily so model-only imports stay light."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import runtime

    value = vars(runtime)[name]
    globals()[name] = value
    return value
