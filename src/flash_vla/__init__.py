"""Extreme-latency VLA inference: one runner, a model definition per model, a Target per device.

    from flash_vla.inference import build

    runner = build("rtx5090/pi05", workload="robodojo")   # seeded weights and fixture
    actions = runner.forward(**runner.sample_inputs(0))

`build` names a Target and one of its workloads (`docs/workloads.md`); the
weights and fixture come from its model's `sources` module.

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
