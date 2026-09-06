"""Backend registry for Pi0 call sites.

Three backends, each satisfying the contract of
`flash_vla.runtime.registry`:

    tilelang        every call site; the reference route
    tilelang-fused  the three action-expert fusions (lazy pre-norm and
                    FlashDecoding) the shipped plan routes to
    gemma-cuda      the shared Gemma backbone component package, which Pi0.5
                    also registers; it owns no scratch crossing a call site,
                    so any of its call sites may be routed here alone

`REGISTRY` is what the Target hands the runner; `tilelang` is the default a
plan does not name.
"""
from __future__ import annotations

from flash_vla.hardware.nvidia.h100.gemma_backbone.backends import cuda as _gemma_cuda
from flash_vla.runtime.registry import Registry

from . import tilelang as _tilelang

BACKENDS = {
    "tilelang": _tilelang,
    "tilelang-fused": _tilelang.fused,
    "gemma-cuda": _gemma_cuda,
}

REGISTRY = Registry(BACKENDS, default="tilelang")

__all__ = ["BACKENDS", "REGISTRY"]
