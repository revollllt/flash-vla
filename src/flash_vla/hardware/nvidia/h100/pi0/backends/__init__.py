"""Backend registry for Pi0 call sites.

Two backends, both TileLang, each satisfying the contract of
`flash_vla.runtime.registry`:

    tilelang        every call site; the reference route
    tilelang-fused  the three action-expert fusions (lazy pre-norm and
                    FlashDecoding) the shipped plan routes to

`REGISTRY` is what the Target hands the runner; `tilelang` is the default a
plan does not name.
"""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import tilelang as _tilelang

BACKENDS = {
    "tilelang": _tilelang,
    "tilelang-fused": _tilelang.fused,
}

REGISTRY = Registry(BACKENDS, default="tilelang")

__all__ = ["BACKENDS", "REGISTRY"]
