"""Backend registry for Pi0 call sites.

Three backends, each satisfying the contract of
`flash_vla.runtime.registry`:

    tilelang        every call site; the reference route
    siglip-cublas   the two pre-norm vision projections as LayerNorm + a fused
                    cuBLASLt epilogue, from the shared SigLIP component package
    tilelang-fused  the three action-expert fusions (lazy pre-norm and
                    FlashDecoding) the shipped plan routes to

`REGISTRY` is what the Target hands the runner; `tilelang` is the default a
plan does not name.
"""
from __future__ import annotations

from flash_vla.hardware.nvidia.h100.siglip.backends import cublas as _siglip_cublas
from flash_vla.runtime.registry import Registry

from . import tilelang as _tilelang

BACKENDS = {
    "tilelang": _tilelang,
    "siglip-cublas": _siglip_cublas,
    "tilelang-fused": _tilelang.fused,
}

REGISTRY = Registry(BACKENDS, default="tilelang")

__all__ = ["BACKENDS", "REGISTRY"]
