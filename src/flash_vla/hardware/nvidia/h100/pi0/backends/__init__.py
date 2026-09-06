"""Backend registry for Pi0 call sites.

Four backends, each satisfying the contract of
`flash_vla.runtime.registry`:

    tilelang        every call site; the reference route
    siglip-cublas   the two pre-norm vision projections as LayerNorm + a fused
                    cuBLASLt epilogue, from the shared SigLIP component package
    siglip-cuda     the fused vision attention kernel, and those two
                    projections with a hand-written LayerNorm
    tilelang-fused  the three action-expert fusions (lazy pre-norm and
                    FlashDecoding) the shipped plan routes to

`REGISTRY` is what the Target hands the runner; `tilelang` is the default a
plan does not name.
"""
from __future__ import annotations

from flash_vla.hardware.nvidia.h100.siglip.backends import cublas as _siglip_cublas
from flash_vla.hardware.nvidia.h100.siglip.backends import cuda as _siglip_cuda
from flash_vla.runtime.registry import Registry

from . import tilelang as _tilelang

BACKENDS = {
    "tilelang": _tilelang,
    "siglip-cublas": _siglip_cublas,
    "siglip-cuda": _siglip_cuda,
    "tilelang-fused": _tilelang.fused,
}

REGISTRY = Registry(BACKENDS, default="tilelang")

__all__ = ["BACKENDS", "REGISTRY"]
