"""Backend registry for Pi0.5 call sites.

Four backends, each satisfying the contract of `flash_vla.runtime.registry`
(`NAMES`, `make_wrappers`, `ROUTE_CONSTRAINTS`, `graph_contract`, `OPS`):

    tilelang       every call site; the reference route
    siglip-cublas  the two pre-norm vision projections as LayerNorm + a fused
                   cuBLASLt epilogue, from the shared SigLIP component package
    cuda           the encoder attention and the decoder attention / FFN halves,
                   shipped launch semantics
    cuda-pdl       the same wrappers with the programmatic-dependent-launch chain
                   armed (early triggers, dependent-launch attributes, waits at
                   the first dependent read); which boundaries overlap is decided
                   by which call sites a plan routes here

`REGISTRY` is what the Target hands the runner; `tilelang` is the default a
plan does not name.
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from flash_vla.hardware.nvidia.h100.siglip.backends import cublas as _siglip_cublas
from flash_vla.runtime.registry import Registry

from . import cuda as _cuda
from . import tilelang as _tilelang

_cuda_pdl = SimpleNamespace(
    NAMES=_cuda.NAMES,
    OPS=_cuda.OPS,
    ROUTE_CONSTRAINTS=_cuda.ROUTE_CONSTRAINTS,
    graph_contract=_cuda.graph_contract,
    make_wrappers=partial(_cuda.make_wrappers, pdl_chain=True),
)

BACKENDS = {
    "tilelang": _tilelang,
    "siglip-cublas": _siglip_cublas,
    "cuda": _cuda,
    "cuda-pdl": _cuda_pdl,
}

REGISTRY = Registry(BACKENDS, default="tilelang")

__all__ = ["BACKENDS", "REGISTRY"]
