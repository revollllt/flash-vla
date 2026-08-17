"""Backend abstraction for Pi0 call sites.

Every backend is a flat dict of call-site wrappers with identical signatures:
    {op_name: callable}

A call site resolves to one implementation at engine construction time, so a
single pipeline can mix backends -- a TileLang wrapper for one op, a hand-written
CUDA kernel for another -- without the pipeline knowing which is which.

Contract for a backend module:

    ALL_WRAPPERS   dict[str, Callable]   unfused call sites
    FUSED_WRAPPERS dict[str, Callable]   fused overlays (may be empty)
"""

from __future__ import annotations

from types import SimpleNamespace

from . import tilelang as _tilelang

# name -> module exposing ALL_WRAPPERS / FUSED_WRAPPERS. A new backend registers here.
BACKENDS = {
    "tilelang": _tilelang,
}

# Default fused plan: every op the TileLang backend provides a fused overlay
# for runs fused; everything else runs its unfused wrapper.
DEFAULT_FUSED_OPS = tuple(sorted(_tilelang.FUSED_WRAPPERS))

__all__ = ["BACKENDS", "DEFAULT_FUSED_OPS", "build_table"]


def build_table(backend: str = "tilelang", fused: bool = True) -> SimpleNamespace:
    """Build the operation table for one backend, optional fused overlays."""
    module = BACKENDS[backend]
    table = dict(module.ALL_WRAPPERS)
    if fused:
        table.update(module.FUSED_WRAPPERS)
    return SimpleNamespace(**table)
