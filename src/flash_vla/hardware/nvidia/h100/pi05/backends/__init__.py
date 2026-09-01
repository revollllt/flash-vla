"""Backend abstraction for Pi0.5 call sites.

Every backend is a flat dict of call-site wrappers with identical signatures:
    {op_name: callable}

A call site resolves to one implementation at engine construction time, so a
single pipeline can mix backends -- a TileLang wrapper for one op, a hand-written
CUDA kernel for another -- without the pipeline knowing which is which.

Contract for a stateless backend module:

    ALL_WRAPPERS   dict[str, Callable]   unfused call sites
    FUSED_WRAPPERS dict[str, Callable]   fused overlays (may be empty)

A stateful backend instead exposes ``WRAPPER_NAMES`` and
``make_wrappers(selected_names=...)``. The factory is called once per operation
table so its route-specific state, scratch, and packed weights follow the
owning engine's lifetime.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from . import cuda as _cuda
from . import tilelang as _tilelang

# PDL-chain variant of the CUDA backend: identical wrappers, launched with the
# programmatic-dependency chain armed (early triggers, dependent-launch
# attributes, waits at first dependent read). Which boundaries overlap is
# decided by which call sites a plan routes here, so one registration serves
# both the FFN-only and the full-chain plans.
_cuda_pdl = SimpleNamespace(
    WRAPPER_NAMES=_cuda.WRAPPER_NAMES,
    FUSED_WRAPPERS=_cuda.FUSED_WRAPPERS,
    make_wrappers=partial(_cuda.make_wrappers, pdl_chain=True),
)

# name -> module exposing the backend contract above. A new backend registers here.
BACKENDS = {
    "tilelang": _tilelang,
    "cuda": _cuda,
    "cuda-pdl": _cuda_pdl,
}

# Default fused plan: every op the TileLang backend provides a fused overlay
# for runs fused; everything else runs its unfused wrapper.
DEFAULT_FUSED_OPS = tuple(sorted(_tilelang.FUSED_WRAPPERS))

__all__ = [
    "BACKENDS", "DEFAULT_FUSED_OPS", "backend_names", "build_backend_table",
    "build_table",
]


def backend_names(backend: str) -> set[str]:
    """Return the call-site manifest without constructing backend state."""
    module = BACKENDS[backend]
    unfused = (module.WRAPPER_NAMES if hasattr(module, "WRAPPER_NAMES")
               else module.ALL_WRAPPERS.keys())
    return set(unfused) | set(module.FUSED_WRAPPERS)


def build_backend_table(
        backend: str, fused: bool = True,
        selected_names: set[str] | None = None) -> dict:
    """Instantiate one backend table, including engine-owned runtime state."""
    module = BACKENDS[backend]
    factory = getattr(module, "make_wrappers", None)
    table = (factory(selected_names=selected_names)
             if factory is not None else dict(module.ALL_WRAPPERS))
    if fused:
        table.update(module.FUSED_WRAPPERS)
    return table


def build_table(backend: str = "tilelang", fused: bool = True) -> SimpleNamespace:
    """Build the operation table for one backend, optional fused overlays."""
    return SimpleNamespace(**build_backend_table(backend, fused=fused))
