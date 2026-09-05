"""Backend registry for Pi0.5 call sites.

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

Either kind may declare ``ROUTE_CONSTRAINTS``: the call sites that share a
buffer contract and must resolve to it together (`flash_vla.runtime.binding`).
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from flash_vla.runtime.binding import RouteConstraint

from . import cuda as _cuda
from . import tilelang as _tilelang

# PDL-chain variant of the CUDA backend: identical wrappers and constraints,
# launched with the programmatic-dependency chain armed (early triggers,
# dependent-launch attributes, waits at first dependent read). Which boundaries
# overlap is decided by which call sites a plan routes here, so one registration
# serves both the FFN-only and the full-chain plans.
_cuda_pdl = SimpleNamespace(
    WRAPPER_NAMES=_cuda.WRAPPER_NAMES,
    FUSED_WRAPPERS=_cuda.FUSED_WRAPPERS,
    ROUTE_CONSTRAINTS=_cuda.ROUTE_CONSTRAINTS,
    graph_contract=_cuda.graph_contract,
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
    "atomic_groups", "build_table", "graph_contract", "provided_names", "route_constraints",
]


def backend_names(backend: str) -> set[str]:
    """Return the call-site manifest without constructing backend state."""
    module = BACKENDS[backend]
    unfused = (module.WRAPPER_NAMES if hasattr(module, "WRAPPER_NAMES")
               else module.ALL_WRAPPERS.keys())
    return set(unfused) | set(module.FUSED_WRAPPERS)


def provided_names() -> dict[str, set[str]]:
    """Call sites each registered backend provides."""
    return {name: backend_names(name) for name in BACKENDS}


def graph_contract(routes) -> dict[str, list[str]]:
    """The union of every routed backend's graph contract for `routes`."""
    merged: dict[str, list[str]] = {"forbid": [], "require_one": []}
    for name, module in BACKENDS.items():
        declare = getattr(module, "graph_contract", None)
        if declare is None or name not in set(routes.values()):
            continue
        contract = declare(routes)
        for key in merged:
            merged[key] += [p for p in contract.get(key, ()) if p not in merged[key]]
    return merged


def atomic_groups(routes) -> tuple[frozenset[str], ...]:
    """Call-site groups that must run together on `routes`: every constraint
    whose members all resolve to the declaring backend."""
    groups: list[frozenset[str]] = []
    for backend, constraints in route_constraints().items():
        for constraint in constraints:
            if all(routes.get(name) == backend for name in constraint.members):
                groups.append(frozenset(constraint.members))
    # Merge overlapping groups (the FFN pair and the out-projection that requires it).
    merged: list[set[str]] = []
    for group in groups:
        hit = [g for g in merged if g & group]
        for g in hit:
            merged.remove(g)
            group = frozenset(group | g)
        merged.append(set(group))
    return tuple(frozenset(g) for g in merged)


def route_constraints() -> dict[str, tuple[RouteConstraint, ...]]:
    """Route constraints each registered backend declares (possibly none)."""
    return {name: tuple(getattr(module, "ROUTE_CONSTRAINTS", ()))
            for name, module in BACKENDS.items()}


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
