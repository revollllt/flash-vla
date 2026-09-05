"""The operation table the forward pass is written against.

`pipeline` calls operations by attribute (`ops.decoder_attention(...)`) rather
than importing them, so the implementation behind each call site is a table
lookup, not control flow.

Backends are selected per call site, not per engine: `op_table` takes a plan
mapping each op name to the backend that implements it. This is how a pipeline
mixes TileLang and hand-written CUDA kernels -- some ops on one backend, others
on another -- while the pipeline itself stays backend-agnostic. The table is
built once and passed down explicitly; no global dispatch state, which matters
because a benchmark routinely holds two configurations alive at the same time.

A plan is validated before any backend state is built: every named backend
must exist and provide the call site, and the route must satisfy every
constraint a backend declares (`backends/cuda` owns the decoder pairs). The
rules live with the backend that owns the buffer contract, and the runtime
applies them (`flash_vla.runtime.binding`).
"""
from __future__ import annotations

from types import SimpleNamespace

from flash_vla.runtime import binding

from .backends import (
    BACKENDS,
    backend_names as _backend_names,
    build_backend_table as _build_backend_table,
    build_table as _build_table,
    provided_names as _provided_names,
    route_constraints as _route_constraints,
)


def resolve_plan(plan: dict[str, str] | None, backend: str = "tilelang") -> dict[str, str]:
    """The backend of every call site under `plan`, validated; `backend` where unnamed."""
    routes = binding.resolve(plan, backend, _op_names(backend))
    binding.check_backends_provide(routes, _provided_names())
    binding.validate(routes, _route_constraints())
    return routes


def op_table(fused: bool = True, backend: str = "tilelang",
             plan: dict[str, str] | None = None) -> SimpleNamespace:
    """Build the operation table.

    `fused=True` overlays each backend's fused kernels. Pi0.5 has none yet --
    both of Pi0's fusions are decoder-only (see `backends/tilelang/fused_wrappers`).
    `plan` maps individual op names to a backend, overriding the single
    `backend` default -- a call-site-level dispatch for mixed-backend pipelines
    (e.g. `{"decoder_attention": "cuda"}`).
    """
    if plan is None:
        return _build_table(backend, fused=fused)

    assignments = resolve_plan(plan, backend)
    selected_by_backend: dict[str, set[str]] = {}
    for op_name, chosen in assignments.items():
        selected_by_backend.setdefault(chosen, set()).add(op_name)
    backend_tables = {
        chosen: _build_backend_table(
            chosen, fused=fused, selected_names=selected_names)
        for chosen, selected_names in selected_by_backend.items()
    }
    table = {}
    for op_name, chosen in assignments.items():
        chosen_table = backend_tables[chosen]
        if op_name in chosen_table:
            table[op_name] = chosen_table[op_name]
    return SimpleNamespace(**table)


def _op_names(backend: str) -> list[str]:
    """Union of op names a backend can provide (unfused + fused)."""
    if backend not in BACKENDS:
        raise KeyError(f"unknown default backend {backend!r}; known: {sorted(BACKENDS)}")
    return sorted(_backend_names(backend))


def op_names(fused: bool = True, backend: str = "tilelang") -> list[str]:
    """Names in the table, for reporting which implementation is active."""
    return sorted(vars(op_table(fused, backend=backend)))
