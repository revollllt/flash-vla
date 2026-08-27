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
"""
from __future__ import annotations

from types import SimpleNamespace

from .backends import (
    BACKENDS,
    backend_names as _backend_names,
    build_backend_table as _build_backend_table,
    build_table as _build_table,
)


def op_table(fused: bool = True, backend: str = "tilelang",
             plan: dict[str, str] | None = None) -> SimpleNamespace:
    """Build the operation table.

    `fused=True` overlays each backend's fused kernels. Pi0.5 has none yet --
    both of Pi0's fusions are decoder-only (see `backends/tilelang/fused_wrappers`). `plan` maps individual op names to a backend, overriding the
    single `backend` default -- a call-site-level dispatch for mixed-backend
    pipelines (e.g. `{"decoder_attention": "cuda"}`).
    """
    if plan is None:
        return _build_table(backend, fused=fused)

    for op_name, chosen in plan.items():
        if chosen not in BACKENDS:
            raise KeyError(f"plan names unknown backend {chosen!r} for {op_name!r}; "
                           f"known: {sorted(BACKENDS)}")
        names = _backend_names(chosen)
        if op_name not in names:
            raise KeyError(
                f"backend {chosen!r} does not implement call site {op_name!r}; "
                f"it provides {sorted(names)}")

    table = {}
    backend_tables = {}
    for op_name in _op_names(backend):
        chosen = plan.get(op_name, backend)
        if chosen not in backend_tables:
            backend_tables[chosen] = _build_backend_table(chosen, fused=fused)
        chosen_table = backend_tables[chosen]
        if op_name in chosen_table:
            table[op_name] = chosen_table[op_name]
    return SimpleNamespace(**table)


def _op_names(backend: str) -> list[str]:
    """Union of op names a backend can provide (unfused + fused)."""
    return sorted(_backend_names(backend))


def op_names(fused: bool = True, backend: str = "tilelang") -> list[str]:
    """Names in the table, for reporting which implementation is active."""
    return sorted(vars(op_table(fused, backend=backend)))
