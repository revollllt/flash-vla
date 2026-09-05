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

A plan is validated through the runtime before any table is built
(`flash_vla.runtime.binding`); Pi0's one backend declares no route constraint.
"""
from __future__ import annotations

from types import SimpleNamespace

from flash_vla.runtime import binding

from .backends import BACKENDS, build_table as _build_table


def _provided() -> dict[str, set[str]]:
    return {name: set(module.ALL_WRAPPERS) | set(module.FUSED_WRAPPERS)
            for name, module in BACKENDS.items()}


def resolve_plan(plan: dict[str, str] | None, backend: str = "tilelang") -> dict[str, str]:
    """The backend of every call site under `plan`, validated; `backend` where unnamed."""
    routes = binding.resolve(plan, backend, _op_names(backend))
    binding.check_backends_provide(routes, _provided())
    binding.validate(routes, {name: tuple(getattr(module, "ROUTE_CONSTRAINTS", ()))
                              for name, module in BACKENDS.items()})
    return routes


def op_table(fused: bool = True, backend: str = "tilelang",
             plan: dict[str, str] | None = None) -> SimpleNamespace:
    """Build the operation table.

    `fused=True` overlays each backend's fused kernels (TileLang fused decoder
    by default). `plan` maps individual op names to a backend, overriding the
    single `backend` default -- a call-site-level dispatch for mixed-backend
    pipelines (e.g. `{"decoder_attention": "cuda"}`).
    """
    if plan is None:
        return _build_table(backend, fused=fused)

    table = {}
    for op_name, chosen in resolve_plan(plan, backend).items():
        module = BACKENDS[chosen]
        if op_name in module.FUSED_WRAPPERS and fused:
            table[op_name] = module.FUSED_WRAPPERS[op_name]
        elif op_name in module.ALL_WRAPPERS:
            table[op_name] = module.ALL_WRAPPERS[op_name]
    return SimpleNamespace(**table)


def _op_names(backend: str) -> list[str]:
    """Union of op names a backend can provide (unfused + fused)."""
    module = BACKENDS[backend]
    return sorted(set(module.ALL_WRAPPERS) | set(module.FUSED_WRAPPERS))


def op_names(fused: bool = True, backend: str = "tilelang") -> list[str]:
    """Names in the table, for reporting which implementation is active."""
    return sorted(vars(op_table(fused, backend=backend)))
