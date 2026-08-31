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

    # The CUDA qkv wrapper leaves the pipeline's token-major Q unwritten: its
    # real Q is head-major scratch its own attention reads. A plan that splits
    # the pair would feed TileLang attention a stale buffer, so reject it here
    # rather than at the first step.
    attn_names = ("decoder_norm_qkv_rope", "decoder_attention")
    attn_route = tuple(plan.get(name, backend) for name in attn_names)
    if len(set(attn_route)) != 1:
        raise ValueError(
            "norm-qkv-rope and attention must select the same backend, got "
            f"{dict(zip(attn_names, attn_route))}")

    ffn_names = (
        "decoder_out_proj_residual",
        "decoder_norm_gated_ffn",
        "decoder_ffn_down_residual",
    )
    ffn_route = tuple(plan.get(name, backend) for name in ffn_names)
    allowed_ffn_routes = {
        ("tilelang", "tilelang", "tilelang"),
        ("tilelang", "cuda", "cuda"),
        ("cuda", "cuda", "cuda"),
    }
    if ffn_route not in allowed_ffn_routes:
        raise ValueError(
            "out-projection/norm-gated/down-residual must select an atomic "
            f"FFN route, got {dict(zip(ffn_names, ffn_route))}")

    table = {}
    assignments = {
        op_name: plan.get(op_name, backend) for op_name in _op_names(backend)
    }
    selected_by_backend: dict[str, set[str]] = {}
    for op_name, chosen in assignments.items():
        selected_by_backend.setdefault(chosen, set()).add(op_name)
    backend_tables = {
        chosen: _build_backend_table(
            chosen, fused=fused, selected_names=selected_names)
        for chosen, selected_names in selected_by_backend.items()
    }
    for op_name, chosen in assignments.items():
        if chosen not in backend_tables:
            raise AssertionError(f"missing backend table for {chosen}")
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
