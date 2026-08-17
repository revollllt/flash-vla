"""Backend-agnostic config tuning.

The sweep skeleton and the candidate-space helpers, with no knowledge of any
backend, model or device: a backend supplies `build` and `invoke`, and derives
its own candidate set from its device spec. Depends on `runtime` for in-graph
timing and on nothing else.

A backend's adapter stays with that backend -- for TileLang, `rewrap` and the
axis derivation live in `hardware/.../backends/tilelang/autotune.py`, because
`tilelang.jit`, `pass_configs` and the raw-kernel registry are private to it and
have no counterpart in a hand-written CUDA backend.
"""

from .loop import SweepResult, report, sweep
from .space import cold_n_inner, grid

__all__ = ["SweepResult", "cold_n_inner", "grid", "report", "sweep"]
