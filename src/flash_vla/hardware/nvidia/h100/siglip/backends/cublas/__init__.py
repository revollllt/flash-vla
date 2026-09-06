"""cuBLASLt vision backend: the two pre-norm projections.

Exposes the backend contract the runtime's registry consumes. `wrappers` stays
importable on its own so a parity script can call the functions directly, and
`norm_gemm_reference` is the T2 ABI mirror they are checked against.
"""
from __future__ import annotations

from . import norm_gemm_reference, wrappers
from .wrappers import NAMES, OPS, ROUTE_CONSTRAINTS, make_wrappers

__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "make_wrappers", "norm_gemm_reference",
           "wrappers"]
