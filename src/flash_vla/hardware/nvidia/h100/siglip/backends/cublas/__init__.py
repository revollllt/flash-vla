"""cuBLASLt vision backend: the two pre-norm projections.

Exposes `BACKEND`, the `flash_vla.runtime.registry.Backend` a Target
registers. `wrappers` stays importable on its own so a parity script can call the functions directly, and
`norm_gemm_reference` is the T2 ABI mirror they are checked against.
"""
from __future__ import annotations

from . import norm_gemm_reference, wrappers
from .wrappers import BACKEND, NAMES, make_wrappers

__all__ = ["BACKEND", "NAMES", "make_wrappers", "norm_gemm_reference",
           "wrappers"]
