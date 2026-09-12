"""Hand-written CUDA stages for Pi0 on the RTX 5090.

`pointwise` is the kernel library (build, load, launch); `wrappers` is the
partial backend the registry sees. Importing this package re-exports the
backend contract, so `BACKENDS["cuda"]` can be this package itself.
"""
from __future__ import annotations

from . import pointwise
from .wrappers import NAMES, OPS, ROUTE_CONSTRAINTS, make_wrappers

__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "make_wrappers", "pointwise"]
