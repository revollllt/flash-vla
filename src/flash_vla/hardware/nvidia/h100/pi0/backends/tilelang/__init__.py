"""TileLang backends for Pi0 call sites.

Two backends satisfy the registry contract of `flash_vla.runtime.registry`:
`wrappers` (every call site; the reference route) and `fused` (the three
action-expert fusions the shipped plan routes to). Both stay importable on
their own for scripts that call their functions directly.

`autotune` is deliberately not imported here. It is this backend's tuning
adapter -- the TileLang half of the sweep that produced the configs in
`wrappers.py`, with the loop itself in `flash_vla.tuning` -- needed only when
re-tuning, so importing it eagerly would pull the tuner into every engine
construction:

    from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang import autotune
"""

from __future__ import annotations

from . import fused, wrappers
from .wrappers import NAMES, OPS, ROUTE_CONSTRAINTS, make_wrappers

__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "fused", "make_wrappers", "wrappers"]
