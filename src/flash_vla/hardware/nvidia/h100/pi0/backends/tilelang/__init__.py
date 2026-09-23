"""TileLang backends for Pi0 call sites.

Two backends, each a `flash_vla.runtime.registry.Backend`: `BACKEND` (every
call site; the reference route) and `fused.BACKEND` (the three action-expert
fusions the shipped plan routes to). Both stay importable on
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
from .wrappers import BACKEND, NAMES, OPS, make_wrappers

__all__ = ["BACKEND", "NAMES", "OPS", "fused", "make_wrappers", "wrappers"]
