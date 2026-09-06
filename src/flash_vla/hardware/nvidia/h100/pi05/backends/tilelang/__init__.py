"""TileLang backend for Pi0.5 call sites.

Exposes the backend contract the runtime's registry consumes
(`flash_vla.runtime.registry`): `NAMES`, `make_wrappers`, `ROUTE_CONSTRAINTS`
and `OPS`. `wrappers` stays importable on its own for the lab scripts that call
its functions directly.

`autotune` is deliberately not imported here. It is this backend's tuning
adapter -- the TileLang half of the sweep that produced the configs in
`wrappers.py`, with the loop itself in `flash_vla.tuning` -- needed only when
re-tuning, so importing it eagerly would pull the tuner into every engine
construction:

    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import autotune
"""

from __future__ import annotations

from . import wrappers
from .wrappers import NAMES, OPS, ROUTE_CONSTRAINTS, make_wrappers

__all__ = ["NAMES", "OPS", "ROUTE_CONSTRAINTS", "make_wrappers", "wrappers"]
