"""TileLang backend for Pi0 call sites.

Exposes the wrapper registry contract (ALL_WRAPPERS / FUSED_WRAPPERS) that the
backend dispatcher consumes: `wrappers` supplies the standard call sites and
`fused_wrappers` overlays the fused decoder kernels. Both modules stay
importable individually for benchmarks that want a specific one.

`autotune` is deliberately not imported here. It is this backend's tuning
adapter -- the TileLang half of the sweep that produced the configs in
`wrappers.py`, with the loop itself in `flash_vla.tuning` -- needed only when
re-tuning, so importing it eagerly would pull the tuner into every engine
construction:

    from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang import autotune
"""

from __future__ import annotations

from . import fused_wrappers, wrappers

ALL_WRAPPERS = dict(wrappers.ALL_WRAPPERS)
FUSED_WRAPPERS = dict(fused_wrappers.FUSED_WRAPPERS)

__all__ = ["ALL_WRAPPERS", "FUSED_WRAPPERS", "wrappers", "fused_wrappers"]
