"""TileLang backend for Pi0 call sites.

Exposes the wrapper registry contract (ALL_WRAPPERS / FUSED_WRAPPERS) that the
backend dispatcher consumes.  overlays the fused decoder
kernels; both modules stay importable individually for benchmarks that want a
specific one.
"""

from __future__ import annotations

from . import fused_wrappers, wrappers

ALL_WRAPPERS = dict(wrappers.ALL_WRAPPERS)
FUSED_WRAPPERS = dict(fused_wrappers.FUSED_WRAPPERS)

__all__ = ["ALL_WRAPPERS", "FUSED_WRAPPERS", "wrappers", "fused_wrappers"]
