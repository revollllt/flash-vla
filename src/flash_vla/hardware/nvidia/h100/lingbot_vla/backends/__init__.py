"""Initial upstream-backed LingBot routes."""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import patch_linear as _patch_linear
from . import rope_frequency as _rope_frequency
from . import rope_table as _rope_table
from . import time_modulation as _time_modulation
from . import upstream as _upstream

BACKENDS = {
    "upstream-reference": _upstream,
    "upstream-shipped": _upstream,
    "rope-frequency": _rope_frequency,
    "patch-linear": _patch_linear,
    "rope-table": _rope_table,
    "time-modulation": _time_modulation,
}
REGISTRY = Registry(BACKENDS, default="upstream-reference")

__all__ = ["BACKENDS", "REGISTRY"]
