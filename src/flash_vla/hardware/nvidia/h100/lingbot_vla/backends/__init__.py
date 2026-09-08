"""Initial upstream-backed LingBot routes."""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import upstream as _upstream

BACKENDS = {
    "upstream-reference": _upstream,
    "upstream-shipped": _upstream,
}
REGISTRY = Registry(BACKENDS, default="upstream-reference")

__all__ = ["BACKENDS", "REGISTRY"]
