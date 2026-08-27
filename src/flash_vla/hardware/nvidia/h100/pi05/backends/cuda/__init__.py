"""Raw-CUDA persistent FFN backend for the Pi0.5 H100 target."""

from . import wrappers
from .taskloop import FFNTaskloop, build_table

ALL_WRAPPERS = dict(wrappers.ALL_WRAPPERS)
FUSED_WRAPPERS = dict(wrappers.FUSED_WRAPPERS)

__all__ = [
    "ALL_WRAPPERS", "FUSED_WRAPPERS", "FFNTaskloop", "build_table",
    "wrappers",
]
