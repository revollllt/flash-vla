"""Raw-CUDA backend for the Pi0.5 H100 target.

`wrappers` provides the call sites a plan can route here (the decoder
attention half); `FFNTaskloop` is the persistent FFN kernel's host side.
"""

from . import wrappers
from .taskloop import FFNTaskloop, build_table

ALL_WRAPPERS = dict(wrappers.ALL_WRAPPERS)
FUSED_WRAPPERS = dict(wrappers.FUSED_WRAPPERS)

__all__ = ["ALL_WRAPPERS", "FUSED_WRAPPERS", "FFNTaskloop", "build_table", "wrappers"]
