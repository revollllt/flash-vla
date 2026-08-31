"""Raw-CUDA backend for the Pi0.5 H100 target.

`wrappers` provides the call sites a plan can route here -- the decoder
attention half and the FFN half -- as a stateful backend, so each op table
owns its own libraries, scratch, and packed weights. `FFNTaskloop` is the
persistent FFN kernel's host side.
"""

from . import wrappers
from .taskloop import FFNTaskloop, build_table

ATTENTION_NAMES = wrappers.ATTENTION_NAMES
WRAPPER_NAMES = wrappers.WRAPPER_NAMES
FUSED_WRAPPERS = dict(wrappers.FUSED_WRAPPERS)
make_wrappers = wrappers.make_wrappers

__all__ = [
    "ATTENTION_NAMES", "WRAPPER_NAMES", "FUSED_WRAPPERS", "make_wrappers",
    "FFNTaskloop", "build_table", "wrappers",
]
