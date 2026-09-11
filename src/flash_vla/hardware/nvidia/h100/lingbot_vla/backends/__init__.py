"""Initial upstream-backed LingBot routes."""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import fused_norm as _fused_norm
from . import expert_route as _expert_route
from . import fused_attention as _fused_attention
from . import fused_prefix as _fused_prefix
from . import fused_rope as _fused_rope
from . import packed_projections as _packed_projections
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
    "fused-norm": _fused_norm,
    "packed-projections": _packed_projections,
    "expert-loop": _expert_route,
    "fused-rope": _fused_rope,
    "fused-attention": _fused_attention,
    "fused-prefix": _fused_prefix,
}
REGISTRY = Registry(BACKENDS, default="upstream-reference")

__all__ = ["BACKENDS", "REGISTRY"]
