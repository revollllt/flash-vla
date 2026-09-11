"""Initial upstream-backed LingBot routes."""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import fused_norm as _fused_norm
from . import aligned_vision as _aligned_vision
from . import expert_route as _expert_route
from . import fused_attention as _fused_attention
from . import fused_attn_kernel as _fused_attn_kernel
from . import fused_backbone as _fused_backbone
from . import fused_mlp as _fused_mlp
from . import fused_prefix as _fused_prefix
from . import fused_rope as _fused_rope
from . import packed_projections as _packed_projections
from . import patch_linear as _patch_linear
from . import rope_frequency as _rope_frequency
from . import rope_table as _rope_table
from . import time_modulation as _time_modulation
from . import streaming_softmax as _streaming_softmax
from . import upstream as _upstream
from . import vision_norm as _vision_norm

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
    "aligned-vision": _aligned_vision,
    "vision-norm": _vision_norm,
    "fused-mlp": _fused_mlp,
    "fused-backbone": _fused_backbone,
    "attention-kernel": _fused_attn_kernel,
    "streaming-softmax": _streaming_softmax,
}
REGISTRY = Registry(BACKENDS, default="upstream-reference")

__all__ = ["BACKENDS", "REGISTRY"]
