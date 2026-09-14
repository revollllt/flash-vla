"""Pi0.5 RTX 5090 routes: torch reference and measured native CUDA fusions."""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import torch_ops as _torch
from . import fused_ffn as _fused_ffn
from . import fused_qkv as _fused_qkv
from . import fused_backbone as _fused_backbone
from . import packed_ffn as _packed_ffn
from . import fused_attention as _fused_attention
from . import fused_residual as _fused_residual
from . import cutlass_backbone as _cutlass_backbone
from . import fused_vision as _fused_vision
from . import fused_prefix_qkv as _fused_prefix_qkv

BACKENDS = {
    "torch": _torch,
    "fused-ffn": _fused_ffn,
    "fused-qkv": _fused_qkv,
    "fused-backbone": _fused_backbone,
    "packed-ffn": _packed_ffn,
    "fused-attention": _fused_attention,
    "fused-residual": _fused_residual,
    "cutlass-backbone": _cutlass_backbone,
    "fused-vision": _fused_vision,
    "fused-prefix-qkv": _fused_prefix_qkv,
}

REGISTRY = Registry(BACKENDS, default="torch")

__all__ = ["BACKENDS", "REGISTRY"]
