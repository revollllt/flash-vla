"""Pi0.5 RTX 5090 routes: torch reference and measured native CUDA fusions."""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import torch_ops as _torch
from . import fused_ffn as _fused_ffn
from . import fused_qkv as _fused_qkv
from . import fused_backbone as _fused_backbone

BACKENDS = {
    "torch": _torch,
    "fused-ffn": _fused_ffn,
    "fused-qkv": _fused_qkv,
    "fused-backbone": _fused_backbone,
}

REGISTRY = Registry(BACKENDS, default="torch")

__all__ = ["BACKENDS", "REGISTRY"]
