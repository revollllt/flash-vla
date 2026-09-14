"""Pi0.5 RTX 5090 routes: torch reference and measured native CUDA fusions."""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import torch_ops as _torch
from . import fused_ffn as _fused_ffn
from . import fused_qkv as _fused_qkv
from . import triton_qkv as _triton_qkv
from . import triton_qkv_finish as _triton_qkv_finish
from . import fused_backbone as _fused_backbone
from . import packed_ffn as _packed_ffn
from . import dual_ffn as _dual_ffn
from . import fused_attention as _fused_attention
from . import triton_qk_attention as _triton_qk_attention
from . import fused_residual as _fused_residual
from . import cutlass_backbone as _cutlass_backbone
from . import cutlass_vision as _cutlass_vision
from . import cutlass_expert_residual as _cutlass_expert_residual
from . import fused_vision as _fused_vision
from . import fused_prefix_qkv as _fused_prefix_qkv

BACKENDS = {
    "torch": _torch,
    "fused-ffn": _fused_ffn,
    "fused-qkv": _fused_qkv,
    "triton-qkv": _triton_qkv,
    "triton-qkv-finish": _triton_qkv_finish,
    "fused-backbone": _fused_backbone,
    "packed-ffn": _packed_ffn,
    "dual-ffn": _dual_ffn,
    "fused-attention": _fused_attention,
    "triton-qk-attention": _triton_qk_attention,
    "fused-residual": _fused_residual,
    "cutlass-backbone": _cutlass_backbone,
    "cutlass-vision": _cutlass_vision,
    "cutlass-expert-residual": _cutlass_expert_residual,
    "fused-vision": _fused_vision,
    "fused-prefix-qkv": _fused_prefix_qkv,
}

REGISTRY = Registry(BACKENDS, default="torch")

__all__ = ["BACKENDS", "REGISTRY"]
