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
from . import bucketed_backbone as _bucketed_backbone
from . import cutlass_vision as _cutlass_vision
from . import cutlass_expert_residual as _cutlass_expert_residual
from . import fused_vision as _fused_vision
from . import fused_prefix_qkv as _fused_prefix_qkv
from . import mxfp8_backbone as _mxfp8_backbone
from . import fake_quant_ffn

BACKENDS = {
    "torch": _torch.BACKEND,
    "fused-ffn": _fused_ffn.BACKEND,
    "fused-qkv": _fused_qkv.BACKEND,
    "triton-qkv": _triton_qkv.BACKEND,
    "triton-qkv-finish": _triton_qkv_finish.BACKEND,
    "fused-backbone": _fused_backbone.BACKEND,
    "packed-ffn": _packed_ffn.BACKEND,
    "dual-ffn": _dual_ffn.BACKEND,
    "fused-attention": _fused_attention.BACKEND,
    "triton-qk-attention": _triton_qk_attention.BACKEND,
    "fused-residual": _fused_residual.BACKEND,
    "cutlass-backbone": _cutlass_backbone.BACKEND,
    "bucketed-backbone": _bucketed_backbone.BACKEND,
    "cutlass-vision": _cutlass_vision.BACKEND,
    "cutlass-expert-residual": _cutlass_expert_residual.BACKEND,
    "fused-vision": _fused_vision.BACKEND,
    "fused-prefix-qkv": _fused_prefix_qkv.BACKEND,
    # Backbone FFN of the mxfp8-llm-ffn recipe (target.py): its kernels and its
    # fake-quant reference, which the recipe-quality tools also run per layer.
    "mxfp8-backbone": _mxfp8_backbone.BACKEND,
    "fake-quant-mxfp8": fake_quant_ffn.backend("mxfp8"),
}

REGISTRY = Registry(BACKENDS, default="torch")

__all__ = ["BACKENDS", "REGISTRY"]
