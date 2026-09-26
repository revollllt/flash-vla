"""Pi0.5 RTX 5090 routes: torch reference and measured native CUDA fusions."""
from __future__ import annotations


from flash_vla.runtime.registry import Registry

from . import torch_ops
from . import fused_ffn
from . import fused_qkv
from . import triton_qkv
from . import triton_qkv_finish
from . import packed_ffn
from . import dual_ffn
from . import fused_attention
from . import triton_qk_attention
from . import split_kv_attention
from . import triton_vision_attention
from . import fused_residual
from . import cutlass_backbone
from . import cutlass_vision
from . import cutlass_expert_residual
from . import fused_vision
from . import fused_prefix_qkv
from . import mxfp8_backbone
from . import fake_quant_ffn

BACKENDS = {
    "torch": torch_ops.BACKEND,
    "fused-ffn": fused_ffn.BACKEND,
    "fused-qkv": fused_qkv.BACKEND,
    "triton-qkv": triton_qkv.BACKEND,
    "triton-qkv-finish": triton_qkv_finish.BACKEND,
    "packed-ffn": packed_ffn.BACKEND,
    "dual-ffn": dual_ffn.BACKEND,
    "fused-attention": fused_attention.BACKEND,
    "triton-qk-attention": triton_qk_attention.BACKEND,
    "split-kv-attention": split_kv_attention.BACKEND,
    "triton-vision-attention": triton_vision_attention.BACKEND,
    "fused-residual": fused_residual.BACKEND,
    "cutlass-backbone": cutlass_backbone.BACKEND,
    "cutlass-vision": cutlass_vision.BACKEND,
    "cutlass-expert-residual": cutlass_expert_residual.BACKEND,
    "fused-vision": fused_vision.BACKEND,
    "fused-prefix-qkv": fused_prefix_qkv.BACKEND,
    # Backbone FFN of the mxfp8-llm-ffn recipe (target.py): its kernels and its
    # fake-quant reference, which the recipe-quality tools also run per layer.
    "mxfp8-backbone": mxfp8_backbone.BACKEND,
    "fake-quant-mxfp8": fake_quant_ffn.backend("mxfp8"),
}

REGISTRY = Registry(BACKENDS, default="torch")

__all__ = ["BACKENDS", "REGISTRY"]
