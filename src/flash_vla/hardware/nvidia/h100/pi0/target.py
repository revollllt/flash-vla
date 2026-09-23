"""Pi0 on H100 SXM5: the model on H100 backends.

The model is `flash_vla.models.pi0`; this Target decides the backends, the
one shipped plan and the reference plan.
"""
from __future__ import annotations

from flash_vla.models.pi0.definition import Pi0Model
from flash_vla.runtime.vla import Target

from .backends import REGISTRY

TARGET = Target(
    name="hardware/nvidia/h100/pi0",
    hardware="h100-sxm5-80gb",
    model=Pi0Model(),
    registry=REGISTRY,
    # The shipped plan: the three action-expert fusions (lazy pre-norm on the
    # gated FFN and the output projection, FlashDecoding attention), and the
    # vision tower's two pre-norm projections and its attention on the shared
    # SigLIP CUDA backend, and the two backbone call sites the shared Gemma
    # component implements faster -- the prefix attention as one fused MQA
    # kernel instead of a four-launch torch chain that also copied its result,
    # and the output projection on cuBLAS. The FFN down projection stays on
    # TileLang: cuBLAS measured slower there in the graph, where the hidden
    # buffer is L2-resident.
    plan={
        "llm_backbone_attention": "gemma-cuda",
        "llm_backbone_out_proj_residual": "gemma-cuda",
        "action_expert_norm_gated_ffn": "tilelang-fused",
        "action_expert_action_out_proj": "tilelang-fused",
        "action_expert_attention": "tilelang-fused",
        "vision_encoder_norm_qkv": "siglip-cuda",
        "vision_encoder_norm_ffn_up": "siglip-cuda",
        "vision_encoder_attention": "siglip-cuda",
    },
    # The reference route: every call site on the unfused TileLang wrappers.
    reference_plan={},
)

__all__ = ["TARGET"]
