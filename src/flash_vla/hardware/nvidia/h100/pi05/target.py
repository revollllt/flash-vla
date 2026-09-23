"""Pi0.5 on H100 SXM5: the model with 64-row padding, on H100 backends.

The model is `flash_vla.models.pi05` (graph, shapes, the prompt host slot);
this Target decides the layout its kernels see, the backends, the one shipped
plan and the reference plan, and the measured ceilings of this machine.
"""
from __future__ import annotations

from flash_vla.models.pi05.definition import Pi05Layout, Pi05Model
from flash_vla.runtime.cost import Ceiling
from flash_vla.runtime.vla import Target

from .backends import REGISTRY

TARGET = Target(
    name="hardware/nvidia/h100/pi05",
    hardware="h100-sxm5-80gb",
    model=Pi05Model(Pi05Layout(row_pad=64, masked_backbone=False)),
    registry=REGISTRY,
    # The shipped plan: the fused backbone attention from the shared Gemma
    # component package (proven bit-identical to this Target's former copy of
    # the same kernel source, job 599788), and the decoder's attention and FFN
    # halves on the CUDA backend with the PDL chain armed.
    plan={
        "llm_backbone_attention": "gemma-cuda",
        "action_expert_norm_qkv_rope": "cuda-pdl",
        "action_expert_attention": "cuda-pdl",
        "action_expert_out_proj_residual": "cuda-pdl",
        "action_expert_norm_gated_ffn": "cuda-pdl",
        "action_expert_ffn_down_residual": "cuda-pdl",
    },
    # The reference route: every call site on TileLang, the backbone attention
    # on the torch chain.
    reference_plan={},
    # The gate/up GEMM's 16.8 MB of weights delivered cold by the copy engine
    # at this call site's own geometry (tma_ring sweep Q, one node, one job):
    # the machine's number for the phase, used in place of the constants' rule.
    ceilings={"action_expert_norm_gated_ffn": Ceiling(us=9.41, tag="tma.bw.dev.burst", job=591174)},
)

__all__ = ["TARGET"]
