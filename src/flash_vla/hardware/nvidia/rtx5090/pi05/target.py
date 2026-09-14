"""Pi0.5 on RTX 5090, with a torch reference and native pointwise fusions.

The graph and host slot are reused from the existing Pi0.5 model path;
backend routing and CUDA kernels belong to this hardware Target.
"""
from __future__ import annotations

from typing import Any, Mapping

from flash_vla.hardware.nvidia.h100.pi05.target import Pi05, forward_prefix, set_task

from .backends import REGISTRY


class Pi05RTX5090(Pi05):
    """Pi0.5 on one RTX 5090: three stages, one host slot, bf16.

    Identity is what makes this a separate Target rather than a flag. A
    measurement taken here must not be recorded against `h100-sxm5-80gb`: the
    two machines differ by 1.8x on streaming bandwidth, 3.4x on tensor-core
    throughput and 2.29x on shared memory per block, so a latency compared
    across them is not a comparison at all.
    """

    name = "hardware/nvidia/rtx5090/pi05"
    hardware = "rtx5090-32gb"

    registry = REGISTRY

    #: Native pointwise fusion around the existing bf16 GEMMs.
    plan: Mapping[str, str] = {
        "action_expert_norm_gated_ffn": "packed-ffn",
        "action_expert_norm_qkv_rope": "fused-qkv",
        "llm_backbone_norm_gated_ffn": "fused-backbone",
        "action_expert_attention": "fused-attention",
        "action_expert_out_proj_residual": "fused-residual",
        "action_expert_ffn_down_residual": "fused-residual",
    }
    reference_plan: Mapping[str, str] = {}

    #: H100's `action_expert_norm_gated_ffn` ceiling is a measured H100 number
    #: (`tma.bw.dev.burst`, job 591174) and says nothing about this part. Cleared
    #: rather than inherited; the floor model falls back to this machine's own
    #: constants until a `tma_ring`-equivalent sweep has been run here.
    CEILINGS: Mapping[str, Any] = {}


#: The Target instance the factory in `flash_vla.inference` hands the runner.
TARGET = Pi05RTX5090()

__all__ = ["TARGET", "Pi05RTX5090", "forward_prefix", "set_task"]
