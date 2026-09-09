"""The H100 / Pi0 Target: the model contract and its computation graph.

`Pi0` declares what the runner needs and nothing that runs: the identity
axes, the configuration, the shape numbers, the weight schema and loader, the
backend registry with the one shipped plan and the reference plan, and the
graph (`pipeline.build`). The runner (`flash_vla.runtime.ModelRunner`) does
the rest. Pi0 has no host slot: its prompt is fixed at load time and every
input is a device copy.

`steps` and `layers` exist for bisection: shortening either keeps the graph
intact while cutting depth, which is how a reference check is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from flash_vla.models.pi0.spec import (
    ACTION_DIM,
    MODEL_REVISION,
    INFERENCE_SIGNATURE,
    DECODER_DIM,
    DECODER_FFN,
    DECODER_HEADS,
    ENCODER_DIM,
    ENCODER_FFN,
    ENCODER_LAYERS,
    HEAD_DIM,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    KV_HEADS,
    QKV_WIDTH,
    STATE_DIM,
    VISION_DIM,
    VISION_FFN,
    VISION_HEAD_DIM,
    VISION_HEADS,
    VISION_LAYERS,
    VISION_TOKENS,
    weight_shapes,
)
from flash_vla.runtime import VLA, Input
from flash_vla.runtime.graph import Graph

from . import pipeline
from .backends import REGISTRY


@dataclass(frozen=True)
class Pi0Config:
    """Construction inputs beyond the checkpoint and the plan."""
    num_views: int = 3
    chunk_size: int = 50
    steps: int = 10
    layers: int = ENCODER_LAYERS
    #: Rows of `language_embeds`; read from the checkpoint when not given.
    prompt_len: int | None = None


class Pi0(VLA):
    """Pi0 on H100 SXM5: three stages, no host slot, bf16."""

    name = "hardware/nvidia/h100/pi0"
    hardware = "h100-sxm5-80gb"
    model = "pi0"
    model_revision = MODEL_REVISION
    inference_signature = INFERENCE_SIGNATURE
    precision = "bf16"
    shape_axes = (
        "batch", "num_views", "image_height", "image_width", "image_channels",
        "visual_tokens_per_view", "visual_tokens", "vision_dim", "vision_ffn_dim",
        "vision_heads", "vision_head_dim", "vision_layers", "prompt_len", "prefix_len",
        "chunk", "expert_tokens", "state_dim", "action_dim", "steps", "layers",
        "encoder_dim", "encoder_ffn_dim", "query_heads", "kv_heads", "head_dim",
        "qkv_width", "expert_dim", "expert_ffn_dim",
    )

    INPUTS = (
        Input("images", lambda s: (s["num_views"], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
              torch.bfloat16, "images"),
        Input("state", lambda s: (STATE_DIM,), torch.bfloat16, "state"),
        Input("noise", lambda s: (s["chunk"], ACTION_DIM), torch.bfloat16, "actions"),
    )

    registry = REGISTRY
    #: The shipped plan: the three action-expert fusions (lazy pre-norm on the
    #: gated FFN and the output projection, FlashDecoding attention), and the
    #: vision tower's two pre-norm projections and its attention on the shared
    #: SigLIP CUDA backend, and the two backbone call sites the shared Gemma
    #: component implements faster -- the prefix attention as one fused MQA
    #: kernel instead of a four-launch torch chain that also copied its result,
    #: and the output projection on cuBLAS. The FFN down projection stays on
    #: TileLang: cuBLAS measured slower there in the graph, where the hidden
    #: buffer is L2-resident.
    plan = {
        "llm_backbone_attention": "gemma-cuda",
        "llm_backbone_out_proj_residual": "gemma-cuda",
        "action_expert_norm_gated_ffn": "tilelang-fused",
        "action_expert_action_out_proj": "tilelang-fused",
        "action_expert_attention": "tilelang-fused",
        "vision_encoder_norm_qkv": "siglip-cuda",
        "vision_encoder_norm_ffn_up": "siglip-cuda",
        "vision_encoder_attention": "siglip-cuda",
    }
    #: The reference route: every call site on the unfused TileLang wrappers.
    reference_plan: Mapping[str, str] = {}

    def configure(self, **config: Any) -> Pi0Config:
        return Pi0Config(**config)

    def shape(self, config: Pi0Config, checkpoint=None) -> dict[str, int]:
        prompt_len = config.prompt_len
        if checkpoint is not None:
            from_checkpoint = len(checkpoint["language_embeds"])
            if prompt_len is not None and prompt_len != from_checkpoint:
                raise ValueError(f"prompt_len={prompt_len} but the checkpoint's language_embeds "
                                 f"has {from_checkpoint} rows")
            prompt_len = from_checkpoint
        if prompt_len is None:
            raise ValueError("Pi0 needs prompt_len, from the checkpoint or the configuration")
        visual_tokens = config.num_views * VISION_TOKENS
        return {
            "batch": 1,
            "num_views": config.num_views,
            "image_height": IMAGE_SIZE,
            "image_width": IMAGE_SIZE,
            "image_channels": IMAGE_CHANNELS,
            "visual_tokens_per_view": VISION_TOKENS,
            "visual_tokens": visual_tokens,
            "vision_dim": VISION_DIM,
            "vision_ffn_dim": VISION_FFN,
            "vision_heads": VISION_HEADS,
            "vision_head_dim": VISION_HEAD_DIM,
            "vision_layers": VISION_LAYERS,
            "prompt_len": prompt_len,
            "prefix_len": visual_tokens + prompt_len,
            "chunk": config.chunk_size,
            "expert_tokens": config.chunk_size + 1,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "steps": config.steps,
            "layers": config.layers,
            "encoder_dim": ENCODER_DIM,
            "encoder_ffn_dim": ENCODER_FFN,
            "query_heads": DECODER_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_width": QKV_WIDTH,
            "expert_dim": DECODER_DIM,
            "expert_ffn_dim": DECODER_FFN,
        }

    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        return weight_shapes(shape["prompt_len"])

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        pipeline.build(g, shape)


TARGET = Pi0()

__all__ = ["Pi0", "Pi0Config", "TARGET"]
