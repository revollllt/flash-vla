"""The H100/BF16 LingBot-VLA-4B Target."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file
import torch

from flash_vla.models.lingbot.spec import (
    ACTION_DIM,
    CHECKPOINT_REVISION,
    MODEL_REVISION,
    INFERENCE_SIGNATURE,
    BACKBONE_DIM,
    BACKBONE_FFN,
    CHUNK,
    EXPERT_DIM,
    EXPERT_FFN,
    HEAD_DIM,
    IMAGE_SIZE,
    KV_HEADS,
    LANGUAGE_SLOTS,
    LAYERS,
    PATCH_ROWS_PER_VIEW,
    PATCH_WIDTH,
    PREFIX_LEN,
    QUERY_HEADS,
    STATE_DIM,
    SUFFIX_LEN,
    VIEWS,
    VISION_DIM,
    VISION_FFN,
    VISION_HEAD_DIM,
    VISION_HEADS,
    VISION_LAYERS,
    VISUAL_TOKENS_PER_VIEW,
    WEIGHT_SHAPES,
)
from flash_vla.models.lingbot.weights import LingBotCheckpoint
from flash_vla.runtime import Input, VLA
from flash_vla.runtime.graph import Graph

from . import pipeline
from .backends import REGISTRY

@dataclass(frozen=True)
class LingBotConfig:
    steps: int = 10
    layers: int = LAYERS

    def __post_init__(self) -> None:
        if not 1 <= self.steps <= 10:
            raise ValueError("steps must be in [1, 10]")
        if not 1 <= self.layers <= LAYERS:
            raise ValueError(f"layers must be in [1, {LAYERS}]")


class LingBotVLA(VLA):
    ASSETS = {
        "checkpoint": CHECKPOINT_REVISION,
        "fixture": "lingbot-robotwin-canonical-v1/seed-42",
        "upstream": "lingbot-vla@4eb34b7693a0565c67433f8fac9c59a2e67eb60b",
        "qwen": "qwen2.5-vl-3b@66285546d2b821cf421d4f5eb2576359d3770cd3",
    }
    name = "hardware/nvidia/h100/lingbot_vla"
    hardware = "h100-sxm5-80gb"
    model = "lingbot-vla"
    model_revision = MODEL_REVISION
    inference_signature = INFERENCE_SIGNATURE
    precision = "bf16"
    shape_axes = (
        "batch", "views", "image_height", "image_width", "patch_rows_per_view",
        "patch_width", "visual_tokens_per_view", "visual_tokens", "vision_layers",
        "vision_dim", "vision_ffn_dim", "vision_heads", "vision_head_dim",
        "language_slots", "prefix_len", "state_dim", "action_dim", "chunk",
        "suffix_len", "steps", "layers", "backbone_dim", "backbone_ffn_dim",
        "expert_dim", "expert_ffn_dim", "query_heads", "kv_heads", "head_dim",
    )
    INPUTS = (
        Input("pixel_values", lambda s: (VIEWS, PATCH_ROWS_PER_VIEW, PATCH_WIDTH),
              torch.bfloat16, "pixel_values"),
        Input("image_masks", lambda s: (VIEWS,), torch.bool, "image_masks"),
        Input("language_tokens", lambda s: (1, LANGUAGE_SLOTS), torch.int64,
              "language_tokens"),
        Input("language_masks", lambda s: (1, LANGUAGE_SLOTS), torch.bool,
              "language_masks"),
        Input("state", lambda s: (1, STATE_DIM), torch.bfloat16, "state"),
        Input("noise", lambda s: (1, CHUNK, ACTION_DIM), torch.bfloat16, "noise"),
    )
    STAGE_OUTPUTS = {
        "vision_encoder": (("vision_embeddings", None),),
        "llm_backbone": (("prefix_k", 0), ("prefix_v", 0)),
        "action_expert": (("velocity_step_0", None), ("actions", None)),
    }
    registry = REGISTRY
    plan = {
        "lingbot_vision": "split-attention",
        "lingbot_prefix": "split-attention",
        "lingbot_action": "split-attention",
    }
    reference_plan: Mapping[str, str] = {}

    def configure(self, **config: Any) -> LingBotConfig:
        return LingBotConfig(**config)

    def shape(self, config: LingBotConfig, checkpoint=None) -> dict[str, int]:
        return {
            "batch": 1,
            "views": VIEWS,
            "image_height": IMAGE_SIZE,
            "image_width": IMAGE_SIZE,
            "patch_rows_per_view": PATCH_ROWS_PER_VIEW,
            "patch_width": PATCH_WIDTH,
            "visual_tokens_per_view": VISUAL_TOKENS_PER_VIEW,
            "visual_tokens": VIEWS * VISUAL_TOKENS_PER_VIEW,
            "vision_layers": VISION_LAYERS,
            "vision_dim": VISION_DIM,
            "vision_ffn_dim": VISION_FFN,
            "vision_heads": VISION_HEADS,
            "vision_head_dim": VISION_HEAD_DIM,
            "language_slots": LANGUAGE_SLOTS,
            "prefix_len": PREFIX_LEN,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "chunk": CHUNK,
            "suffix_len": SUFFIX_LEN,
            "steps": config.steps,
            "layers": config.layers,
            "backbone_dim": BACKBONE_DIM,
            "backbone_ffn_dim": BACKBONE_FFN,
            "expert_dim": EXPERT_DIM,
            "expert_ffn_dim": EXPERT_FFN,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
        }

    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        return WEIGHT_SHAPES

    def checkpoint_shapes(self, checkpoint):
        if isinstance(checkpoint, LingBotCheckpoint):
            return checkpoint.shapes
        return super().checkpoint_shapes(checkpoint)

    def load_weights(self, checkpoint: Mapping[str, torch.Tensor],
                     weights: Mapping[str, torch.Tensor]) -> None:
        if isinstance(checkpoint, LingBotCheckpoint):
            checkpoint.copy_into(weights)
            return
        super().load_weights(checkpoint, weights)

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        pipeline.build(g, shape)

    def sample_inputs(self, shape: Mapping[str, int], seed: int,
                      device, *, assets: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if seed != 42:
            raise ValueError("the frozen LingBot fixture uses seed 42")
        fixture = Path(assets["fixture"])
        values = load_file(fixture)
        return {inp.name: values[inp.name].to(device=device) for inp in self.INPUTS}


TARGET = LingBotVLA()

__all__ = ["LingBotConfig", "LingBotVLA", "TARGET"]
