"""Pi0 as a model definition: identity, configuration, shapes and inputs.

Pi0 has no host slot: its prompt is fixed at load time and every input is a
device copy. `steps` and `layers` exist for bisection: shortening either keeps
the graph intact while cutting depth, which is how a reference check is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from flash_vla.runtime.graph import Graph
from flash_vla.runtime.vla import CheckpointReader, ConfigValue, Input, ModelDefinition

from . import graph
from .ops import OPS
from .spec import (
    ACTION_DIM,
    DECODER_DIM,
    DECODER_FFN,
    DECODER_HEADS,
    ENCODER_DIM,
    ENCODER_FFN,
    ENCODER_LAYERS,
    HEAD_DIM,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    INFERENCE_SIGNATURE,
    KV_HEADS,
    MODEL_REVISION,
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


@dataclass(frozen=True)
class Pi0Config:
    """Construction inputs beyond the checkpoint and the plan."""
    num_views: int = 3
    chunk_size: int = 50
    steps: int = 10
    layers: int = ENCODER_LAYERS
    #: Rows of `language_embeds`; read from the checkpoint when not given.
    prompt_len: int | None = None


class Pi0Model(ModelDefinition[Pi0Config, None]):
    """Pi0: three stages, no host slot; the robot state enters as a token."""

    name = "pi0"
    model_revision = MODEL_REVISION
    inference_signature = INFERENCE_SIGNATURE
    shape_axes = (
        "batch", "num_views", "image_height", "image_width", "image_channels",
        "visual_tokens_per_view", "visual_tokens", "vision_dim", "vision_ffn_dim",
        "vision_heads", "vision_head_dim", "vision_layers", "prompt_len", "prefix_len",
        "chunk", "expert_tokens", "state_dim", "action_dim", "steps", "layers",
        "encoder_dim", "encoder_ffn_dim", "query_heads", "kv_heads", "head_dim",
        "qkv_width", "expert_dim", "expert_ffn_dim",
    )
    inputs = (
        Input("images", lambda s: (s["num_views"], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
              torch.bfloat16, "images"),
        Input("state", lambda s: (STATE_DIM,), torch.bfloat16, "state"),
        Input("noise", lambda s: (s["chunk"], ACTION_DIM), torch.bfloat16, "actions"),
    )
    ops = OPS

    def configure(self, **config: ConfigValue) -> Pi0Config:
        return Pi0Config(**config)

    def shape(self, config: Pi0Config, checkpoint: CheckpointReader | None) -> dict[str, int]:
        """The shape numbers; `prompt_len` is the checkpoint's `language_embeds` rows when
        a checkpoint is given, and the configuration's otherwise (then required)."""
        if checkpoint is None and config.prompt_len is None:
            raise ValueError("Pi0 needs prompt_len, from the checkpoint or the configuration")
        prompt_len = (config.prompt_len if checkpoint is None
                      else checkpoint.shapes["language_embeds"][0])
        if config.prompt_len is not None and prompt_len != config.prompt_len:
            raise ValueError(f"prompt_len={config.prompt_len} but the checkpoint's "
                             f"language_embeds has {prompt_len} rows")
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
        graph.build(g, shape)


__all__ = ["Pi0Config", "Pi0Model"]
