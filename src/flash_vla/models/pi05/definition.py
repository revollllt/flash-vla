"""Pi0.5 as a model definition: identity, configuration, shapes, inputs, host slot.

`Pi05Model` declares what a runner needs and nothing that runs; the graph is
`graph.build` and the prompt host slot is `prompt.PrefixInputs`. The model is
hardware-free: a Target composes it with a `Pi05Layout` (row padding, whether
the backbone's dense call sites take the prefix mask) and a device's backends.

`steps` and `layers` exist for bisection: shortening either keeps the graph
intact while cutting depth, which is how a reference check is read -- on
random weights a deep run diverges chaotically between any two
implementations that are not bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from flash_vla.runtime.graph import Graph
from flash_vla.runtime.vla import CheckpointReader, ConfigValue, Input, ModelDefinition

from . import graph
from .graph import Pi05Layout
from .ops import MASKED_OPS
from .prompt import PrefixInputs
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
    MAX_TOKEN_LEN,
    MODEL_REVISION,
    QKV_WIDTH,
    STATE_DIM,
    VISION_DIM,
    VISION_FFN,
    VISION_HEAD_DIM,
    VISION_HEADS,
    VISION_LAYERS,
    VISION_TOKENS,
    runtime_shapes,
)
from .tokenize import Pi05Tokenizer, TaskTokenizer


@dataclass(frozen=True)
class Pi05Config:
    """Construction inputs beyond the checkpoint, the plan and the assets."""
    num_views: int = 3
    chunk_size: int = 50
    steps: int = 10
    layers: int = ENCODER_LAYERS
    prompt_len: int = MAX_TOKEN_LEN
    #: The task string installed at construction; `session.set_task` changes it later.
    prompt: str | None = None
    #: Whether the prompt carries the discretized state (OpenPI's
    #: `discrete_state_input`); a task-only checkpoint such as pi05_libero sets False.
    discrete_state: bool = True


class Pi05Model(ModelDefinition[Pi05Config, PrefixInputs]):
    """Pi0.5: three stages and one host slot, `prompt`, which tokenizes the state.

    The tokenizer model is the runner's `tokenizer` asset (the PaliGemma
    SentencePiece model); `Pi05Config.discrete_state` chooses state-carrying or
    task-only prompts.
    """

    name = "pi05"
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
        # The state is tokenized on the host into the prompt buffers.
        Input("state", lambda s: (STATE_DIM,), torch.float32, None),
        Input("noise", lambda s: (s["chunk"], ACTION_DIM), torch.bfloat16, "actions"),
    )

    def __init__(self, layout: Pi05Layout) -> None:
        self.layout = layout
        self.ops = MASKED_OPS if layout.masked_backbone else ()

    def configure(self, **config: ConfigValue) -> Pi05Config:
        return Pi05Config(**config)

    def shape(self, config: Pi05Config, checkpoint: CheckpointReader | None) -> dict[str, int]:
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
            "prompt_len": config.prompt_len,
            "prefix_len": visual_tokens + config.prompt_len,
            "chunk": config.chunk_size,
            "expert_tokens": config.chunk_size,
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
        """The folded runtime schema (`weights.fold`), per flow step."""
        return runtime_shapes(shape["steps"])

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        graph.build(g, shape, self.layout)

    def host_state(self, config: Pi05Config, shape: Mapping[str, int],
                   assets: Mapping[str, Path]) -> PrefixInputs:
        tokenizer_type = Pi05Tokenizer if config.discrete_state else TaskTokenizer
        tokenizer = tokenizer_type(assets["tokenizer"], max_token_len=config.prompt_len)
        prefix = PrefixInputs(tokenizer, config.num_views, config.chunk_size)
        if config.prompt is None:
            return prefix
        tokenizer.set_task(config.prompt)
        return prefix

    def host(self, slot: str, host_state: PrefixInputs, buffers: Mapping[str, torch.Tensor],
             inputs: Mapping[str, torch.Tensor]) -> None:
        """`prompt`, the one slot: tokenize `inputs["state"]` and stage the prompt
        inputs; it runs while vision does."""
        host_state.build(inputs["state"])
        host_state.copy_into(buffers)


__all__ = ["Pi05Config", "Pi05Layout", "Pi05Model"]
