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

from flash_vla.models.pi0.spec import ENCODER_LAYERS, weight_shapes
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
    precision = "bf16"

    INPUTS = (
        Input("images", lambda s: (s["num_views"], 224, 224, 3), torch.bfloat16, "images"),
        Input("state", lambda s: (32,), torch.bfloat16, "state"),
        Input("noise", lambda s: (s["chunk"], 32), torch.bfloat16, "actions"),
    )

    registry = REGISTRY
    #: The shipped plan: the three action-expert fusions (lazy pre-norm on the
    #: gated FFN and the output projection, FlashDecoding attention).
    plan = {
        "action_expert_norm_gated_ffn": "tilelang-fused",
        "action_expert_action_out_proj": "tilelang-fused",
        "action_expert_attention": "tilelang-fused",
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
        return {"num_views": config.num_views, "chunk": config.chunk_size,
                "steps": config.steps, "layers": config.layers, "prompt_len": prompt_len}

    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        return weight_shapes(shape["prompt_len"])

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        pipeline.build(g, shape)


TARGET = Pi0()

__all__ = ["Pi0", "Pi0Config", "TARGET"]
