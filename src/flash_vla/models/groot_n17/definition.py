"""GR00T N1.7 as a model definition: identity, configuration, shapes, inputs.

The checkpoint fixes four denoising steps and sixteen backbone layers; the
workload fixes the number of 256x256 views and the text sequence. Inputs come
from a prepared fixture (the `fixture` asset); only the noise is drawn, from
the seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from flash_vla.runtime.graph import Graph
from flash_vla.runtime.vla import CheckpointReader, ConfigValue, Input, ModelDefinition, Workload

from . import graph
from .ops import OPS
from .spec import (
    ACTION_DIM,
    CHUNK,
    IMAGE_SIZE,
    INFERENCE_SIGNATURE,
    LAYERS,
    MODEL_REVISION,
    PATCH_WIDTH,
    PATCHES_PER_VIEW,
    STATE_DIM,
    STEPS,
    VIEW_GRID,
    VISUAL_TOKENS_PER_VIEW,
)
from .weights import weight_shapes


@dataclass(frozen=True)
class GrootConfig:
    #: Cameras and the text sequence (visual tokens included): the workload's.
    views: int
    sequence_length: int
    steps: int = STEPS
    layers: int = LAYERS

    def __post_init__(self) -> None:
        if self.steps != STEPS or self.layers != LAYERS:
            raise ValueError(f"The checkpoint runs {STEPS} denoising steps and "
                             f"{LAYERS} backbone layers")
        visual_tokens = self.views * VISUAL_TOKENS_PER_VIEW
        if self.sequence_length < visual_tokens:
            raise ValueError(f"The sequence must contain all {visual_tokens} visual tokens")


class GrootModel(ModelDefinition[GrootConfig, None]):
    """GR00T N1.7: a Qwen3-VL vision tower and backbone and a DiT flow-matching head."""

    name = "groot-n17"
    model_revision = MODEL_REVISION
    inference_signature = INFERENCE_SIGNATURE
    shape_axes = ("batch", "views", "image_height", "image_width", "sequence_length",
                  "visual_tokens", "state_dim", "action_dim", "chunk", "steps", "layers")
    workloads = (
        # RoboDojo: XPolicyLab `GR00T_N17` with `robodojo_arx_x5_config`, three
        # cameras; the sequence holds its longest task instruction (docs/workloads.md).
        Workload("robodojo", {"views": 3, "sequence_length": 237}),
        # The official GR00T-N1.7-LIBERO `libero_10` checkpoint's two cameras, at
        # the prepared fixture's sequence.
        Workload("libero", {"views": 2, "sequence_length": 156}),
    )
    inputs = (
        Input("pixel_values", lambda s: (s["views"] * PATCHES_PER_VIEW, PATCH_WIDTH),
              torch.bfloat16, "pixel_values"),
        Input("input_ids", lambda s: (1, s["sequence_length"]), torch.int64, "input_ids"),
        Input("attention_mask", lambda s: (1, s["sequence_length"]), torch.int64, "attention_mask"),
        Input("position_ids", lambda s: (3, 1, s["sequence_length"]), torch.int64, "position_ids"),
        Input("image_indices", lambda s: (s["visual_tokens"],), torch.int64, "image_indices"),
        Input("state", lambda s: (1, 1, STATE_DIM), torch.bfloat16, "state"),
        Input("embodiment_id", lambda s: (1,), torch.int64, "embodiment_id"),
        Input("noise", lambda s: (1, CHUNK, ACTION_DIM), torch.bfloat16, "noise"),
    )
    stage_outputs = {
        "vision_encoder": (("vision_embeddings", None), ("deepstack", None)),
        "llm_backbone": (("backbone_features", None),),
        "action_expert": (("actions", None), ("velocity_step_0", None)),
    }
    ops = OPS

    def configure(self, **config: ConfigValue) -> GrootConfig:
        return GrootConfig(**config)

    def shape(self, config: GrootConfig, checkpoint: CheckpointReader | None) -> dict[str, int]:
        return dict(batch=1, views=config.views, image_height=IMAGE_SIZE, image_width=IMAGE_SIZE,
                    sequence_length=config.sequence_length,
                    visual_tokens=config.views * VISUAL_TOKENS_PER_VIEW,
                    state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk=CHUNK,
                    steps=config.steps, layers=config.layers)

    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        return weight_shapes()

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        graph.build(g, shape)

    def sample_inputs(self, config: GrootConfig, shape: Mapping[str, int], seed: int,
                      device: torch.device, assets: Mapping[str, Path]) -> dict[str, torch.Tensor]:
        """The prepared fixture's observation with noise drawn from `seed`."""
        values = torch.load(assets["fixture"], map_location="cpu", weights_only=True)["inputs"]
        if not torch.equal(values["image_grid_thw"], torch.tensor((VIEW_GRID,) * shape["views"])):
            raise ValueError(f"This workload requires {shape['views']} {IMAGE_SIZE}x{IMAGE_SIZE} "
                             "views after preprocessing")
        dims = {spec.name: spec.dims(shape) for spec in self.inputs}
        observed = {name: values[name].to(device=device) for name in dims if name != "noise"}
        mismatched = {name: tuple(tensor.shape) for name, tensor in observed.items()
                      if tensor.shape != dims[name]}
        if mismatched:
            raise ValueError(f"Fixture shapes differ from the workload: {mismatched}")
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, CHUNK, ACTION_DIM), dtype=torch.bfloat16, device=device,
                            generator=generator)
        return {**observed, "noise": noise}


__all__ = ["GrootConfig", "GrootModel"]
