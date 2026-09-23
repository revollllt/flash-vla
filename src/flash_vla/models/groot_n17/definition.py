"""GR00T N1.7 LIBERO as a model definition: identity, configuration, shapes, inputs.

The workload is fixed by the checkpoint: four denoising steps, sixteen backbone
layers and two 256x256 views. Inputs come from a prepared fixture (the
`fixture` asset); only the noise is drawn, from the seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from flash_vla.runtime.graph import Graph
from flash_vla.runtime.vla import CheckpointReader, ConfigValue, Input, ModelDefinition

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
    PATCHES,
    SEQUENCE_LENGTH,
    STATE_DIM,
    STEPS,
    VIEWS,
    VISUAL_TOKENS,
)
from .weights import weight_shapes


@dataclass(frozen=True)
class GrootConfig:
    sequence_length: int = SEQUENCE_LENGTH
    steps: int = STEPS
    layers: int = LAYERS

    def __post_init__(self) -> None:
        if self.steps != STEPS or self.layers != LAYERS:
            raise ValueError(f"This LIBERO workload runs {STEPS} denoising steps and "
                             f"{LAYERS} backbone layers")
        if self.sequence_length < VISUAL_TOKENS:
            raise ValueError(f"The sequence must contain all {VISUAL_TOKENS} visual tokens")


class GrootModel(ModelDefinition[GrootConfig, None]):
    """GR00T N1.7: a Qwen3-VL vision tower and backbone and a DiT flow-matching head."""

    name = "groot-n17"
    model_revision = MODEL_REVISION
    inference_signature = INFERENCE_SIGNATURE
    shape_axes = ("batch", "views", "image_height", "image_width", "sequence_length",
                  "visual_tokens", "state_dim", "action_dim", "chunk", "steps", "layers")
    inputs = (
        Input("pixel_values", lambda s: (PATCHES, PATCH_WIDTH), torch.bfloat16, "pixel_values"),
        Input("input_ids", lambda s: (1, s["sequence_length"]), torch.int64, "input_ids"),
        Input("attention_mask", lambda s: (1, s["sequence_length"]), torch.int64, "attention_mask"),
        Input("position_ids", lambda s: (3, 1, s["sequence_length"]), torch.int64, "position_ids"),
        Input("image_indices", lambda s: (VISUAL_TOKENS,), torch.int64, "image_indices"),
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
        return dict(batch=1, views=VIEWS, image_height=IMAGE_SIZE, image_width=IMAGE_SIZE,
                    sequence_length=config.sequence_length, visual_tokens=VISUAL_TOKENS,
                    state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk=CHUNK,
                    steps=config.steps, layers=config.layers)

    def weight_shapes(self, shape: Mapping[str, int]) -> Mapping[str, tuple[int, ...]]:
        return weight_shapes()

    def build(self, g: Graph, shape: Mapping[str, int]) -> None:
        graph.build(g, shape)

    def sample_inputs(self, shape: Mapping[str, int], seed: int, device: torch.device,
                      assets: Mapping[str, Path]) -> dict[str, torch.Tensor]:
        """The prepared fixture's observation with noise drawn from `seed`."""
        values = torch.load(assets["fixture"], map_location="cpu", weights_only=True)["inputs"]
        expected_grid = torch.tensor([[1, 16, 16]] * VIEWS)
        if not torch.equal(values["image_grid_thw"], expected_grid):
            raise ValueError(f"This workload requires {VIEWS} {IMAGE_SIZE}x{IMAGE_SIZE} views "
                             "after preprocessing")
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
