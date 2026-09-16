"""RTX 5090 / BF16 GR00T N1.7 LIBERO reference Target."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_vla.models.groot_n17.weights import CHECKPOINT_ID, FIXTURE_ID, weight_shapes
from flash_vla.runtime import Input, VLA
from . import pipeline
from .backends import REGISTRY


@dataclass(frozen=True)
class GrootConfig:
    sequence_length: int = 156
    steps: int = 4
    layers: int = 16

    def __post_init__(self):
        if self.steps != 4 or self.layers != 16:
            raise ValueError("This LIBERO Target runs 4 denoising steps and 16 backbone layers")
        if self.sequence_length < 128:
            raise ValueError("The sequence must contain all 128 visual tokens")


class GrootN17(VLA):
    name = "hardware/nvidia/rtx5090/groot_n17"
    hardware = "rtx5090-32gb"
    model = "groot-n17"
    model_revision = "groot-n17-libero-v1"
    inference_signature = "groot-n17-qwen3vl16-dit32-libero-bf16-v1"
    precision = "bf16"
    ASSETS = {"checkpoint": CHECKPOINT_ID, "fixture": FIXTURE_ID}
    shape_axes = ("batch", "views", "image_height", "image_width", "sequence_length",
                  "visual_tokens", "state_dim", "action_dim", "chunk", "steps", "layers")
    INPUTS = (
        Input("pixel_values", lambda s: (512, 1536), torch.bfloat16, "pixel_values"),
        Input("input_ids", lambda s: (1, s["sequence_length"]), torch.int64, "input_ids"),
        Input("attention_mask", lambda s: (1, s["sequence_length"]), torch.int64, "attention_mask"),
        Input("position_ids", lambda s: (3, 1, s["sequence_length"]), torch.int64, "position_ids"),
        Input("image_indices", lambda s: (128,), torch.int64, "image_indices"),
        Input("state", lambda s: (1, 1, 132), torch.bfloat16, "state"),
        Input("embodiment_id", lambda s: (1,), torch.int64, "embodiment_id"),
        Input("noise", lambda s: (1, 40, 132), torch.bfloat16, "noise"),
    )
    STAGE_OUTPUTS = {
        "vision_encoder": (("vision_embeddings", None), ("deepstack", None)),
        "llm_backbone": (("backbone_features", None),),
        "action_expert": (("actions", None), ("velocity_step_0", None)),
    }
    registry = REGISTRY
    plan = reference_plan = {}

    def configure(self, **config):
        return GrootConfig(**config)

    def shape(self, config, checkpoint=None):
        return dict(batch=1, views=2, image_height=256, image_width=256,
                    sequence_length=config.sequence_length, visual_tokens=128,
                    state_dim=132, action_dim=132, chunk=40, steps=config.steps, layers=config.layers)

    def weight_shapes(self, shape):
        return weight_shapes()

    def checkpoint_shapes(self, checkpoint):
        return checkpoint.shapes

    def load_weights(self, checkpoint, weights):
        checkpoint.copy_into(weights)

    def build(self, graph, shape):
        pipeline.build(graph, shape)

    def sample_inputs(self, shape, seed, device, *, assets):
        values = torch.load(assets["fixture"], map_location="cpu", weights_only=True)["inputs"]
        expected_grid = torch.tensor([[1, 16, 16], [1, 16, 16]])
        if not torch.equal(values["image_grid_thw"], expected_grid):
            raise ValueError("This Target requires two 256x256 views after preprocessing")
        inputs = {inp.name: values[inp.name].to(device=device) for inp in self.INPUTS if inp.name != "noise"}
        for inp in self.INPUTS[:-1]:
            if inputs[inp.name].shape != inp.dims(shape):
                raise ValueError(f"Fixture {inp.name} shape differs from the Target: {inputs[inp.name].shape}")
        inputs["noise"] = torch.randn((1, 40, 132), dtype=torch.bfloat16, device=device,
                                      generator=torch.Generator(device=device).manual_seed(seed))
        return inputs


TARGET = GrootN17()
