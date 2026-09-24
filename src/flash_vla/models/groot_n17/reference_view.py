"""How the GR00T engine's buffers correspond to the reference's stages.

The engine runs the reference's own modules on the runner's weights
(`backends.torch_ops`), so `reference_outputs` is simply the reference end to
end on the engine's observation, and each declared stage output pairs with
the reference's one to one:

    vision_embeddings  the final patch merger's visual tokens
    deepstack          the three DeepStack features, stacked
    backbone_features  the backbone's last hidden state
    actions            the chunk after the executed steps
    velocity_step_0    the first step's velocity
"""
from __future__ import annotations

from typing import Literal, Mapping

import torch

from .reference import GrootOutputs, load
from .spec import GRID


def reference_outputs(weights: Mapping[str, torch.Tensor], inputs: Mapping[str, torch.Tensor],
                      buffers: Mapping[str, torch.Tensor], *, shape: Mapping[str, int], seed: int,
                      precision: Literal["bfloat16", "float32"] = "bfloat16") -> GrootOutputs:
    """The reference on the engine's observation: `inputs` as the engine was
    given them (the fixture's, with the seed's noise), `shape` its shape
    numbers. `buffers` and `seed` are unused: nothing the reference needs is
    computed on the device, and the noise is already drawn. In `float32` the
    reference holds a float32 copy of the weights beside `weights`."""
    reference = load(weights, precision=precision)
    return reference(inputs["pixel_values"], grid=GRID, input_ids=inputs["input_ids"],
                     attention_mask=inputs["attention_mask"], position_ids=inputs["position_ids"],
                     image_indices=inputs["image_indices"], state=inputs["state"],
                     embodiment_id=inputs["embodiment_id"], noise=inputs["noise"],
                     steps=shape["steps"])


def comparable(outputs: GrootOutputs,
               buffers: Mapping[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Each declared stage output: (reference, engine)."""
    return {
        "vision_embeddings": (outputs.vision_embeddings, buffers["vision_embeddings"]),
        "deepstack": (torch.stack(outputs.deepstack), buffers["deepstack"]),
        "backbone_features": (outputs.backbone_features, buffers["backbone_features"]),
        "actions": (outputs.actions, buffers["actions"]),
        "velocity_step_0": (outputs.velocity_step_0, buffers["velocity_step_0"]),
    }


__all__ = ["comparable", "reference_outputs"]
