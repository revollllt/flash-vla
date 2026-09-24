"""How the LingBot engine's buffers correspond to the reference's stages.

The engine runs LingBot on the official layout itself -- nothing is converted
or folded -- so `reference_outputs` is the reference end to end on the
engine's observation, at the engine's `steps` and `layers`, and each declared
stage output pairs with the reference's directly:

    vision_embeddings  the merged visual tokens of each view
    prefix_k, prefix_v each executed layer's prefix keys and values, float32
    velocity_step_0    the first step's velocity
    actions            the chunk after the executed steps
"""
from __future__ import annotations

from typing import Literal, Mapping

import torch

from .reference import LingBotOutputs, load
from .spec import GRID


def reference_outputs(weights: Mapping[str, torch.Tensor], inputs: Mapping[str, torch.Tensor],
                      buffers: Mapping[str, torch.Tensor], *, shape: Mapping[str, int], seed: int,
                      precision: Literal["bfloat16", "float32"] = "bfloat16") -> LingBotOutputs:
    """The reference on the engine's observation: `inputs` as the engine was
    given them, `shape` its shape numbers. `buffers` and `seed` are unused:
    nothing the reference needs is computed on the device, and the inputs are
    already drawn. In `float32` the reference holds a float32 copy of the
    weights beside `weights`."""
    reference = load(weights, precision=precision)
    return reference(inputs["pixel_values"], grid=GRID, image_masks=inputs["image_masks"],
                     language_tokens=inputs["language_tokens"], language_masks=inputs["language_masks"],
                     state=inputs["state"], noise=inputs["noise"], steps=shape["steps"],
                     depth=shape["layers"])


def comparable(outputs: LingBotOutputs,
               buffers: Mapping[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Each declared stage output: (reference, engine)."""
    depth = len(outputs.prefix.cache)
    return {
        "vision_embeddings": (outputs.vision_embeddings, buffers["vision_embeddings"]),
        "prefix_k": (torch.stack([key[0] for key, _ in outputs.prefix.cache]), buffers["prefix_k"][:depth]),
        "prefix_v": (torch.stack([value[0] for _, value in outputs.prefix.cache]),
                     buffers["prefix_v"][:depth]),
        "velocity_step_0": (outputs.velocity_step_0, buffers["velocity_step_0"]),
        "actions": (outputs.actions, buffers["actions"]),
    }


__all__ = ["comparable", "reference_outputs"]
