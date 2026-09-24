"""How the Pi0.5 engine's buffers correspond to the reference's stages.

The engine runs Pi0.5 on converted, folded weights, in padded buffers laid
out for its kernels; the reference (`reference.py`) runs it on the official
checkpoint in OpenPI's layouts. `reference_outputs` feeds the reference the
observation the engine ran -- its images and noise, and the prompt its host
slot tokenized -- and `comparable` pairs every stage output the engine
declares (`definition.Pi05Model.stage_outputs`) with the reference's value in
the engine's layout, over the rows both define:

    vision_encoder_x   SigLIP's last hidden state before the final LayerNorm
    prefix_k, prefix_v each layer's prefix keys and values; keys in the
                       engine's adjacent-pair RoPE order (`pi0.openpi.pair_layout`); valid
                       rows only, since the engine zeroes padded prompt rows
                       that OpenPI embeds and masks
    actions            the denoised chunk
    suffix_k, suffix_v the expert's own keys and values at the last step
"""
from __future__ import annotations

from typing import Literal, Mapping

import torch

from ..pi0.openpi import pair_layout
from .reference import Pi05Outputs, load


def reference_outputs(weights: Mapping[str, torch.Tensor], inputs: Mapping[str, torch.Tensor],
                      buffers: Mapping[str, torch.Tensor], *, shape: Mapping[str, int], seed: int,
                      precision: Literal["bfloat16", "float32"] = "bfloat16") -> Pi05Outputs:
    """The reference on the engine's observation: `inputs` as the engine was
    given them, `buffers` after its forward (the prompt its host slot
    tokenized, which already carries the fixture), `shape` its shape numbers.
    The fixture `seed` selects nothing further. In `float32` the reference
    holds a float32 copy of the weights beside `weights`, about twice their memory."""
    images = inputs["images"]
    return load(weights, precision=precision)(
        images.float(), image_masks=torch.ones(images.shape[0], dtype=torch.bool, device=images.device),
        prompt_ids=buffers["prompt_ids"].long(), prompt_mask=buffers["prompt_scale"][:, 0] > 0,
        noise=inputs["noise"].float(), steps=shape["steps"], depth=shape["layers"])


def comparable(outputs: Pi05Outputs,
               buffers: Mapping[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Each declared stage output: (reference, engine), in the engine's layout."""
    valid = outputs.prefix.pad_masks[0]
    depth = len(outputs.prefix.cache)
    prefix_keys = torch.stack([pair_layout(key[0, 0]) for key, _ in outputs.prefix.cache])
    prefix_values = torch.stack([value[0, 0] for _, value in outputs.prefix.cache])
    suffix_keys = torch.stack([pair_layout(key[0, 0]) for key, _ in outputs.suffix_cache])
    suffix_values = torch.stack([value[0, 0] for _, value in outputs.suffix_cache])
    return {
        "vision_encoder_x": (outputs.vision_hidden, buffers["vision_encoder_x"]),
        "prefix_k": (prefix_keys[:, valid], buffers["prefix_k"][:depth, valid]),
        "prefix_v": (prefix_values[:, valid], buffers["prefix_v"][:depth, valid]),
        "actions": (outputs.actions, buffers["actions"]),
        "suffix_k": (suffix_keys, buffers["suffix_k"][:depth]),
        "suffix_v": (suffix_values, buffers["suffix_v"][:depth]),
    }


__all__ = ["comparable", "reference_outputs"]
