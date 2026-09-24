"""How the Pi0 engine's buffers correspond to the reference's stages.

The engine runs Pi0 on converted weights, its prompt embedded at load and its
flow schedule folded into the weights: ten Euler steps of dt = -0.1
(`spec.FLOW_STEPS`), of which a run with fewer `steps` executes the first
ones. `reference_outputs` therefore composes the reference's stages the same
way -- the prefix, then the first `steps` of the ten-step schedule -- on the
engine's observation (its images, state and noise, and the fixture's prompt,
`sources.fixture_prompt`). `comparable` pairs every stage output the engine
declares with the reference's value in the engine's layout:

    vision_encoder_x   SigLIP's last hidden state before the final LayerNorm
    prefix_k, prefix_v each layer's prefix keys and values; keys in the
                       engine's adjacent-pair RoPE order
    actions            the chunk after the executed steps
    suffix_k, suffix_v the expert's own keys and values at the last step: the
                       state token's row, then the chunk's
"""
from __future__ import annotations

from typing import Literal, Mapping

import torch

from .openpi import pair_layout
from .reference import Pi0Outputs, load
from .sources import fixture_prompt
from .spec import FLOW_STEPS


def reference_outputs(weights: Mapping[str, torch.Tensor], inputs: Mapping[str, torch.Tensor],
                      buffers: Mapping[str, torch.Tensor], *, shape: Mapping[str, int], seed: int,
                      precision: Literal["bfloat16", "float32"] = "bfloat16") -> Pi0Outputs:
    """The reference on the engine's observation: `inputs` as the engine was
    given them, `shape` its shape numbers, `seed` the fixture's (which draws
    the prompt). `buffers` is unused: nothing the reference needs is computed
    on the device. In `float32` the reference holds a float32 copy of the
    weights beside `weights`, about twice their memory."""
    images, device = inputs["images"], inputs["images"].device
    reference = load(weights, precision=precision)
    prompt_ids = fixture_prompt(seed, shape["prompt_len"]).to(device)
    vision_hidden, prefix = reference.prefix(
        images.float(), image_masks=torch.ones(images.shape[0], dtype=torch.bool, device=device),
        prompt_ids=prompt_ids, prompt_mask=torch.ones_like(prompt_ids, dtype=torch.bool),
        depth=shape["layers"])
    dt = torch.tensor(-1.0 / FLOW_STEPS, dtype=torch.float32, device=device)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    noisy = inputs["noise"].float()
    for _ in range(shape["steps"]):
        velocity, suffix_cache = reference.velocity(prefix, noisy, state=inputs["state"].float(),
                                                    time=time, depth=shape["layers"])
        noisy = noisy + dt * velocity
        time = time + dt
    return Pi0Outputs(vision_hidden=vision_hidden, prefix=prefix, actions=noisy,
                      suffix_cache=suffix_cache)


def comparable(outputs: Pi0Outputs,
               buffers: Mapping[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Each declared stage output: (reference, engine), in the engine's layout."""
    depth = len(outputs.prefix.cache)
    prefix_keys = torch.stack([pair_layout(key[0, 0]) for key, _ in outputs.prefix.cache])
    prefix_values = torch.stack([value[0, 0] for _, value in outputs.prefix.cache])
    suffix_keys = torch.stack([pair_layout(key[0, 0]) for key, _ in outputs.suffix_cache])
    suffix_values = torch.stack([value[0, 0] for _, value in outputs.suffix_cache])
    return {
        "vision_encoder_x": (outputs.vision_hidden, buffers["vision_encoder_x"]),
        "prefix_k": (prefix_keys, buffers["prefix_k"][:depth]),
        "prefix_v": (prefix_values, buffers["prefix_v"][:depth]),
        "actions": (outputs.actions, buffers["actions"]),
        "suffix_k": (suffix_keys, buffers["suffix_k"][:depth]),
        "suffix_v": (suffix_values, buffers["suffix_v"][:depth]),
    }


__all__ = ["comparable", "reference_outputs"]
