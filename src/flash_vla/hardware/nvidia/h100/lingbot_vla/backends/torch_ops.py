"""LingBot's three stages as the model reference runs them, on the runner's buffers and weights.

Each op binds the reference's parts it runs to its own weights, without a
copy, on its first call (`official.bind_parts`): the vision op, which carries
every weight of the model, binds the vision tower; the prefix op the
backbone; the action op the expert and the heads. The prefix's keys and values
cross from the prefix to the action stage through the runner's buffers. The
vision tables, which the reference builds from host-side lists, are built
once, during warmup, into runner workspace. Plain PyTorch and no upstream
code: the Target's reference route, runnable on any CUDA device.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from flash_vla.models.lingbot.ops import CALL_SITES
from flash_vla.models.lingbot.reference import PREFIXES, Prefix, VisionTables, make_reference
from flash_vla.models.lingbot.spec import BACKBONE_WEIGHT_NAMES, EXPERT_WEIGHT_NAMES, GRID, WEIGHT_NAMES
from flash_vla.models.official import bind_parts
from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch


def make_wrappers(scratch: Scratch, selected_names: frozenset[str] | None = None) -> dict[str, Wrapper]:
    reference = make_reference()
    bound: set[str] = set()
    tables: VisionTables | None = None

    def bind_once(parts: tuple[str, ...], names: tuple[str, ...],
                  tensors: tuple[torch.Tensor, ...]) -> None:
        if bound.issuperset(parts):
            return
        bind_parts({part: reference.parts()[part] for part in parts}, prefixes=PREFIXES,
                   weights=dict(zip(names, tensors)), precision="bfloat16")
        bound.update(parts)

    def staged_tables() -> VisionTables:
        nonlocal tables
        if tables is not None:
            return tables
        built = reference.vision.tables(GRID)
        tables = replace(built, **{
            role: scratch(f"lingbot_reference_{role}", table.shape, table.dtype, table.device).copy_(table)
            for role, table in (("cos", built.cos), ("sin", built.sin),
                                ("window_order", built.window_order),
                                ("restore_order", built.restore_order))})
        return tables

    @torch.no_grad()
    def vision(pixel_values: torch.Tensor, out: torch.Tensor, layers: int,
               *weights: torch.Tensor) -> None:
        bind_once(("vision",), WEIGHT_NAMES, weights)
        out.copy_(reference.vision(pixel_values, tables=staged_tables()).view(out.shape))

    @torch.no_grad()
    def prefix(vision: torch.Tensor, image_masks: torch.Tensor, language_tokens: torch.Tensor,
               language_masks: torch.Tensor, prefix_masks: torch.Tensor, prefix_k: torch.Tensor,
               prefix_v: torch.Tensor, layers: int, *weights: torch.Tensor) -> None:
        bind_once(("backbone",), BACKBONE_WEIGHT_NAMES, weights)
        prefix_pass = reference.prefix(vision, image_masks=image_masks, language_tokens=language_tokens,
                                       language_masks=language_masks, depth=layers)
        prefix_masks.copy_(prefix_pass.pad_masks)
        for layer, (key, value) in enumerate(prefix_pass.cache):
            prefix_k[layer].copy_(key[0])
            prefix_v[layer].copy_(value[0])

    @torch.no_grad()
    def action(state: torch.Tensor, noise: torch.Tensor, prefix_masks: torch.Tensor,
               prefix_k: torch.Tensor, prefix_v: torch.Tensor, actions: torch.Tensor,
               velocity_step_0: torch.Tensor, steps: int, layers: int, *weights: torch.Tensor) -> None:
        bind_once(("expert", "head"), EXPERT_WEIGHT_NAMES, weights)
        cached = Prefix(cache=tuple((prefix_k[layer][None], prefix_v[layer][None])
                                    for layer in range(layers)), pad_masks=prefix_masks)
        denoised, velocity = reference.denoise(cached, noise, state=state, steps=steps)
        actions.copy_(denoised)
        velocity_step_0.copy_(velocity)

    wrappers = {"lingbot_vision": vision, "lingbot_prefix": prefix, "lingbot_action": action}
    return {name: wrappers[name] for name in (selected_names or CALL_SITES)}


#: The model reference itself: the Target's reference route.
BACKEND = Backend(names=CALL_SITES, make_wrappers=make_wrappers)

__all__ = ["BACKEND", "make_wrappers"]
