"""Binding a model reference to weights in the upstream checkpoint's own layout.

A model's `reference.py` is a set of plain torch modules, one per part of the
upstream model, whose own parameter names are the upstream checkpoint's names
below that part's prefix (`PREFIXES`): the vision tower, the language
backbone, the action expert, the heads. `official_schema` is therefore the
upstream checkpoint's schema, read off modules built on the meta device, and
`bind` puts an official state dict into those modules without a copy. Neither
reads a file: the caller owns the checkpoint.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


def official_schema(parts: Mapping[str, nn.Module], *,
                    prefixes: Mapping[str, str]) -> dict[str, tuple[int, ...]]:
    """Every official tensor name the parts hold, with its shape."""
    return {prefixes[part] + name: tuple(tensor.shape)
            for part, module in parts.items() for name, tensor in module.state_dict().items()}


def bind(parts: Mapping[str, nn.Module], *, prefixes: Mapping[str, str],
         weights: Mapping[str, torch.Tensor]) -> None:
    """Assign each part the official tensors below its prefix, strictly and without a copy.

    Upstream tensors the reference does not run (a tied language-model head,
    for instance) are simply not read.
    """
    for part, module in parts.items():
        module.load_state_dict({name: weights[prefixes[part] + name] for name in module.state_dict()},
                               strict=True, assign=True)


def random_official(parts: Mapping[str, nn.Module], *, prefixes: Mapping[str, str], seed: int,
                    device: str | torch.device, scale: float) -> dict[str, torch.Tensor]:
    """Seeded official-layout weights for the parts: every tensor of their
    schema, drawn in the order of the official names (so the draw does not
    depend on how a reference declares its modules), N(0, scale^2) rounded to
    bfloat16 as released checkpoints store them."""
    generator = torch.Generator(device=device).manual_seed(seed)
    schema = official_schema(parts, prefixes=prefixes)
    return {name: (torch.randn(schema[name], generator=generator, device=device) * scale
                   ).to(torch.bfloat16) for name in sorted(schema)}


__all__ = ["bind", "official_schema", "random_official"]
