"""Streaming loader for the frozen LingBot safetensors checkpoint."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from .spec import WEIGHT_SHAPES


class LingBotCheckpoint(Mapping[str, torch.Tensor]):
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.path = path / "model.safetensors" if path.is_dir() else path
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with safe_open(self.path, framework="pt", device="cpu") as source:
            keys = set(source.keys())
            expected = set(WEIGHT_SHAPES)
            if keys != expected:
                raise ValueError(
                    f"checkpoint keys differ: missing={sorted(expected - keys)}, "
                    f"extra={sorted(keys - expected)}"
                )
            for name, shape in WEIGHT_SHAPES.items():
                if tuple(source.get_slice(name).get_shape()) != shape:
                    raise ValueError(
                        f"{name} has shape {tuple(source.get_slice(name).get_shape())}, "
                        f"expected {shape}"
                    )

    def __len__(self) -> int:
        return len(WEIGHT_SHAPES)

    def __iter__(self) -> Iterator[str]:
        return iter(WEIGHT_SHAPES)

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in WEIGHT_SHAPES:
            raise KeyError(name)
        with safe_open(self.path, framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    def copy_into(self, weights: Mapping[str, torch.Tensor]) -> None:
        if set(weights) != set(WEIGHT_SHAPES):
            raise ValueError("runner weight schema differs from the frozen LingBot schema")
        with safe_open(self.path, framework="pt", device="cpu") as source:
            for name in WEIGHT_SHAPES:
                weights[name].copy_(source.get_tensor(name))


def load_checkpoint(path: str | Path) -> LingBotCheckpoint:
    return LingBotCheckpoint(path)


__all__ = ["LingBotCheckpoint", "load_checkpoint"]
