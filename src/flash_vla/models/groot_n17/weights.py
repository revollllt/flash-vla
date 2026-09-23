"""Load the inference tensors from the two official LIBERO safetensors shards."""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path

from safetensors import safe_open
import torch

from flash_vla.runtime.vla import CheckpointReader

from .reference import PREFIXES, make_modules

CHECKPOINT_ID = "nvidia/GR00T-N1.7-LIBERO@2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21/libero_10"
FIXTURE_ID = "isaac-gr00t@51d4c89/libero-demo/episode-0/frame-0"


@lru_cache(maxsize=1)
def weight_shapes() -> dict[str, tuple[int, ...]]:
    return {PREFIXES[part] + name: tuple(tensor.shape)
            for part, module in make_modules().items()
            for name, tensor in module.state_dict().items()}


class Checkpoint(CheckpointReader, Mapping[str, torch.Tensor]):
    """The official LIBERO shards, validated on open and streamed shard by shard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        config = json.loads((self.path / "config.json").read_text())
        for key, value in {"model_type": "Gr00tN1d7", "select_layer": 16,
                           "action_horizon": 40, "use_alternate_vl_dit": True}.items():
            if config[key] != value:
                raise ValueError(f"This Target requires {key}={value!r}, got {config[key]!r}")
        self.files = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        # GR00T consumes hidden states, never language-generation logits.
        self.files = {name: file for name, file in self.files.items()
                      if name != "backbone.model.lm_head.weight"}
        self.shapes = {}
        for shard in sorted(set(self.files.values())):
            with safe_open(self.path / shard, framework="pt", device="cpu") as source:
                self.shapes.update({name: tuple(source.get_slice(name).get_shape())
                                    for name, file in self.files.items() if file == shard})

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return iter(self.files)

    def __getitem__(self, name):
        with safe_open(self.path / self.files[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    def copy_into(self, weights: Mapping[str, torch.Tensor]) -> None:
        for shard in sorted(set(self.files.values())):
            with safe_open(self.path / shard, framework="pt", device="cpu") as source:
                for name, file in self.files.items():
                    if file == shard:
                        weights[name].copy_(source.get_tensor(name))
