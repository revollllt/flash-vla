"""`Scratch`: the workspace allocator the runner injects into every backend.

A backend's wrapper factory receives one `Scratch` and takes every piece of
device memory that outlives a single call from it, so the runner accounts for
it and forbids allocation once capture begins. `assets` is the runner's
read-only mapping of asset roles to local paths, for backends that initialize
from files.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

#: One workspace allocation: its role, shape, dtype and device.
ScratchKey = tuple[str, tuple[int, ...], torch.dtype, str]


class Scratch:
    """Workspace keyed by role, shape, dtype and device; frozen after warmup.

    Each key is allocated once, zero-filled, on first request and reused. The
    runner records which graph node asked (`current`). After warmup the
    allocator is frozen: a request warmup did not cover raises instead of
    allocating during graph capture.
    """

    def __init__(self, device: torch.device, *, assets: Mapping[str, Path] | None = None) -> None:
        self.device = device
        self.assets: Mapping[str, Path] = MappingProxyType(dict(assets or {}))
        self.allocations: dict[ScratchKey, torch.Tensor] = {}
        self.owners: dict[ScratchKey, int | None] = {}
        self.current: int | None = None
        self.frozen = False

    def __call__(self, role: str, shape: Sequence[int], dtype: torch.dtype,
                 device: torch.device | str) -> torch.Tensor:
        key: ScratchKey = (role, tuple(shape), dtype, str(device))
        if key in self.allocations:
            return self.allocations[key]
        if self.frozen:
            raise RuntimeError(f"workspace is frozen but {key} was requested: warmup did "
                               "not cover it, so it would allocate mid-capture")
        allocation = torch.zeros(tuple(shape), dtype=dtype, device=device)
        self.allocations[key] = allocation
        self.owners[key] = self.current
        return allocation

    def freeze(self) -> None:
        self.frozen = True

    @property
    def nbytes(self) -> int:
        return sum(allocation.numel() * allocation.element_size()
                   for allocation in self.allocations.values())

    def __len__(self) -> int:
        return len(self.allocations)


__all__ = ["Scratch"]
