"""`Scratch`: the workspace allocator the runner injects into every backend.

A backend's wrapper factory receives one `Scratch` and takes every piece of
device memory that outlives a single call from it, so the runner accounts for
it and forbids allocation once capture begins. `assets` is the runner's
read-only mapping of asset roles to local paths, for backends that initialize
from files. A backend whose kernel runs the exact replay-time length registers
a hook (`on_replay`) the runner calls with it before each replay.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import torch

#: One workspace allocation: its role, shape, dtype and device.
ScratchKey = tuple[str, tuple[int, ...], torch.dtype, str]
#: Called with the valid rows along the replay axis and the rows of their bucket.
ReplayHook = Callable[[int, int], None]


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
        self.replay_hooks: list[ReplayHook] = []
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

    def on_replay(self, hook: ReplayHook) -> None:
        """Have `hook(valid_rows, bucket_rows)` run on the host, on the caller's
        stream, before the stages the replay axis reaches replay; and before each
        bucket is warmed up and captured, with the bucket's rows as the valid ones.
        A kernel that runs the exact length plans it there, outside any capture."""
        self.replay_hooks.append(hook)

    def freeze(self) -> None:
        self.frozen = True

    @property
    def nbytes(self) -> int:
        return sum(allocation.numel() * allocation.element_size()
                   for allocation in self.allocations.values())

    def __len__(self) -> int:
        return len(self.allocations)


__all__ = ["Scratch"]
