"""Static addresses for a captured pipeline: the arena.

A Target's graph declares its buffers as data -- one `Buffer` per name, with
the allocation shape, dtype, initialization and the region the graph sees --
and `StaticArena` materializes them once, at addresses that never move for
the engine's lifetime. The runtime does not infer lifetimes, aliasing or
padding: the declaration is authoritative, and a kernel that needs the padded
allocation behind a view recovers it from the view's storage, which is why a
view is always cut from its own base allocation here rather than copied.

Backend workspaces (the split partials of FlashDecoding, a staged projection)
are not declared here: the runner hands every backend a workspace allocator
that allocates on first use and is frozen before capture (`runtime/runner.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Union

import torch

#: How a buffer starts: uninitialized, zero, a tensor of the allocation shape,
#: or a callable producing one on the target device.
Init = Union[str, torch.Tensor, Callable[[torch.device], torch.Tensor]]


@dataclass(frozen=True)
class Buffer:
    """One entry of a Target's buffer plan.

    `shape` is the allocation, padding included. `view` is the region the
    pipeline is handed, as a tuple of slices over the leading dims; `None`
    exposes the whole allocation. `init` is `"empty"`, `"zero"`, a tensor, or
    a callable of the device returning a tensor; a tensor or callable must
    match the allocation shape, so a value that lives in the padding (a mask
    bias on padded keys) is declared together with the buffer it pads.

    `alias` names another entry whose allocation this one views instead of
    owning its own: `shape`, `dtype` and `init` are then taken from that entry
    and `view` cuts the region. This is how a Target names a stage's contract
    region of a larger buffer (the prefix rows of a KV cache) without a copy.
    """
    shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.bfloat16
    init: Init = "empty"
    view: tuple[slice, ...] | None = None
    alias: str | None = None

    def exposed_shape(self) -> tuple[int, ...]:
        """The shape the pipeline sees."""
        if self.view is None:
            return tuple(self.shape)
        return tuple(len(range(*s.indices(n))) for s, n in zip(self.view, self.shape)) \
            + tuple(self.shape[len(self.view):])


class StaticArena:
    """The materialized buffer plan: base allocations and the views on them."""

    def __init__(self, plan: Mapping[str, Buffer], device) -> None:
        self.device = torch.device(device)
        self.plan = dict(plan)
        self.base: dict[str, torch.Tensor] = {}
        self.buffers: dict[str, torch.Tensor] = {}
        for name, spec in self.plan.items():
            if spec.alias is not None:
                continue
            base = self._materialize(name, spec)
            self.base[name] = base
            self.buffers[name] = base if spec.view is None else base[spec.view]
        for name, spec in self.plan.items():
            if spec.alias is None:
                continue
            if spec.alias not in self.base:
                raise KeyError(f"buffer {name!r} aliases {spec.alias!r}, which is not an "
                               "allocation in this plan")
            base = self.base[spec.alias]
            self.buffers[name] = base if spec.view is None else base[spec.view]

    def _materialize(self, name: str, spec: Buffer) -> torch.Tensor:
        shape = tuple(spec.shape)
        init = spec.init
        if init == "empty":
            return torch.empty(shape, dtype=spec.dtype, device=self.device)
        if init == "zero":
            return torch.zeros(shape, dtype=spec.dtype, device=self.device)
        value = init(self.device) if callable(init) else init
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"buffer {name!r}: init must be 'empty', 'zero', a tensor or a "
                            f"callable returning one, got {type(init).__name__}")
        if tuple(value.shape) != shape:
            raise ValueError(f"buffer {name!r}: init value has shape {tuple(value.shape)}, "
                             f"allocation is {shape}")
        base = torch.empty(shape, dtype=spec.dtype, device=self.device)
        base.copy_(value)
        return base

    @property
    def nbytes(self) -> int:
        """Bytes held by the base allocations."""
        return sum(t.numel() * t.element_size() for t in self.base.values())

    def allocation(self, name: str) -> torch.Tensor:
        """The base allocation behind buffer `name`, padding included."""
        spec = self.plan[name]
        return self.base[spec.alias if spec.alias is not None else name]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.buffers[name]

    def __contains__(self, name: str) -> bool:
        return name in self.buffers

    def __len__(self) -> int:
        return len(self.buffers)
