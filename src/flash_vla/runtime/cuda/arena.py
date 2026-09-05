"""Static addresses for a captured pipeline: the arena and the scratch pool.

A Target declares its buffer plan as data -- one `Buffer` per name, with the
allocation shape, dtype, initialization and the region the pipeline sees --
and `StaticArena` materializes it once, at addresses that never move for the
engine's lifetime. The runtime does not infer lifetimes, aliasing or padding:
the declaration is authoritative, and a kernel that needs the padded
allocation behind a view recovers it from the view's storage, which is why a
view is always cut from its own base allocation here rather than copied.

Most of the pipeline writes straight into those buffers, but a handful of
operations produce an intermediate -- the score matrix in the unfused
attention, the partial outputs in FlashDecoding, the projected QKV in the
encoder. Nothing inside a captured CUDA graph may allocate, so those come from
the `ScratchPool` instead of `torch.empty`. Each (role, shape, dtype) is
allocated once, zeroed, and reused. After warmup the pool is frozen: a request
for a shape that warmup did not cover then raises instead of allocating during
capture, which would otherwise fail in a far more confusing way.
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
    """
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16
    init: Init = "empty"
    view: tuple[slice, ...] | None = None

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
            base = self._materialize(name, spec)
            self.base[name] = base
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

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.buffers[name]

    def __contains__(self, name: str) -> bool:
        return name in self.buffers

    def __len__(self) -> int:
        return len(self.buffers)


class ScratchPool:
    def __init__(self) -> None:
        self._buffers: dict = {}
        self._frozen = False

    def get(self, role: str, shape, dtype, device) -> torch.Tensor:
        """Return the buffer for this (role, shape, dtype), allocating it on first use."""
        key = (role, tuple(shape), dtype, str(device))
        buffer = self._buffers.get(key)
        if buffer is None:
            if self._frozen:
                raise RuntimeError(
                    f"ScratchPool is frozen but {key} was requested: warmup did not cover this "
                    "shape, so it would allocate mid-capture.")
            buffer = torch.zeros(shape, dtype=dtype, device=device)
            self._buffers[key] = buffer
        return buffer

    def freeze(self) -> None:
        """Forbid further allocation; call after warmup and before graph capture."""
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        return len(self._buffers)
