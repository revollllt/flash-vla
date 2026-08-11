"""Pre-allocated scratch for the few operations that need a temporary.

Most of the pipeline writes straight into the caller's buffers, but a handful of
operations produce an intermediate -- the score matrix in the unfused attention,
the partial outputs in FlashDecoding, the projected QKV in the encoder. Nothing
inside a captured CUDA graph may allocate, so those come from here instead of
`torch.empty`.

Each (role, shape, dtype) is allocated once, zeroed, and reused. After warmup
the pool is frozen: a request for a shape that warmup did not cover then raises
instead of allocating during capture, which would otherwise fail in a far more
confusing way.
"""
from __future__ import annotations

import torch


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

    def __len__(self) -> int:
        return len(self._buffers)
