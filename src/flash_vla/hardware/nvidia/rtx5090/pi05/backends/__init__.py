"""Backend registry for Pi0.5 on the RTX 5090.

One backend: `torch`, every call site in plain torch. Both plans point at it
until kernels for this machine exist. H100/Pi0.5's six are not registered --
`wgmma` does not assemble for sm_120a and its TileLang tiles assume 227 KB of
shared memory against this part's 99 KB. See `../../measured/README.md`.
"""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import torch_ops as _torch

BACKENDS = {"torch": _torch}

REGISTRY = Registry(BACKENDS, default="torch")

__all__ = ["BACKENDS", "REGISTRY"]
