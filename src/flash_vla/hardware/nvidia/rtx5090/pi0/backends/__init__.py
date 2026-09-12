"""Backend registry for Pi0 on the RTX 5090.

One backend: `torch`, every call site in plain torch (`torch_ops.py`). It is the
bring-up route -- correct, unfused, untuned -- and both the shipped and the
reference plan point at it until hand-written kernels for this machine exist.

H100/Pi0's four other backends are not registered here, and will not be ported
as they stand:

    siglip-cuda     hand-written sm_90a CUDA whose GEMM mainloop is `wgmma`,
    gemma-cuda      which ptxas refuses on sm_120a entirely [isa.wgmma.absent]
    tilelang        tile shapes tuned against 227 KB of shared memory; ten of
    tilelang-fused  nineteen exceed this part's 99 KB [smem.bytes.cta.max] and
                    fail at launch rather than running slower

The architecture asks for a route customised per hardware, so the replacements
get designed against this machine's measured constants rather than re-tiled from
H100's -- see `../../measured/README.md`.
"""
from __future__ import annotations

from flash_vla.runtime.registry import Registry

from . import torch_ops as _torch

BACKENDS = {"torch": _torch}

REGISTRY = Registry(BACKENDS, default="torch")

__all__ = ["BACKENDS", "REGISTRY"]
