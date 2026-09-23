"""The one geometry mirror for the SigLIP vision encoder on H100.

Every kernel, wrapper and reference in this package reads its dimensions from
here. Two mirrors of one model spec drift, so there is exactly one: the named
constants come from `models.pi05.spec`, and `tests/test_layering.py` checks
that `models.pi0.spec`'s vision weights have the same shapes, so a divergence
between the two model specs fails a test rather than silently compiling a
kernel to the wrong extent.

`VIEWS` is deliberately absent. It is a Target configuration (`num_views`,
default 3), not a model constant, so wrappers derive the GEMM row count from
the activation they are handed rather than from this module.
"""
from __future__ import annotations

from flash_vla.models.pi05 import spec as pi05_spec

#: Transformer layers in the vision tower. Vision runs at full depth in every
#: configuration -- the Targets' `--layers` bisection cuts the backbone only.
LAYERS: int = pi05_spec.VISION_LAYERS

#: Model width, the K of three of the four GEMMs and the N of two.
DIM: int = pi05_spec.VISION_DIM

#: Feed-forward width. 4304 = 2**4 * 269 with 269 prime, so it divides no
#: usable tile width: an N tiling of this axis always has a ragged tail.
FFN: int = pi05_spec.VISION_FFN

#: Patches per view, 224/14 == 16 per side.
TOKENS: int = pi05_spec.VISION_TOKENS

#: Attention heads and per-head width; the attention wrapper reads the packed
#: QKV buffer as (-1, TOKENS, 3, 16, 72). HEAD_DIM 72 is not a multiple of the
#: wgmma K step of 16, which is why every tensor-core path over it pads to 80.
HEADS: int = pi05_spec.VISION_HEADS
HEAD_DIM: int = pi05_spec.VISION_HEAD_DIM

#: Columns of the packed QKV projection: Q|K|V blocks of DIM, head-major inside
#: a block.
QKV_DIM: int = 3 * DIM

#: LayerNorm epsilon, matching the value both Targets' shipped vision route
#: compiles into its kernel.
NORM_EPS: float = 1e-5


__all__ = ["DIM", "FFN", "HEADS", "HEAD_DIM", "LAYERS", "NORM_EPS", "QKV_DIM", "TOKENS"]
