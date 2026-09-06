"""The one geometry mirror for the SigLIP vision encoder on H100.

Every kernel, wrapper and reference in this package reads its dimensions from
here. Two mirrors of one model spec drift, so there is exactly one: the named
constants come from `models.pi05.spec`, and `_check_pi0_agrees` re-derives them
from `models.pi0.spec`'s weight shapes at import so a divergence between the
two model specs fails loudly here rather than silently in a kernel compiled to
the wrong extent.

`VIEWS` is deliberately absent. It is a Target configuration (`num_views`,
default 3), not a model constant, so wrappers derive the GEMM row count from
the activation they are handed rather than from this module.
"""
from __future__ import annotations

from flash_vla.models.pi0 import spec as _pi0_spec
from flash_vla.models.pi05 import spec as _pi05_spec

#: Transformer layers in the vision tower. Vision runs at full depth in every
#: configuration -- the Targets' `--layers` bisection cuts the backbone only.
LAYERS: int = _pi05_spec.VISION_LAYERS

#: Model width, the K of three of the four GEMMs and the N of two.
DIM: int = _pi05_spec.VISION_DIM

#: Feed-forward width. 4304 = 2**4 * 269 with 269 prime, so it divides no
#: usable tile width: an N tiling of this axis always has a ragged tail.
FFN: int = _pi05_spec.VISION_FFN

#: Patches per view, 224/14 == 16 per side.
TOKENS: int = _pi05_spec.VISION_TOKENS

#: Attention heads and per-head width. Neither is named in either model spec;
#: both are fixed by the production attention wrapper, which reads the packed
#: QKV buffer as (-1, TOKENS, 3, 16, 72). HEAD_DIM 72 is not a multiple of the
#: wgmma K step of 16, which is why every tensor-core path over it pads to 80.
HEADS: int = 16
HEAD_DIM: int = DIM // HEADS

#: Columns of the packed QKV projection: Q|K|V blocks of DIM, head-major inside
#: a block.
QKV_DIM: int = 3 * DIM

#: LayerNorm epsilon, matching the value both Targets' shipped vision route
#: compiles into its kernel.
NORM_EPS: float = 1e-5


def _check_pi0_agrees() -> None:
    """Fail at import if the two model specs disagree about vision.

    A shared component package is only sound while both models really are the
    same shape here; this is the assertion that keeps that claim honest.
    """
    # prompt_len sizes only Pi0's prompt embedding; every vision weight is
    # independent of it, so the value here is arbitrary.
    shapes = _pi0_spec.weight_shapes(prompt_len=0)
    expected = {
        "vision_attn_qkv_w": (LAYERS, DIM, QKV_DIM),
        "vision_attn_o_w": (LAYERS, DIM, DIM),
        "vision_ffn_up_w": (LAYERS, DIM, FFN),
        "vision_ffn_down_w": (LAYERS, FFN, DIM),
        "vision_pre_attn_norm_w": (LAYERS, DIM),
        "vision_pre_ffn_norm_w": (LAYERS, DIM),
    }
    disagreements = {name: (shapes[name], want)
                     for name, want in expected.items() if shapes[name] != want}
    if disagreements:
        raise ValueError(
            "pi0 and pi05 disagree about the vision tower, so one shared SigLIP "
            f"package is unsound: {disagreements}")
    if HEADS * HEAD_DIM != DIM:
        raise ValueError(f"{HEADS} heads x {HEAD_DIM} != DIM {DIM}")


_check_pi0_agrees()

__all__ = ["DIM", "FFN", "HEADS", "HEAD_DIM", "LAYERS", "NORM_EPS", "QKV_DIM", "TOKENS"]
