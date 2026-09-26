"""Pi0.5 vision attention: one flash-attention kernel over the packed QKV rows.

Each view's 256 tokens attend to each other, 16 heads of width 72, read in
place from the `(views, 256, 3 * 1152)` QKV buffer and written into the
`(views, 256, 1152)` output. The length is fixed at construction, so there is
nothing to plan: one CTA per `BLOCK_M` query rows of one view's head walks the
256 keys with an online softmax.

FlashInfer has no head width 72, and the tensor-core `dot` needs power-of-two
widths, so each head is read as its first 64 columns and a 16-column tail of
which the last 8 are masked to zero (so any width in (64, 80] fits): 11% more
MMA work than 72 columns, where padding to 128 would add 78%. Against the
deployed compiled `scaled_dot_product_attention`, 12.9 against 16.9 us per
layer at RoboDojo's three views and 7.8 against 16.5 at LIBERO's two
(`lab/pi05/attention_screen.py`); the tile is the one within 2% of the fastest
at both view counts (`lab/pi05/attention_tile_sweep.py`).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flash_vla.models.pi05.spec import VISION_HEAD_DIM, VISION_HEADS, VISION_TOKENS
from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch

NAMES = frozenset({"vision_encoder_attention"})
BLOCK_M = 64
BLOCK_N = 32
WARPS = 4
STAGES = 3


@triton.jit
def vision_attention(QKV, Out, scale, TOKENS: tl.constexpr, HEADS: tl.constexpr,
                     HEAD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Query rows `program_id(0)` of head `program_id(1) % HEADS` of view
    `program_id(1) // HEADS`, over all of that view's keys."""
    view = tl.program_id(1) // HEADS
    head = tl.program_id(1) % HEADS
    row_stride = 3 * HEADS * HEAD
    base = QKV + view * TOKENS * row_stride + head * HEAD
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    low = tl.arange(0, 64)
    high = 64 + tl.arange(0, 16)
    inside = high < HEAD
    q_low = tl.load(base + rows[:, None] * row_stride + low[None, :])
    q_high = tl.load(base + rows[:, None] * row_stride + high[None, :], mask=inside[None, :],
                     other=0.0)
    peak = tl.full((BLOCK_M,), -1.0e30, tl.float32)
    total = tl.zeros((BLOCK_M,), tl.float32)
    out_low = tl.zeros((BLOCK_M, 64), tl.float32)
    out_high = tl.zeros((BLOCK_M, 16), tl.float32)
    for start in tl.range(0, TOKENS, BLOCK_N):
        keys = base + (start + tl.arange(0, BLOCK_N))[:, None] * row_stride
        k_low = tl.load(keys + HEADS * HEAD + low[None, :])
        k_high = tl.load(keys + HEADS * HEAD + high[None, :], mask=inside[None, :], other=0.0)
        logits = (tl.dot(q_low, tl.trans(k_low)) + tl.dot(q_high, tl.trans(k_high))) * scale
        new_peak = tl.maximum(peak, tl.max(logits, 1))
        weights = tl.exp(logits - new_peak[:, None])
        rescale = tl.exp(peak - new_peak)
        total = total * rescale + tl.sum(weights, 1)
        probabilities = weights.to(tl.bfloat16)
        out_low = out_low * rescale[:, None] + tl.dot(
            probabilities, tl.load(keys + 2 * HEADS * HEAD + low[None, :]))
        out_high = out_high * rescale[:, None] + tl.dot(
            probabilities, tl.load(keys + 2 * HEADS * HEAD + high[None, :], mask=inside[None, :],
                                   other=0.0))
        peak = new_peak
    out = Out + view * TOKENS * HEADS * HEAD + rows[:, None] * HEADS * HEAD + head * HEAD
    tl.store(out + low[None, :], (out_low / total[:, None]).to(tl.bfloat16))
    tl.store(out + high[None, :], (out_high / total[:, None]).to(tl.bfloat16),
             mask=inside[None, :])


def vision_encoder_attention(QKV: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Multi-head self-attention over a packed QKV buffer, (views, 256, 3*1152), into `out` (views, 256, 1152)."""
    vision_attention[(VISION_TOKENS // BLOCK_M, QKV.shape[0] * VISION_HEADS)](
        QKV, out, VISION_HEAD_DIM ** -0.5, VISION_TOKENS, VISION_HEADS, VISION_HEAD_DIM,
        BLOCK_M, BLOCK_N, num_warps=WARPS, num_stages=STAGES)
    return out


def make_wrappers(scratch: Scratch, selected_names: frozenset[str] | None = None
                  ) -> dict[str, Wrapper]:
    wrappers = {"vision_encoder_attention": vision_encoder_attention}
    return {name: wrappers[name] for name in (NAMES if selected_names is None else selected_names)}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=NAMES, make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
