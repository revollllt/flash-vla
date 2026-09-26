"""The row buckets of a Pi0.5 prefix, as the bucketed BF16 and the MXFP8 backbone plan them.

The prefix is the image tokens, always valid, then the prompt slots, valid
from the first. The short bucket ends at the first 128-row tile boundary past
the image tokens: a prompt shorter than that boundary leaves every tile after
it padding. The full bucket is every prefix row. With three views the buckets
are M896/M968, with two M640/M712; every RoboDojo and LIBERO prompt fits the
short one (docs/workloads.md). The kernels run both plans and the prefix mask
at the short bucket's row selects one on the device.
"""
from __future__ import annotations

from typing import Mapping

#: Rows of the GEMM tile (`cutlass_backbone.cu`, `mxfp8_backbone.cu`): a bucket ends on one.
TILE_ROWS = 128


def bucket_rows(shape: Mapping[str, int]) -> tuple[int, int]:
    """The short and the full row bucket of a prefix."""
    return (shape["visual_tokens"] // TILE_ROWS + 1) * TILE_ROWS, shape["prefix_len"]


def has_short_bucket(shape: Mapping[str, int]) -> bool:
    """Whether the short bucket skips at least one tile of the prefix."""
    short_rows, full_rows = bucket_rows(shape)
    return short_rows < full_rows


__all__ = ["TILE_ROWS", "bucket_rows", "has_short_bucket"]
