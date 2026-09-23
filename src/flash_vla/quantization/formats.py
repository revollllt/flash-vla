"""Block-scaled activation formats and the scale layout Blackwell GEMMs read.

MXFP8 is E4M3 values with one UE8M0 scale per 32 elements along K; NVFP4 is
E2M1 values packed two per byte (the even element in the low nibble) with one
UE4M3 scale per 16 and an FP32 per-tensor encode scale. Scales are stored in
the 128x4 layout of CUTLASS's ``Sm1xxBlockScaledConfig``, which cuDNN and
FlashInfer (``get_sf_out_offset_128x4``) also use: rows padded to 128, scale
columns to 4, and each [128 row, 4 column] tile laid out as [32][4][4]. Padding
entries are never written by a producer and must stay zero, because a UE8M0
0xFF is NaN and the GEMM reads the padded columns of the last K tile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

BlockFormatName = Literal["mxfp8", "nvfp4"]


@dataclass(frozen=True)
class BlockFormat:
    name: BlockFormatName
    block: int        # elements along K that share one scale
    value_bits: int   # bits per stored value


MXFP8 = BlockFormat("mxfp8", 32, 8)
NVFP4 = BlockFormat("nvfp4", 16, 4)
FORMATS: dict[str, BlockFormat] = {MXFP8.name: MXFP8, NVFP4.name: NVFP4}


def scale_shape(rows: int, cols: int, fmt: BlockFormatName) -> tuple[int, int]:
    """(rows padded to 128, scale columns padded to 4) of the swizzled scale buffer."""
    block = FORMATS[fmt].block
    if cols % block:
        raise ValueError(f"{fmt} needs K divisible by {block}, got {cols}")
    return (rows + 127) // 128 * 128, (cols // block + 3) // 4 * 4


def swizzle_offsets(rows: int, blocks: int, device: torch.device | str) -> torch.Tensor:
    """Flat offset of scale (m, kb) in the 128x4 layout: int64 [rows, blocks]."""
    m = torch.arange(rows, device=device)[:, None]
    kb = torch.arange(blocks, device=device)[None, :]
    k_tiles = (blocks + 3) // 4
    return ((m // 128) * k_tiles * 512 + (kb // 4) * 512 + (m % 32) * 16
            + ((m % 128) // 32) * 4 + kb % 4)


def unswizzle(swizzled: torch.Tensor, rows: int, blocks: int) -> torch.Tensor:
    """Real (unpadded) entries of a flat swizzled scale buffer as row-major [rows, blocks]."""
    return swizzled.flatten()[swizzle_offsets(rows, blocks, swizzled.device)]
