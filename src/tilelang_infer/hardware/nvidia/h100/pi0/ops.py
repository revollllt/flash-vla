"""The operation table the forward pass is written against.

`pipeline` calls operations by attribute (`ops.decoder_attention(...)`) rather
than importing them, so swapping the fused decoder for the unfused one is a
different table, not different control flow. The table is built once and passed
down explicitly -- no global dispatch state, which matters because a benchmark
routinely holds two configurations alive at the same time.
"""
from __future__ import annotations

from types import SimpleNamespace

from . import fused_wrappers, wrappers


def op_table(fused: bool = True) -> SimpleNamespace:
    """Build the operation table; `fused=True` overlays the fused decoder kernels."""
    table = dict(wrappers.ALL_WRAPPERS)
    if fused:
        table.update(fused_wrappers.FUSED_WRAPPERS)
    return SimpleNamespace(**table)


def op_names(fused: bool = True) -> list[str]:
    """Names in the table, for reporting which implementation is active."""
    return sorted(vars(op_table(fused)))
