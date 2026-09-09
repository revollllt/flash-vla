"""LingBot upstream route with invariant RoPE frequencies cached at warmup."""
from __future__ import annotations

from . import upstream

ACTION_WEIGHT_PARAMS = upstream.ACTION_WEIGHT_PARAMS
BACKBONE_WEIGHT_PARAMS = upstream.BACKBONE_WEIGHT_PARAMS
NAMES = upstream.NAMES
OPS = upstream.OPS
ROUTE_CONSTRAINTS = upstream.ROUTE_CONSTRAINTS
WEIGHT_PARAMS = upstream.WEIGHT_PARAMS


def make_wrappers(scratch, selected_names=None):
    return upstream.make_wrappers(
        scratch, selected_names, cache_rope_frequency=True,
    )


__all__ = [
    "ACTION_WEIGHT_PARAMS", "BACKBONE_WEIGHT_PARAMS", "NAMES", "OPS",
    "ROUTE_CONSTRAINTS", "WEIGHT_PARAMS", "make_wrappers",
]
