"""LingBot candidate: GEMM for the already expanded vision patches."""
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
        linear_patch_embedding=True,
    )
