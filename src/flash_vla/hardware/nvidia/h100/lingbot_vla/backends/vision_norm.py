"""Extend the aligned vision route with the single-launch RMSNorm."""
from .upstream import NAMES, OPS, ROUTE_CONSTRAINTS, make_wrappers as _make_wrappers


def make_wrappers(scratch, selected_names=None):
    return _make_wrappers(scratch, selected_names, cache_rope_frequency=True,
                          linear_patch_embedding=True, cache_rope_tables=True,
                          precompute_time_modulation=True, fuse_norm=True,
                          pack_expert_projections=True, grouped_attention=True,
                          specialized_loop=True, fused_rope=True, fused_attention=True,
                          specialized_prefix=True, pad_vision_ffn=True,
                          fused_vision_norm=True)
