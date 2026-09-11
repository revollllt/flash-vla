"""Remove the expert layer's redundant launches: packed GEMMs and grouped attention."""
from .upstream import NAMES, OPS, ROUTE_CONSTRAINTS, make_wrappers as _make_wrappers


def make_wrappers(scratch, selected_names=None):
    return _make_wrappers(scratch, selected_names, cache_rope_frequency=True,
                          linear_patch_embedding=True, cache_rope_tables=True,
                          precompute_time_modulation=True, fuse_norm=True,
                          pack_expert_projections=True, grouped_attention=True)
