"""Fuse LingBot AdaRMS pointwise operations while retaining FP32 intermediates."""
import torch

from .upstream import NAMES, OPS, ROUTE_CONSTRAINTS, make_wrappers as _make_wrappers


@torch.compile(fullgraph=True)
def ada_rms(hidden_states, weight, gamma, beta, epsilon):
    """Return BF16 AdaRMS output; compilation occurs during eager warmup."""
    values = hidden_states.float()
    variance = values.pow(2).mean(-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + epsilon)
    normalized = weight * normalized
    return ((1 + gamma.float()) * normalized + beta.float()).to(hidden_states.dtype)


def make_wrappers(scratch, selected_names=None):
    return _make_wrappers(scratch, selected_names, cache_rope_frequency=True,
                          linear_patch_embedding=True, cache_rope_tables=True,
                          precompute_time_modulation=True, fuse_norm=True)
