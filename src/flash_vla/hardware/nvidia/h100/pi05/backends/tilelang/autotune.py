"""TileLang tuning adapter: the parts of a sweep that are not portable.

The sweep loop itself is in `flash_vla.tuning`. What stays here is everything
that only means something to TileLang -- `tilelang.jit`, `pass_configs`, the raw
builder registry -- plus the translation from a device capability to a tuning
axis, which is the piece people expect to be a hardware property and is not.

Warp specialization is the case that shows why. It is a TileLang lowering
strategy: `T.copy` becomes TMA plus a producer/consumer warp split. So the axis
exists only where TMA does, and `axes()` reads `SUPPORTS_TMA` to decide. But a
hand-written CUDA backend on the same H100 would use TMA directly with no such
flag, and on a device without TMA would reach for `cp.async` instead -- same
hardware fact, entirely different axis. That mapping belongs to a backend, which
is why `flash_vla.tuning` never sees it and `H100Spec` never hears the phrase
"warp specialization".

Whether the axis is worth sweeping is a separate question from whether it is
legal, and both matter: below one wave the producer warp has no work to hide and
costs warps plus mbarrier traffic, so small decoder GEMMs want it off while the
encoder and vision stages at M=768 want it on. It is also not universally legal
even on H100 -- the fused dual-GEMM gate reuses one shared tile across two
weight stages, which the no-WS pipeline planner rejects at compile time. Those
candidates are counted and skipped, not fatal.

Kernels are re-wrapped from `kernels.RAW_KERNELS` rather than reconfigured in
place: `.compile(pass_configs=...)` is rejected as unhashable, and a decorated
kernel does not expose its underlying builder.
"""
from __future__ import annotations

import tilelang

from flash_vla.hardware.nvidia.h100.spec import H100Spec
from flash_vla.tuning import grid, sweep

from .kernels import base as kernels


def rewrap(name: str, warp_spec: bool):
    """Re-wrap the raw builder for `name` with warp specialization on or off."""
    raw, out_idx = kernels.RAW_KERNELS[name]
    pass_configs = kernels.FAST_MATH if warp_spec else kernels.NO_WARP_SPEC
    if out_idx == "default":
        return tilelang.jit(raw, pass_configs=pass_configs)
    return tilelang.jit(raw, out_idx=out_idx, pass_configs=pass_configs)


def axes(spec=H100Spec, **tile_axes) -> list[dict]:
    """Candidate set for `spec`: the caller's tile axes crossed with warp specialization.

    Without TMA the axis collapses to a single False rather than disappearing,
    so it still shows up in every result row and in the report -- a config table
    should say the flag was considered and forced, not leave it unmentioned.

    `spec` is a parameter rather than a hardcoded import so the derivation can be
    read and tested against a different device without one existing yet. It is
    not a portability claim: a second device gets its own target directory, and
    the two adapters only merge if they turn out identical.
    """
    warp_spec = (True, False) if spec.SUPPORTS_TMA else (False,)
    return grid(warp_spec=warp_spec, **tile_axes)


def builder(name: str, const_kwargs: dict):
    """Return the `build` callable `tuning.sweep` needs for kernel `name`.

    Splits `warp_spec` back out of the candidate: it selects the wrapper, while
    everything else is a compile-time constant of the kernel itself.
    """
    def build(candidate: dict):
        config = dict(candidate)
        warp_spec = config.pop("warp_spec")
        return rewrap(name, warp_spec).compile(**const_kwargs, **config)

    return build


def sweep_kernel(name: str, const_kwargs: dict, tile_axes: dict, invoke, *, spec=H100Spec, **kwargs):
    """Sweep one kernel over its tile axes and the warp-specialization flag.

    const_kwargs   shape constants passed to every compile (M, N, K, HEAD_DIM, ...)
    tile_axes      tunable axes as lists: BLOCK_M=[16, 32], NUM_STAGES=[2, 3], ...
    invoke(k, i)   issue one launch of compiled kernel k against the i-th cold input set

    Remaining keyword arguments go to `tuning.sweep` -- `correct=` in particular,
    which a re-tune must not skip: `kernels.tl_scaled_gate` is numerically
    sensitive to its tiling and some tilings produce garbage rather than failing.
    """
    return sweep(axes(spec, **tile_axes), builder(name, const_kwargs), invoke,
                 label=name, **kwargs)
