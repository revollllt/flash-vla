"""Building the candidate set a sweep runs over.

Pure functions, no torch and no device: a tuning space is a list of dicts, one
per candidate, and every entry in a dict is whatever the backend's `build` knows
how to consume. Nothing here knows what a tile is, what warp specialization is,
or which device is being tuned.

Hardware enters the space through the backend, not through this module. The
backend reads a device spec and decides three separate things, and it is worth
keeping them separate because they fail in different ways:

  axis existence   an axis collapses to a single value when the device cannot do
                   the thing at all -- e.g. TileLang's warp specialization
                   lowers `T.copy` to TMA, so without TMA the axis is just False
  axis range       an axis keeps its meaning but loses values, because a
                   candidate would not fit -- shared memory per block is the
                   usual binding limit, and it eliminates far more configs on a
                   smaller GPU than any missing feature does
  measurement      how many distinct input sets are needed before the reads are
                   actually cold, which follows from cache size

Only the first is a mask. The second is a predicate over a candidate, which
`sweep` takes as `feasible=` and evaluates before paying for a compile. The
third is `cold_n_inner` below.
"""
from __future__ import annotations

import itertools
import math


def grid(**axes) -> list[dict]:
    """Cartesian product of named axes: grid(BLOCK_M=[16, 32], NUM_STAGES=[2, 3]).

    An axis given a single value contributes no combinations but still appears
    in every candidate, which is how a collapsed axis stays visible in results
    and reports instead of silently disappearing.
    """
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*[axes[k] for k in keys])]


def cold_n_inner(footprint_bytes: int, cache_bytes: int, *, margin: float = 2.0,
                 minimum: int = 8) -> int:
    """Distinct input sets needed before `cache_bytes` stops serving the reads.

    `graph_time_cold` calls `invoke(i)` for i in range(n_inner). Those reads are
    only cold if the caller actually indexes into that many distinct buffers and
    their total exceeds the cache -- otherwise the second graph replay onward
    finds everything resident and the measurement quietly becomes a hot one.

    `footprint_bytes` is what one call reads that the next call will not reuse,
    which for these kernels is the weight, not the activations. `margin` covers
    imperfect replacement and competing traffic; 2x is a heuristic, not a
    derivation.

    A small footprint gives a large answer -- defeating a 50 MB L2 with a 64 KB
    weight takes over a thousand sets. That is a real result, not an error: it
    says this kernel cannot be measured cold at a sane capture size, and the
    caller has to decide whether to accept a warm number or restructure. The
    value is deliberately not capped, so the cost shows up rather than being
    silently truncated.
    """
    if footprint_bytes <= 0:
        raise ValueError(f"footprint_bytes must be positive, got {footprint_bytes}")
    if cache_bytes <= 0:
        raise ValueError(f"cache_bytes must be positive, got {cache_bytes}")
    required = math.ceil(cache_bytes * margin / footprint_bytes)
    return max(minimum, required)


__all__ = ["cold_n_inner", "grid"]
