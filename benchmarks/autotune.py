"""Per-kernel config sweep: tiling x warp-specialization, at the real shape.

Two things make this different from a generic autotuner, and both were learned
the hard way:

Warp specialization is a tuning axis, not a fixed property. Below one wave the
producer warp has no work to hide and costs warps plus mbarrier traffic, so
small decoder GEMMs want it off; at high occupancy the encoder/vision GEMMs want
it on. It is also not universally legal -- the fused dual-GEMM gate rejects the
no-WS pipeline plan at compile time -- so compile failures are skipped, not
fatal.

Measurement uses `graph_time_cold`, not eager timing, and cycles distinct
weights so the reads are cold. Eager timing cannot resolve these kernels at all;
three production config bugs survived an eager benchmark and were only found
once the measurement moved into a CUDA graph.

Kernels are re-wrapped from `kernels.RAW_KERNELS` rather than reconfigured in
place: `.compile(pass_configs=...)` is rejected as unhashable, and a decorated
kernel does not expose its underlying builder.
"""
from __future__ import annotations

import itertools

import tilelang

from flash_vla.hardware.nvidia.h100.pi0.kernels import base as kernels
from .metrics import graph_time_cold


def rewrap(name: str, warp_spec: bool):
    """Re-wrap the raw builder for `name` with warp specialization on or off."""
    raw, out_idx = kernels.RAW_KERNELS[name]
    pass_configs = kernels.FAST_MATH if warp_spec else kernels.NO_WARP_SPEC
    if out_idx == "default":
        return tilelang.jit(raw, pass_configs=pass_configs)
    return tilelang.jit(raw, out_idx=out_idx, pass_configs=pass_configs)


def grid(**axes) -> list[dict]:
    """Cartesian product of named axes: grid(BLOCK_M=[16, 32], NUM_STAGES=[2, 3])."""
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*[axes[k] for k in keys])]


def sweep(name, const_kwargs, config_grid, invoke, correct=None, warp_spec_options=(True, False),
          n_inner=48, reps=40, verbose=True) -> list[dict]:
    """Time every (config x warp_spec) variant of one kernel, fastest first.

    const_kwargs   shape constants passed to every compile (M, N, K, HEAD_DIM, ...)
    config_grid    tunable compile kwargs (BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, THREADS)
    invoke(k, i)   issue one launch of compiled kernel k against the i-th cold input set
    correct(k)     optional bool check; configs that fail it are dropped, not reported
    """
    results, compile_failures, incorrect = [], 0, 0
    for warp_spec in warp_spec_options:
        try:
            factory = rewrap(name, warp_spec)
        except Exception:
            continue
        for config in config_grid:
            try:
                kernel = factory.compile(**const_kwargs, **config)
            except Exception:
                compile_failures += 1
                continue
            if correct is not None:
                try:
                    ok = correct(kernel)
                except Exception:
                    ok = False
                if not ok:
                    incorrect += 1
                    continue
            try:
                us = graph_time_cold(lambda i: invoke(kernel, i), n_inner=n_inner, reps=reps)
            except Exception:
                continue
            results.append({"us": round(us, 3), "ws": warp_spec, **config})
    results.sort(key=lambda r: r["us"])

    if verbose:
        print(f"[{name}] {len(results)} valid configs "
              f"({compile_failures} compile-fail, {incorrect} incorrect)")
        for r in results[:6]:
            config = {k: v for k, v in r.items() if k not in ("us", "ws")}
            print(f"    us={r['us']:8.3f}  ws={str(r['ws']):>5}  {config}")
        if results:
            per_ws = {w: min((r["us"] for r in results if r["ws"] == w), default=None)
                      for w in (True, False)}
            print(f"    BEST ws={results[0]['ws']} -> {results[0]['us']}us "
                  f"(best ws-on={per_ws[True]}, ws-off={per_ws[False]})")
    return results
