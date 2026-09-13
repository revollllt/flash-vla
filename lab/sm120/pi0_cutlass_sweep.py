#!/usr/bin/env python3
"""Every GEMM call site in the route: cuBLAS against CUTLASS stream-K.

One row per site, at Pi0's real shape and with the epilogue that site actually
uses -- plain, a residual through beta, or a bias broadcast with a zero leading
dimension. The bar is parity: a site moves off cuBLAS when some config matches
it, because the point is as much control as speed. PDL needs the producer and
the consumer to be kernels this repo can put `griddepcontrol` in, and a cuBLAS
GEMM between two hand-written kernels breaks that chain.
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import cutlass_gemm as cg

DEV, DT = "cuda", torch.bfloat16
REPS, INNER = 20, 20

#: (label, M, K, N, calls per forward, epilogue)
#: epilogue: "plain" | "residual" (beta=1, C aliases D) | "bias" (broadcast row)
ALL_SITES = [
    ("vision qkv",            768,  1152,  3456,  27, "bias"),
    ("vision out_proj",       768,  1152,  1152,  27, "residual"),
    ("vision ffn_up",         768,  1152,  4304,  27, "bias"),
    ("vision ffn_down",       768,  4304,  1152,  27, "residual"),
    ("backbone qkv",          768,  2048,  2560,  18, "plain"),
    ("backbone out_proj",     768,  2048,  2048,  17, "residual"),
    ("backbone gate|up",      768,  2048, 16384,  34, "plain"),
    ("backbone ffn_down",     768, 16384,  2048,  17, "residual"),
    ("expert gate|up packed",  51,  1024,  8192, 180, "plain"),
    ("expert ffn_down",        51,  4096,  1024, 180, "residual"),
    ("expert out_proj",        51,  2048,  1024, 180, "residual"),
]
#: FLASH_VLA_SITES=expert limits the sweep while tuning one family.
import os as _os
_only = _os.environ.get("FLASH_VLA_SITES", "")
SITES = [s for s in ALL_SITES if _only in s[0]] if _only else ALL_SITES


def time_us(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    s = []
    for _ in range(REPS):
        a.record()
        for _ in range(INNER):
            fn()
        b.record(); b.synchronize()
        s.append(a.elapsed_time(b) * 1000.0 / INNER)
    s.sort()
    return s[len(s) // 2]


def main() -> int:
    torch.manual_seed(0)
    ncfg = cg.library().cutlass_gemm_config_count()
    print(f"  {'site':22s} {'shape':20s} {'cuBLAS':>8s} {'CUTLASS':>8s} "
          f"{'cfg':>4s} {'ratio':>6s} {'ms/fwd':>9s}  cos")
    total = 0.0
    picks = {}
    for name, m, k, n, calls, kind in SITES:
        a = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
        w = torch.randn(k, n, device=DEV, dtype=DT) * 0.02
        ref = torch.empty(m, n, device=DEV, dtype=DT)
        got = torch.empty(m, n, device=DEV, dtype=DT)
        res0 = torch.randn(m, n, device=DEV, dtype=DT) * 0.1
        bias = torch.randn(n, device=DEV, dtype=DT) * 0.1

        if kind == "plain":
            base = lambda: torch.mm(a, w, out=ref)
            kw = dict(beta=0.0)
        elif kind == "bias":
            base = lambda: torch.addmm(bias, a, w, beta=1, alpha=1, out=ref)
            kw = dict(c=bias, beta=1.0, broadcast_c=True)
        else:
            # The residual sites alias: C and D are the same buffer, and the
            # residual is already sitting in it when the site runs.
            ref.copy_(res0)
            base = lambda: torch.addmm(ref, a, w, beta=1, alpha=1, out=ref)
            kw = dict(beta=1.0)
        if kind == "residual":
            ref.copy_(res0)
        base(); torch.cuda.synchronize()
        gold = ref.clone().float()
        t_ref = time_us(base)

        best = None
        for ci in range(ncfg):
            # Plan once, then time the bare launch. That is what the deployed
            # path does -- the plan is built during warmup and the graph
            # replays only the launch -- and it keeps the dictionary lookup in
            # `gemm()` out of a measurement of an 8 us kernel.
            try:
                planned = cg.plan(a, w, got, config=ci,
                                  c=(got if kind == "residual" else kw.get("c")),
                                  beta=kw["beta"],
                                  broadcast_c=kw.get("broadcast_c", False))
            except cg.Unsupported:
                continue
            if planned is None:
                continue
            if kind == "residual":
                got.copy_(res0)
            cg.run(planned)
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                gold.flatten(), got.float().flatten(), dim=0).item()
            if cos < 0.999:
                continue
            t = time_us(lambda planned=planned: cg.run(planned))
            if best is None or t < best[0]:
                best = (t, ci, cos)
        if best is None:
            print(f"  {name:22s} {f'{m}x{k}x{n}':20s} {t_ref:7.2f}u   no config was correct")
            continue
        t, ci, cos = best
        delta = (t - t_ref) * calls / 1000.0
        total += delta
        picks[name] = ci
        flag = "" if t <= t_ref * 1.02 else "  <-- short of parity"
        print(f"  {name:22s} {f'{m}x{k}x{n}':20s} {t_ref:7.2f}u {t:7.2f}u "
              f"{ci:4d} {t_ref / t:6.2f} {delta:+8.3f}  {cos:.6f}{flag}")
    print(f"\n  whole route off cuBLAS: {total:+.3f} ms per forward (standalone,"
          f" L2-warm)")
    print(f"  picks = {picks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
