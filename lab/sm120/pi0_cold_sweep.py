#!/usr/bin/env python3
"""The expert's GEMMs measured with COLD weights, which is how they deploy.

Every tile sweep before this one reused one weight buffer, so after the first
iteration it sat in L2 and the kernel was measured against a cache. The action
expert does not run that way: it walks 18 layers, each with its own weights, ten
times a forward, so 94 MB of QKV weight alone passes through a cache that cannot
hold it and every read is cold.

This cycles a distinct weight per iteration to reproduce that, and re-picks the
tile under it. A configuration chosen L2-warm is chosen against the wrong
constraint -- warm, these shapes are latency-bound and want warps; cold, they
are bandwidth-bound and want requests in flight.
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import cutlass_gemm as cg
from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu

DEV, DT = "cuda", torch.bfloat16
#: Pi0's action expert walks this many layers before it revisits a weight.
LAYERS = 18
REPS = 30


def time_us(fns):
    """Median microseconds per call, cycling `fns` so no weight stays in L2.

    The list is always walked whole, so the warm and cold cases amortize the
    same number of launches. Timing one call between two events instead would
    charge it the whole launch latency and make the single-buffer case look
    slower than the cycling one, which is an artefact of the harness.
    """
    for f in fns:
        f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    s = []
    for _ in range(REPS):
        a.record()
        for f in fns:
            f()
        b.record(); b.synchronize()
        s.append(a.elapsed_time(b) * 1000.0 / len(fns))
    s.sort()
    return s[len(s) // 2]


def main() -> int:
    torch.manual_seed(0)
    ncfg = cg.library().cutlass_gemm_config_count()
    #: (label, M, K, N, calls, epilogue, distinct weights, config now routed)
    sites = [("expert gate|up packed", 51, 1024, 8192, 180, "plain", 18, 3),
             ("expert ffn_down", 51, 4096, 1024, 180, "residual", 18, 8),
             ("expert out_proj", 51, 2048, 1024, 180, "residual", 18, 9),
             ("vision qkv", 768, 1152, 3456, 27, "plain", 27, 5),
             ("vision out_proj", 768, 1152, 1152, 27, "residual", 27, 10),
             ("vision ffn_down", 768, 4304, 1152, 27, "residual", 27, 10),
             ("backbone qkv", 768, 2048, 2560, 18, "plain", 18, 10),
             ("backbone out_proj", 768, 2048, 2048, 17, "residual", 17, 10),
             ("backbone ffn_down", 768, 16384, 2048, 17, "residual", 17, 5)]
    print(f"  {'site':22s} {'shape':18s} {'now':>8s} {'best':>8s} {'cfg':>4s} "
          f"{'gain':>6s} {'ms/fwd':>8s}")
    total = 0.0
    for name, m, k, n, calls, kind, layers, routed in sites:
        a = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
        ws = [torch.randn(k, n, device=DEV, dtype=DT) * 0.02 for _ in range(layers)]
        d = torch.empty(m, n, device=DEV, dtype=DT)
        c = d if kind == "residual" else None
        beta = 1.0 if kind == "residual" else 0.0
        best, now = None, None
        for ci in range(ncfg):
            try:
                plans = [cg.plan(a, w, d, config=ci, c=c, beta=beta) for w in ws]
            except cg.Unsupported:
                continue
            if any(p is None for p in plans):
                continue
            cold = time_us([lambda p=p: cg.run(p) for p in plans])
            if ci == routed:
                now = cold
            if best is None or cold < best[0]:
                best = (cold, ci)
        if best is None or now is None:
            print(f"  {name:22s} no config")
            continue
        cold, ci = best
        delta = (cold - now) * calls / 1000.0
        total += delta
        mark = "" if ci == routed else "  <-- re-pick"
        print(f"  {name:22s} {f'{m}x{k}x{n}':18s} {now:7.2f}u {cold:7.2f}u {ci:4d} "
              f"{now / cold:6.2f} {delta:+8.3f}{mark}")
        del ws
        torch.cuda.empty_cache()

    print(f"  {'re-picking every tile cold':22s} {'':18s} {'':8s} {'':8s} {'':4s} "
          f"{'':6s} {total:+8.3f} ms/fwd")

    # The fused QKV kernel, same treatment.
    m, k, n, hd = 51, 1024, 2560, 256
    x = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
    ws = [torch.randn(k, n, device=DEV, dtype=DT) * 0.02 for _ in range(LAYERS)]
    rope = torch.randn(m, hd, device=DEV, dtype=DT)
    q = torch.empty(m * 8, hd, device=DEV, dtype=DT)
    kk = torch.empty(m, hd, device=DEV, dtype=DT)
    vv = torch.empty(m, hd, device=DEV, dtype=DT)
    warm = time_us([lambda: cu.expert_qkv(x, ws[0], rope, q, kk, vv)] * LAYERS)
    cold = time_us([lambda w=w: cu.expert_qkv(x, w, rope, q, kk, vv) for w in ws])
    print(f"  {'expert qkv (fused)':22s} {f'{m}x{k}x{n}':18s} "
          f"{LAYERS * k * n * 2 / 1e6:7.0f}M {warm:7.2f}u {cold:7.2f}u "
          f"{'':4s}  {cold / warm:.2f}x the warm number")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
