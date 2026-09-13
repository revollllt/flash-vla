#!/usr/bin/env python3
"""Can a K split fill the wave cuBLAS leaves half empty at M=768?

The wave probe shows 768 x 2048 x 2048 and 768 x 4304 x 1152 taking the same
time as the same shapes at M=1024: both are one wave of 128x128 tiles, and at
M=768 that wave is 56% (or 32%) occupied. Splitting K into S independent GEMMs
and summing them multiplies the tile count by S without changing the arithmetic,
so the idle SMs get work -- provided the S GEMMs actually run at the same time.
A batched GEMM is the cheapest way to ask for that: one kernel, S times the
tiles. The sum of the partials is the price.
"""
from __future__ import annotations

import torch

DEV, DT = "cuda", torch.bfloat16
REPS, INNER = 20, 20


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


def probe(label, m, k, n, splits):
    x = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
    w = torch.randn(k, n, device=DEV, dtype=DT) * 0.02
    out = torch.empty(m, n, device=DEV, dtype=DT)
    gflop = 2 * m * k * n / 1e9
    base = time_us(lambda: torch.mm(x, w, out=out))
    ref = torch.mm(x, w)
    tiles = ((m + 127) // 128) * ((n + 127) // 128)
    print(f"\n  {label}: {m}x{k}x{n}, {gflop:.2f} GFLOP, "
          f"{tiles} tiles of 128x128 on 170 SMs")
    print(f"  {'splits':>6s} {'tiles':>6s} {'gemm':>8s} {'+sum':>8s} "
          f"{'total':>8s} {'TFLOP/s':>8s} {'vs mm':>6s}  cos")
    print(f"  {1:6d} {tiles:6d} {base:8.2f} {0.0:8.2f} {base:8.2f} "
          f"{gflop / base * 1e3:8.1f} {1.0:6.2f}")
    for s in splits:
        if k % s:
            continue
        kk = k // s
        a3 = x.as_strided((s, m, kk), (kk, k, 1))
        b3 = w.view(s, kk, n)
        part = torch.empty(s, m, n, device=DEV, dtype=DT)
        t_gemm = time_us(lambda: torch.bmm(a3, b3, out=part))
        t_all = time_us(lambda: torch.sum(torch.bmm(a3, b3, out=part), 0, out=out))
        got = torch.bmm(a3, b3).sum(0)
        cos = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), got.float().flatten(), dim=0).item()
        print(f"  {s:6d} {tiles * s:6d} {t_gemm:8.2f} {t_all - t_gemm:8.2f} "
              f"{t_all:8.2f} {gflop / t_all * 1e3:8.1f} {base / t_all:6.2f}  {cos:.6f}")


def main() -> int:
    torch.manual_seed(0)
    probe("backbone out-proj", 768, 2048, 2048, [2, 4])
    probe("vision ffn-down", 768, 4304, 1152, [2, 4])
    probe("vision out-proj", 768, 1152, 1152, [2, 3, 4])
    probe("vision ffn-up", 768, 1152, 4304, [2, 3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
