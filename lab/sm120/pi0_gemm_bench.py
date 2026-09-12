#!/usr/bin/env python3
"""The hand-written GEMM against cuBLAS at Pi0's projection shapes.

The wave probe says cuBLAS runs these one wave short of full: 768 x 2048 x 2048
is 96 tiles of 128 x 128 on a 170-SM part. This asks the first question that
matters -- whether a hand-written mainloop is even in the same class as cuBLAS
per tile -- before any retiling is attempted. If it is not, retiling cannot
save it.
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu

DEV, DT = "cuda", torch.bfloat16
PEAK = 253.0  # [mma.tflops.dev.bf16]


def time_us(fn, inner=20, reps=20):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    s = []
    for _ in range(reps):
        a.record()
        for _ in range(inner):
            fn()
        b.record(); b.synchronize()
        s.append(a.elapsed_time(b) * 1000.0 / inner)
    s.sort()
    return s[len(s) // 2]


CASES = [
    ("backbone out_proj", 768, 2048, 2048),
    ("vision out_proj", 768, 1152, 1152),
    ("vision ffn_down", 768, 4304, 1152),
    ("vision ffn_up", 768, 1152, 4304),
    ("vision qkv", 768, 1152, 3456),
    ("backbone gate/up", 768, 2048, 16384),
    ("backbone qkv", 768, 2048, 2560),
    ("backbone ffn_down", 768, 16384, 2048),
    ("vision ffn_up", 768, 1152, 4352),
]


def main() -> int:
    torch.manual_seed(0)
    print(f"  {'site':20s} {'shape':20s} {'cuBLAS':>9s} {'mine':>9s} {'ratio':>6s} "
          f"{'cuBLAS':>8s} {'mine':>8s}  cos")
    for name, m, k, n in CASES:
        if m % 64 or n % 64 or k % 32:
            print(f"  {name:20s} does not tile")
            continue
        a = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
        b = torch.randn(k, n, device=DEV, dtype=DT) * 0.02
        ref = torch.empty(m, n, device=DEV, dtype=DT)
        got = torch.empty(m, n, device=DEV, dtype=DT)
        torch.mm(a, b, out=ref)
        cu.tiled_gemm(a, b, got)
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), got.float().flatten(), dim=0).item()
        t_ref = time_us(lambda: torch.mm(a, b, out=ref))
        t_got = time_us(lambda: cu.tiled_gemm(a, b, got))
        gf = 2 * m * k * n / 1e9
        print(f"  {name:20s} {f'{m}x{k}x{n}':20s} {t_ref:8.2f}us {t_got:8.2f}us "
              f"{t_ref / t_got:6.2f} {gf / t_ref * 1e3:7.1f} {gf / t_got * 1e3:7.1f}"
              f"  {cos:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
