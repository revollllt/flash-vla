#!/usr/bin/env python3
"""Is cuBLAS wave-quantized at Pi0's projection shapes on a 170-SM part?

Several call sites sit at half the tensor ceiling, and the suspicion is tile
count rather than tile efficiency: 768 x 2048 output at a 128 x 128 tiling is
96 CTAs, which is one wave that leaves 44% of the part idle. If that is what is
happening, throughput should be flat as M grows until the tile count crosses
170, then step -- and a hand-written tiling chosen for this part could win. If
instead throughput rises smoothly with M, the tiles are already small enough
and there is nothing for a different tiling to recover.
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


def sweep(label, k, n, ms):
    print(f"\n  {label}:  K={k} N={n}   [mma.tflops.dev.bf16] = 253")
    print(f"  {'M':>6s} {'us':>8s} {'TFLOP/s':>9s} {'% peak':>7s} "
          f"{'128x128 tiles':>14s} {'waves@170':>10s}")
    w = torch.randn(k, n, device=DEV, dtype=DT) * 0.02
    for m in ms:
        x = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
        out = torch.empty(m, n, device=DEV, dtype=DT)
        t = time_us(lambda: torch.mm(x, w, out=out))
        tf = 2 * m * k * n / t * 1e-6
        tiles = ((m + 127) // 128) * ((n + 127) // 128)
        print(f"  {m:6d} {t:8.2f} {tf:9.1f} {tf / 253 * 100:6.1f}% "
              f"{tiles:14d} {tiles / 170:10.2f}")


def main() -> int:
    torch.manual_seed(0)
    # llm_backbone_out_proj_residual: 768 x 2048 x 2048, measured at 51% of peak.
    sweep("backbone out-proj", 2048, 2048, [256, 512, 768, 1024, 1536, 2048, 4096])
    # vision_encoder_out_proj_residual: 768 x 1152 x 1152, at ~56%.
    sweep("vision out-proj", 1152, 1152, [256, 512, 768, 1024, 1536, 2048, 4096])
    # vision_encoder_ffn_down_residual: 768 x 4304 x 1152, at ~60%.
    sweep("vision ffn-down", 4304, 1152, [256, 512, 768, 1024, 1536, 2048])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
