#!/usr/bin/env python3
"""Validate and time the hand-written expert pointwise kernels against torch.

Correctness first, then the local gain: a kernel that is wrong is not fast, and
a standalone ratio is enough to kill a candidate but not to ship one -- the
model-level A/B decides that.

Shapes are Pi0's action expert at its real sizes: 51 tokens, 1024 wide, 8 query
heads over one 256-wide KV head, 4096 feed-forward.
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu
from flash_vla.hardware.nvidia.rtx5090.pi0.backends import torch_ops as tt

DEV, DT = "cuda", torch.bfloat16
M, K, HEADS, HEAD_DIM, FFN = 51, 1024, 8, 256, 4096
REPS = 200


def metrics(a, b):
    a32, b32 = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a32, b32, dim=0).item()
    rel = (torch.linalg.vector_norm(a32 - b32)
           / torch.linalg.vector_norm(a32).clamp_min(1e-12)).item()
    return cos, rel


def time_us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(REPS):
        a.record(); fn(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def report(name, ref, got, t_torch, t_cuda):
    cos, rel = metrics(ref, got)
    ok = cos >= 0.999 and rel <= 2e-2
    print(f"  {name:16} {'PASS' if ok else '*** FAIL ***':12} cos {cos:.6f} "
          f"rel {rel:.2e} | torch {t_torch:7.2f} us  cuda {t_cuda:7.2f} us  "
          f"{t_torch / t_cuda:5.2f}x")
    return ok


def main() -> int:
    torch.manual_seed(0)
    ok = []

    # -- RMSNorm ------------------------------------------------------------
    x = torch.randn(M, K, device=DEV, dtype=DT) * 0.1
    out = torch.empty_like(x)
    ref = tt._rms(x)
    cu.rms_norm(x, out)
    torch.cuda.synchronize()
    ok.append(report("rms_norm", ref, out,
                     time_us(lambda: tt._rms(x)),
                     time_us(lambda: cu.rms_norm(x, out))))

    # -- RoPE scatter -------------------------------------------------------
    n = HEADS * HEAD_DIM + 2 * HEAD_DIM
    packed = torch.randn(M, n, device=DEV, dtype=DT) * 0.1
    rope = torch.randn(M, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    q = torch.zeros(M * HEADS, HEAD_DIM, device=DEV, dtype=DT)
    k = torch.zeros(M, HEAD_DIM, device=DEV, dtype=DT)
    v = torch.zeros(M, HEAD_DIM, device=DEV, dtype=DT)
    qr, kr, vr = (torch.zeros_like(t) for t in (q, k, v))
    tt._scatter_qkv(packed, rope, qr, kr, vr, HEAD_DIM, HEADS)
    cu.rope_scatter(packed, rope, q, k, v)
    torch.cuda.synchronize()
    ok.append(report("rope_scatter", torch.cat([qr.flatten(), kr.flatten(), vr.flatten()]),
                     torch.cat([q.flatten(), k.flatten(), v.flatten()]),
                     time_us(lambda: tt._scatter_qkv(packed, rope, qr, kr, vr, HEAD_DIM, HEADS)),
                     time_us(lambda: cu.rope_scatter(packed, rope, q, k, v))))

    # -- Gated activation ---------------------------------------------------
    gate = torch.randn(M, FFN, device=DEV, dtype=DT) * 0.1
    up = torch.randn(M, FFN, device=DEV, dtype=DT) * 0.1
    go = torch.empty_like(gate)
    gref = tt._gelu(gate) * up
    cu.gelu_mul(gate, up, go)
    torch.cuda.synchronize()
    ok.append(report("gelu_mul", gref, go,
                     time_us(lambda: tt._gelu(gate) * up),
                     time_us(lambda: cu.gelu_mul(gate, up, go))))

    print(f"\n{sum(ok)}/{len(ok)} correct")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
