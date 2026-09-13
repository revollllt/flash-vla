#!/usr/bin/env python3
"""Where the action expert's norm+QKV+RoPE call site spends its 12.21 us.

The floor model puts this site at 173% of a 7.05 us ceiling built from
[ld.bw.dev.dram] over its 5.37 MB. That ceiling assumes one kernel; the shipped
route is three. This times each of them at Pi0's real shape so the split
between bytes and per-kernel fixed cost is visible rather than assumed.
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu

DEV, DT = "cuda", torch.bfloat16
M, K, HEADS, HEAD_DIM = 51, 1024, 8, 256
N = HEADS * HEAD_DIM + 2 * HEAD_DIM
REPS = 300


#: Launches per event pair. One event around one launch measures that launch's
#: latency, which in a stream is [launch.lat.dev.ramp]'s 2.05 us and in the
#: deployed graph is 0.45 us -- so a per-call event overstates every small
#: kernel. Timing a run of back-to-back launches amortizes the enqueue the way
#: graph replay does, which is the number this site is being compared against.
INNER = 50


def time_us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    s = []
    for _ in range(REPS // 10):
        a.record()
        for _ in range(INNER):
            fn()
        b.record(); b.synchronize()
        s.append(a.elapsed_time(b) * 1000.0 / INNER)
    s.sort()
    return s[len(s) // 2]


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEV, dtype=DT) * 0.1
    w = torch.randn(K, N, device=DEV, dtype=DT) * 0.02
    rope = torch.randn(M, HEAD_DIM, device=DEV, dtype=DT)
    normed = torch.empty(M, K, device=DEV, dtype=DT)
    packed = torch.empty(M, N, device=DEV, dtype=DT)
    q = torch.empty(M * HEADS, HEAD_DIM, device=DEV, dtype=DT)
    k = torch.empty(M, HEAD_DIM, device=DEV, dtype=DT)
    v = torch.empty(M, HEAD_DIM, device=DEV, dtype=DT)

    bytes_w = K * N * 2
    bytes_all = x.numel() * 2 * 2 + bytes_w + packed.numel() * 2 * 2
    print(f"  shape {M}x{K}x{N}, weight {bytes_w / 1e6:.2f} MB, "
          f"site bytes {bytes_all / 1e6:.2f} MB")
    print(f"  [ld.bw.dev.dram] on the weight alone: "
          f"{3.35 + bytes_w / 1e6 / 1.524:.2f} us")

    t_norm = time_us(lambda: cu.rms_norm(x, normed))
    t_mm = time_us(lambda: torch.mm(normed, w, out=packed))
    t_rope = time_us(lambda: cu.rope_scatter(packed, rope, q, k, v))

    def chain():
        cu.rms_norm(x, normed)
        torch.mm(normed, w, out=packed)
        cu.rope_scatter(packed, rope, q, k, v)

    t_chain = time_us(chain)

    # The fused kernel, against the three-launch chain it would replace.
    fq = torch.empty_like(q)
    fk = torch.empty_like(k)
    fv = torch.empty_like(v)

    def fused():
        cu.expert_qkv(x, w, rope, fq, fk, fv)

    chain(); fused(); torch.cuda.synchronize()
    t_fused = time_us(fused)
    print(f"  {'rms_norm':16} {t_norm:7.2f} us   moves "
          f"{x.numel() * 4 / 1e6:.2f} MB")
    print(f"  {'mm (cuBLAS)':16} {t_mm:7.2f} us   moves "
          f"{bytes_w / 1e6:.2f} MB")
    print(f"  {'rope_scatter':16} {t_rope:7.2f} us   moves "
          f"{packed.numel() * 4 / 1e6:.2f} MB")
    print(f"  {'sum':16} {t_norm + t_mm + t_rope:7.2f} us")
    print(f"  {'chained':16} {t_chain:7.2f} us")
    print(f"  {'fused, one kernel':16} {t_fused:7.2f} us   "
          f"{t_chain / t_fused:5.2f}x vs the chain")
    ok = True
    for name, ref, got in (("Q", q, fq), ("K", k, fk), ("V", v, fv)):
        a32, b32 = ref.float().flatten(), got.float().flatten()
        cos = torch.nn.functional.cosine_similarity(a32, b32, dim=0).item()
        rel = (torch.linalg.vector_norm(a32 - b32)
               / torch.linalg.vector_norm(a32)).item()
        good = cos >= 0.999 and rel <= 2e-2
        ok &= good
        print(f"  {name}  {'PASS' if good else '*** FAIL ***':12} "
              f"cos {cos:.6f} rel {rel:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
