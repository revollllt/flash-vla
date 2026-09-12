#!/usr/bin/env python3
"""Can the vision FFN's GELU ride in the GEMM epilogue instead of its own pass?

`vision_encoder_norm_ffn_up` is a LayerNorm, a GEMM and a GELU. The GELU reads
and writes the whole 768 x 4304 bf16 expansion -- 13.2 MB of traffic for one
elementwise function, which at [ld.bw.dev.dram] is 12 us of a 51.10 us call
site. cuBLASLt can apply GELU as a GEMM epilogue, which torch reaches through
`aten::_addmm_activation`; this checks that its GELU is the tanh approximation
Pi0 uses, and whether the epilogue costs the GEMM anything.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu

DEV, DT = "cuda", torch.bfloat16
M, K, N = 768, 1152, 4304
REPS, INNER = 30, 50


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
    x = torch.randn(M, K, device=DEV, dtype=DT) * 0.1
    w = torch.randn(K, N, device=DEV, dtype=DT) * 0.03
    bias = torch.randn(N, device=DEV, dtype=DT) * 0.1
    out = torch.empty(M, N, device=DEV, dtype=DT)
    ref = torch.empty_like(out)

    def split():
        torch.addmm(bias, x, w, beta=1, alpha=1, out=ref)
        cu.gelu_(ref)

    def fused():
        torch.ops.aten._addmm_activation.out(bias, x, w, beta=1, alpha=1,
                                             use_gelu=True, out=out)

    split(); fused(); torch.cuda.synchronize()
    a32, b32 = ref.float().flatten(), out.float().flatten()
    cos = F.cosine_similarity(a32, b32, dim=0).item()
    rel = (torch.linalg.vector_norm(a32 - b32)
           / torch.linalg.vector_norm(a32)).item()

    # Which GELU did cuBLASLt apply? Compare against both spellings in fp32.
    exact = torch.empty_like(ref)
    torch.addmm(bias, x, w, beta=1, alpha=1, out=exact)
    g_tanh = F.gelu(exact.float(), approximate="tanh")
    g_erf = F.gelu(exact.float(), approximate="none")
    d_tanh = (out.float() - g_tanh).abs().max().item()
    d_erf = (out.float() - g_erf).abs().max().item()

    ts, tf = time_us(split), time_us(fused)
    gflop = 2 * M * K * N / 1e9
    print(f"  shape {M}x{K}x{N}, {gflop:.2f} GFLOP, "
          f"expansion {M * N * 2 / 1e6:.2f} MB")
    print(f"  addmm + gelu_ pass    {ts:7.2f} us  {gflop / ts * 1e3:6.1f} TFLOP/s")
    print(f"  _addmm_activation     {tf:7.2f} us  {gflop / tf * 1e3:6.1f} TFLOP/s"
          f"  {ts / tf:5.2f}x")
    print(f"  agreement with the shipped pass: cos {cos:.6f} rel {rel:.2e}")
    print(f"  max |epilogue - gelu_tanh| {d_tanh:.3e}   "
          f"|epilogue - gelu_erf| {d_erf:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
