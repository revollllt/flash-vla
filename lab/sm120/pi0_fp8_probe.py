#!/usr/bin/env python3
"""What fp8 would buy on the GEMM-bound call sites, and what it would cost.

sm_120 runs fp8 `mma` at 1024 FLOP/cycle/SM against bf16's 512 -- measured, see
the hardware axis's measured/unit-mma.md -- so the ceiling is exactly twice.
This measures the delivered ratio through `torch._scaled_mm` at the deployed
shapes, and the relative error against an fp32 reference at two scaling
granularities, because the speed is only interesting next to what it spends.

Tensor-wise is the crudest recipe and row-wise is the finer one. On this
fixture they come out the same: the error is e4m3's three mantissa bits, not
the scaling granularity. A trained checkpoint with heavier outliers would
separate them; random weights do not.
"""
import torch
DEV = "cuda"
def time_us(fn, inner=20, reps=20):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); s=[]
    for _ in range(reps):
        a.record()
        for _ in range(inner): fn()
        b.record(); b.synchronize(); s.append(a.elapsed_time(b)*1000.0/inner)
    s.sort(); return s[len(s)//2]

fmax = torch.finfo(torch.float8_e4m3fn).max
CASES = [("backbone gate/up", 768, 2048, 16384),
         ("backbone ffn_down", 768, 16384, 2048),
         ("backbone out_proj", 768, 2048, 2048)]
print(f"  {'site':20s} {'bf16 us':>8s} {'relerr':>9s} | {'fp8 tensorwise':>15s} {'relerr':>9s} | "
      f"{'fp8 rowwise':>12s} {'relerr':>9s}")
for name, m, k, n in CASES:
    torch.manual_seed(0)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16) * 0.1
    w = torch.randn(k, n, device=DEV, dtype=torch.bfloat16) * 0.02
    out = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
    ref = x.float() @ w.float()
    def rel(v): return (torch.linalg.vector_norm(v.float()-ref)/torch.linalg.vector_norm(ref)).item()
    t_bf = time_us(lambda: torch.mm(x, w, out=out)); e_bf = rel(torch.mm(x, w))

    # tensorwise
    sx = (x.abs().max()/fmax).clamp_min(1e-12); sw = (w.abs().max()/fmax).clamp_min(1e-12)
    xq = (x.float()/sx).clamp(-fmax,fmax).to(torch.float8_e4m3fn)
    wq = (w.float()/sw).clamp(-fmax,fmax).to(torch.float8_e4m3fn)
    wqt = wq.t().contiguous().t()
    g1 = torch._scaled_mm(xq, wqt, scale_a=sx.float().squeeze(), scale_b=sw.float().squeeze(),
                          out_dtype=torch.bfloat16)
    t1 = time_us(lambda: torch._scaled_mm(xq, wqt, scale_a=sx.float().squeeze(),
                                          scale_b=sw.float().squeeze(), out_dtype=torch.bfloat16))
    # rowwise: one scale per row of A, one per column of B
    rx = (x.abs().amax(dim=1, keepdim=True).float()/fmax).clamp_min(1e-12)
    rw = (w.abs().amax(dim=0, keepdim=True).float()/fmax).clamp_min(1e-12)
    xq2 = (x.float()/rx).clamp(-fmax,fmax).to(torch.float8_e4m3fn)
    wq2 = (w.float()/rw).clamp(-fmax,fmax).to(torch.float8_e4m3fn)
    wq2t = wq2.t().contiguous().t()
    try:
        g2 = torch._scaled_mm(xq2, wq2t, scale_a=rx, scale_b=rw, out_dtype=torch.bfloat16)
        t2 = time_us(lambda: torch._scaled_mm(xq2, wq2t, scale_a=rx, scale_b=rw,
                                              out_dtype=torch.bfloat16))
        print(f"  {name:20s} {t_bf:7.2f}u {e_bf:9.2e} | {t1:14.2f}u {rel(g1):9.2e} | "
              f"{t2:11.2f}u {rel(g2):9.2e}")
    except Exception as e:
        print(f"  {name:20s} {t_bf:7.2f}u {e_bf:9.2e} | {t1:14.2f}u {rel(g1):9.2e} | "
              f"rowwise failed: {str(e)[:60]}")
