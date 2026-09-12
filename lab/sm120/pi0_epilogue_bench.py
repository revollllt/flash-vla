#!/usr/bin/env python3
"""Two pointwise passes the GEMM's epilogue can absorb.

Both of these are a launch and a full pass over the output that exist only
because the GEMM stopped one operation early:

  vision ffn_up      addmm(bias) then gelu_          ->  gelu(AB + bias)
  backbone gated ffn two GEMMs then gelu_mul         ->  gelu(gate) * up

The second works because the two GEMMs write the same tile coordinates, so the
gate projection's epilogue can read the up projection's output as its C operand
and finish the expression there.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import cutlass_gemm as cg
from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu

DEV, DT = "cuda", torch.bfloat16
REPS, INNER = 20, 20
CFG_LINEAR, CFG_GELU, CFG_GELUMUL_32, CFG_GELUMUL_64 = 5, 12, 13, 14


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


def report(name, calls, t_now, t_new, cos, rel):
    ok = "PASS" if cos >= 0.999 and rel <= 2e-2 else "*** FAIL ***"
    print(f"  {name:26} now {t_now:7.2f}u  fused {t_new:7.2f}u  {t_now/t_new:5.2f}x"
          f"  {(t_new - t_now) * calls / 1000:+7.3f} ms/fwd  {ok} cos {cos:.6f}")


def main() -> int:
    torch.manual_seed(0)

    # -- vision feed-forward expansion: bias and GELU into the epilogue -----
    m, k, n = 768, 1152, 4304
    x = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
    w = torch.randn(k, n, device=DEV, dtype=DT) * 0.02
    bias = torch.randn(n, device=DEV, dtype=DT) * 0.1
    ref = torch.empty(m, n, device=DEV, dtype=DT)
    got = torch.empty(m, n, device=DEV, dtype=DT)

    p_lin = cg.plan(x, w, ref, config=CFG_LINEAR, c=bias, beta=1.0, broadcast_c=True)
    p_gelu = cg.plan(x, w, got, config=CFG_GELU, c=bias, beta=1.0, broadcast_c=True)

    def now():
        cg.run(p_lin)
        cu.gelu_(ref)

    now(); cg.run(p_gelu); torch.cuda.synchronize()
    gold = F.gelu((x.float() @ w.float() + bias.float()), approximate="tanh")
    cos = F.cosine_similarity(gold.flatten(), got.float().flatten(), dim=0).item()
    rel = (torch.linalg.vector_norm(gold - got.float())
           / torch.linalg.vector_norm(gold)).item()
    report("vision ffn_up", 27, time_us(now), time_us(lambda: cg.run(p_gelu)), cos, rel)

    # -- backbone gated feed-forward: gelu_mul into the gate GEMM's epilogue -
    m, k, ffn = 768, 2048, 16384
    x2 = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
    gw = torch.randn(k, ffn, device=DEV, dtype=DT) * 0.02
    uw = torch.randn(k, ffn, device=DEV, dtype=DT) * 0.02
    gate = torch.empty(m, ffn, device=DEV, dtype=DT)
    out = torch.empty(m, ffn, device=DEV, dtype=DT)
    out2 = torch.empty(m, ffn, device=DEV, dtype=DT)

    p_gate = cg.plan(x2, gw, gate, config=CFG_LINEAR)
    p_up = cg.plan(x2, uw, out, config=CFG_LINEAR)
    p_up2 = cg.plan(x2, uw, out2, config=CFG_LINEAR)

    def now2():
        cg.run(p_gate)
        cg.run(p_up)
        cu.gelu_mul(gate, out, out)

    for label, cfg in (("backbone gated ffn /32", CFG_GELUMUL_32),
                       ("backbone gated ffn /64", CFG_GELUMUL_64)):
        # up first, then the gate GEMM finishes the expression in its epilogue.
        p_fused = cg.plan(x2, gw, out2, config=cfg, c=out2, beta=1.0)

        def fused():
            cg.run(p_up2)
            cg.run(p_fused)

        now2(); fused(); torch.cuda.synchronize()
        gold2 = (F.gelu((x2.float() @ gw.float()), approximate="tanh")
                 * (x2.float() @ uw.float()))
        c2 = F.cosine_similarity(gold2.flatten(), out2.float().flatten(), dim=0).item()
        r2 = (torch.linalg.vector_norm(gold2 - out2.float())
              / torch.linalg.vector_norm(gold2)).item()
        report(label, 17, time_us(now2), time_us(fused), c2, r2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
