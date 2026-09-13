#!/usr/bin/env python3
"""Are the large pointwise kernels at DRAM bandwidth, or is there slack?

The backbone's gated feed-forward and the vision feed-forward both end in a
pointwise pass over a large expansion, and those passes are pure traffic: what
they should cost is [ld.bw.dev.dram], 3.35 + MB/1.524 us. This measures them
at their deployed shapes so the answer is a number rather than an assumption.
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu

DEV, DT = "cuda", torch.bfloat16
REPS, INNER = 30, 20


def time_us(fn):
    for _ in range(5):
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


def report(name, fn, mb):
    t = time_us(fn)
    floor = 3.35 + mb / 1.524
    print(f"  {name:26} {t:7.2f} us   moves {mb:6.2f} MB   floor {floor:6.2f} us"
          f"   {floor / t * 100:5.1f}% of it   {mb / t * 1e3:6.0f} GB/s")


def main() -> int:
    torch.manual_seed(0)
    # Backbone gated feed-forward: 768 tokens, 16384 wide.
    gate = torch.randn(768, 16384, device=DEV, dtype=DT)
    up = torch.randn(768, 16384, device=DEV, dtype=DT)
    mb = gate.numel() * 2 * 3 / 1e6
    report("backbone gelu_mul", lambda: cu.gelu_mul(gate, up, up), mb)

    packed = torch.randn(768, 32768, device=DEV, dtype=DT)
    out = torch.empty(768, 16384, device=DEV, dtype=DT)
    report("backbone gelu_mul_packed", lambda: cu.gelu_mul_packed(packed, out),
           (packed.numel() + out.numel()) * 2 / 1e6)

    # Vision feed-forward: 768 tokens, 4304 wide.
    h = torch.randn(768, 4304, device=DEV, dtype=DT)
    report("vision gelu_", lambda: cu.gelu_(h), h.numel() * 2 * 2 / 1e6)

    # The norms that feed them.
    x = torch.randn(768, 2048, device=DEV, dtype=DT)
    o = torch.empty_like(x)
    report("backbone rms_norm", lambda: cu.rms_norm(x, o), x.numel() * 2 * 2 / 1e6)

    v = torch.randn(768, 1152, device=DEV, dtype=DT)
    vo = torch.empty_like(v)
    w = torch.randn(1152, device=DEV, dtype=DT)
    bb = torch.randn(1152, device=DEV, dtype=DT)
    report("vision layer_norm", lambda: cu.layer_norm(v, w, bb, vo),
           v.numel() * 2 * 2 / 1e6)

    # The two remaining hand-written kernels that still move data narrower
    # than 128 bits: the RoPE scatter reads and writes 32-bit pairs, and the
    # masked softmax reads the score row one bf16 per lane.
    rows, qd, hd = 768, 2048, 256
    packed2 = torch.randn(rows, qd + 2 * hd, device=DEV, dtype=DT)
    rope = torch.randn(rows, hd, device=DEV, dtype=DT)
    qq = torch.empty(rows, qd, device=DEV, dtype=DT)
    kk = torch.empty(rows, hd, device=DEV, dtype=DT)
    vv = torch.empty(rows, hd, device=DEV, dtype=DT)
    report("backbone rope_scatter",
           lambda: cu.rope_scatter(packed2, rope, qq, kk, vv),
           packed2.numel() * 2 * 2 / 1e6)

    sc = torch.randn(408, 819, device=DEV, dtype=DT)
    report("expert masked_softmax",
           lambda: cu.expert_masked_softmax(sc, sc, heads=8, prefix=768,
                                            scale=0.0625),
           sc.numel() * 2 * 2 / 1e6)

    # A plain copy at the largest size, as the machine's own answer for what
    # a pure streaming pass costs.
    src = torch.randn(768, 16384, device=DEV, dtype=DT)
    dst = torch.empty_like(src)
    report("reference: copy_", lambda: dst.copy_(src), src.numel() * 2 * 2 / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
