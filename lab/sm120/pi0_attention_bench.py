#!/usr/bin/env python3
"""Validate and sweep the hand-written expert attention against the torch chain.

Pi0's action expert is multi-query over a flat (token, head) query axis: 51
tokens x 8 heads = 408 queries, 768 prefix + 51 suffix = 819 keys, head_dim 256.
BLOCK_M trades K/V re-reads against CTA count and is swept rather than assumed.
"""
from __future__ import annotations

import torch

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu
from flash_vla.hardware.nvidia.rtx5090.pi0.backends import torch_ops as tt

DEV, DT = "cuda", torch.bfloat16
QUERIES, KEYS, HEAD_DIM, HEADS, PREFIX = 408, 819, 256, 8, 768


def time_us(fn, reps=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    s = []
    for _ in range(reps):
        a.record(); fn(); b.record(); b.synchronize()
        s.append(a.elapsed_time(b) * 1000.0)
    s.sort()
    return s[len(s) // 2]


def main() -> int:
    torch.manual_seed(0)
    q = torch.randn(QUERIES, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    k = torch.randn(KEYS, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    v = torch.randn(KEYS, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    ref = torch.empty(QUERIES, HEAD_DIM, device=DEV, dtype=DT)
    got = torch.empty_like(ref)

    tt.action_expert_attention(q, k, v, None, ref, PREFIX)
    torch.cuda.synchronize()
    t_torch = time_us(lambda: tt.action_expert_attention(q, k, v, None, ref, PREFIX))
    print(f"  {'torch chain':<22} {t_torch:8.2f} us")

    ok = True
    for bm in (8, 16, 32):
        cu.expert_attention(q, k, v, got, heads=HEADS, prefix=PREFIX, block_m=bm)
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), got.float().flatten(), dim=0).item()
        rel = (torch.linalg.vector_norm(ref.float() - got.float())
               / torch.linalg.vector_norm(ref.float())).item()
        t = time_us(lambda bm=bm: cu.expert_attention(q, k, v, got, heads=HEADS,
                                                      prefix=PREFIX, block_m=bm))
        good = cos >= 0.999 and rel <= 2e-2
        ok &= good
        print(f"  cuda BLOCK_M={bm:<10} {t:8.2f} us  {t_torch / t:5.2f}x  "
              f"{'PASS' if good else '*** FAIL ***'}  cos {cos:.6f} rel {rel:.2e}")
    return 0 if ok else 1





def bench_split() -> None:
    """The shipped form: cuBLAS GEMMs with a hand-written masked softmax between."""
    torch.manual_seed(0)
    q = torch.randn(QUERIES, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    k = torch.randn(KEYS, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    v = torch.randn(KEYS, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    ref = torch.empty(QUERIES, HEAD_DIM, device=DEV, dtype=DT)
    got = torch.empty_like(ref)
    scores = torch.empty(QUERIES, KEYS, device=DEV, dtype=DT)

    def split():
        torch.mm(q, k.t(), out=scores)
        cu.expert_masked_softmax(scores, scores, heads=HEADS, prefix=PREFIX,
                                 scale=float(HEAD_DIM ** -0.5))
        torch.mm(scores, v, out=got)

    tt.action_expert_attention(q, k, v, None, ref, PREFIX)
    split()
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), got.float().flatten(), dim=0).item()
    t_torch = time_us(lambda: tt.action_expert_attention(q, k, v, None, ref, PREFIX))
    t_split = time_us(split)
    print(f"  {'cuBLAS + fused softmax':<22} {t_split:8.2f} us  "
          f"{t_torch / t_split:5.2f}x  cos {cos:.6f}")





def bench_mma() -> None:
    """The tensor-core single kernel, against the shipped split form."""
    torch.manual_seed(0)
    q = torch.randn(QUERIES, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    k = torch.randn(KEYS, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    v = torch.randn(KEYS, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    ref = torch.empty(QUERIES, HEAD_DIM, device=DEV, dtype=DT)
    got = torch.empty_like(ref)
    scores = torch.empty(QUERIES, KEYS, device=DEV, dtype=DT)

    def split():
        torch.mm(q, k.t(), out=scores)
        cu.expert_masked_softmax(scores, scores, heads=HEADS, prefix=PREFIX,
                                 scale=float(HEAD_DIM ** -0.5))
        torch.mm(scores, v, out=ref)

    def mma():
        cu.expert_attention_mma(q, k, v, got, heads=HEADS, prefix=PREFIX)

    def split_sweep():
        """Split-KV: 26 query tiles times S slices of the key axis."""
        print("  key splits -> wall time, against the split form's "
              f"{time_us(split):.2f} us")
        for s in (1, 2, 4, 6, 8, 12, 16):
            out_s = torch.empty_like(ref)
            fn = lambda: cu.expert_attention_mma(q, k, v, out_s, heads=HEADS,
                                                 prefix=PREFIX, splits=s)
            fn(); torch.cuda.synchronize()
            c = torch.nn.functional.cosine_similarity(
                ref.float().flatten(), out_s.float().flatten(), dim=0).item()
            r = (torch.linalg.vector_norm(ref.float() - out_s.float())
                 / torch.linalg.vector_norm(ref.float())).item()
            t = time_us(fn)
            ok = "PASS" if c >= 0.999 and r <= 2e-2 else "*** FAIL ***"
            print(f"  {s:3d} splits, {26 * s:4d} CTAs  {t:7.2f} us  "
                  f"{ts / t:5.2f}x  {ok}  cos {c:.6f} rel {r:.2e}")

    split(); mma(); torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), got.float().flatten(), dim=0).item()
    rel = (torch.linalg.vector_norm(ref.float() - got.float())
           / torch.linalg.vector_norm(ref.float())).item()
    ts, tm = time_us(split), time_us(mma)
    split_sweep()
    print(f"  {'cuBLAS + fused softmax':<24} {ts:8.2f} us")
    print(f"  {'tensor-core one kernel':<24} {tm:8.2f} us  {ts / tm:5.2f}x  "
          f"{'PASS' if cos >= 0.999 and rel <= 2e-2 else '*** FAIL ***'}  "
          f"cos {cos:.6f} rel {rel:.2e}")




def bench_mma_scaling() -> None:
    """Does the tensor-core kernel's wall time track its CTA count?

    Pi0's shape gives 408/16 = 26 CTAs on a 170-SM part, so 85% of the machine
    is idle and a warp scheduler holds one warp -- every instruction latency is
    exposed. If that is the binding constraint, multiplying the query count
    should cost far less than proportionally until the grid fills the part.
    """
    torch.manual_seed(0)
    k = torch.randn(KEYS, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    v = torch.randn(KEYS, HEAD_DIM, device=DEV, dtype=DT) * 0.1
    print("  query tiles (CTAs) -> wall time, one CTA per SM either way")
    base = None
    for mult in (1, 2, 4, 6, 8, 12):
        rows = QUERIES * mult
        q = torch.randn(rows, HEAD_DIM, device=DEV, dtype=DT) * 0.1
        out = torch.empty(rows, HEAD_DIM, device=DEV, dtype=DT)
        fn = lambda: cu.expert_attention_mma(q, k, v, out, heads=HEADS,
                                             prefix=PREFIX)
        fn(); torch.cuda.synchronize()
        t = time_us(fn)
        base = base or t
        ctas = (rows + 15) // 16
        print(f"  {ctas:5d} CTAs  {t:8.2f} us   {t / base:5.2f}x time for "
              f"{mult}x work")


def _run_all() -> int:
    rc = main()
    bench_split()
    bench_mma()
    bench_mma_scaling()
    return rc


if __name__ == "__main__":
    raise SystemExit(_run_all())
