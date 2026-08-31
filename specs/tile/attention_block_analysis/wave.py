"""Tile / wave analysis for the Pi0.5 action-expert attention block.

Floors use MEASURED constants (hardware-unit-test/sm90/constants.yaml):
  ld.bw.dev.dram        2.77 TB/s marginal HBM        (not the 3.35 datasheet)
  tma.bw.cta.dram   132.7 GB/s into ONE CTA; >=23 CTAs for peak BW, ~48 for 90%
  tma.issue.warp      248 ns per TMA per producer warp
  tma.bytes.txn.max  box row <= 128 B; frame grows via boxDim[1] <= 256; max 32 KB
  L2             6447 GB/s (measured at 264 CTAs; pessimistic below that)

Model: 2 producer warps per CTA (one ring each for A and W / K and V), so the
two copy columns run in parallel.  per-CTA time = max(bytes/133GB/s, ring_txns*248ns).
"""
SMS, CTA_CEIL, BW, L2, TXN = 132, 132.7e9, 2.77e12, 6447e9, 248e-9
M, D, H, DH, QKV, KEYS, MPAD = 50, 1024, 8, 256, 2560, 1018, 64
QROWS = M * H


def frame_ok(rows, cols_elems):
    """tma.bytes.txn.max: one box row <= 128 B (64 bf16) and boxDim[1] <= 256."""
    return cols_elems * 2 <= 128 and rows <= 256


def row(label, ctas, per_a, per_w, txn_a, txn_w, hbm, frames):
    per = per_a + per_w
    t_cta = per / CTA_CEIL
    t_ring = max(txn_a, txn_w) * TXN
    t_hbm = hbm / BW
    t_l2 = ctas * per / L2
    floor = max(t_cta, t_ring, t_hbm, t_l2)
    bind = {t_cta: "cta-133GB/s", t_ring: "tma-issue", t_hbm: "hbm-bw", t_l2: "l2-bw"}[floor]
    legal = "" if all(frames) else "  <-- ILLEGAL TMA frame"
    return (f"{label:<26}{ctas:>5}{ctas/SMS:>6.2f}{per/1024:>8.0f}"
            f"{t_cta*1e6:>7.2f}{t_ring*1e6:>7.2f}{t_hbm*1e6:>7.2f}{t_l2*1e6:>7.2f}"
            f"{floor*1e6:>8.2f}  {bind}{legal}")


def gemm(label, bm, bn, k, n, sk=1, bk=256, m=MPAD):
    ctas = -(-m // bm) * -(-n // bn) * sk
    kc = k // sk
    nk = -(-kc // bk)
    return row(label, ctas, bm * kc * 2, kc * bn * 2, nk, nk, k * n * 2,
               [frame_ok(bk, bm), frame_ok(bk, bn)])


def attn(label, bmq, S, bkk=64):
    ctas = -(-QROWS // bmq) * S
    kv = (KEYS / S) * DH * 2
    nk = -(-int(KEYS / S) // bkk)
    return row(label, ctas, bmq * DH * 2, 2 * kv, 1, nk, KEYS * DH * 2 * 2,
               [frame_ok(bkk, 64)])


HDR = (f"{'tile':<26}{'CTAs':>5}{'wave':>6}{'KB/CTA':>8}"
       f"{'cta':>7}{'tma':>7}{'hbm':>7}{'l2':>7}{'floor':>8}  binds")

print("=== qkv   M=64  K=1024  N=2560   weights 5.24 MB ===\n" + HDR)
for r in [gemm("BN=128 BK=256          ", MPAD, 128, D, QKV),
          gemm("BN=64  BK=256          ", MPAD, 64, D, QKV),
          gemm("BN=32  BK=256 (shipped)", MPAD, 32, D, QKV),
          gemm("BN=32  BK=128          ", MPAD, 32, D, QKV, bk=128),
          gemm("BN=64  BK=256 splitK=2 ", MPAD, 64, D, QKV, sk=2),
          gemm("BN=32  BK=256 splitK=2 ", MPAD, 32, D, QKV, sk=2),
          gemm("BN=32  BK=128 splitK=2 ", MPAD, 32, D, QKV, sk=2, bk=128)]:
    print(r)
print(f"{'MEASURED (PLAN 4.9)':<26}{'':>41}{10.44:>8.2f}")

print("\n=== o_proj   M=64  K=2048  N=1024   weights 4.19 MB ===\n" + HDR)
for r in [gemm("BN=64  BK=256          ", MPAD, 64, 2048, D),
          gemm("BN=32  BK=256          ", MPAD, 32, 2048, D),
          gemm("BN=32  BK=256 splitK=2 ", MPAD, 32, 2048, D, sk=2),
          gemm("BN=64  BK=256 splitK=2 ", MPAD, 64, 2048, D, sk=2),
          gemm("BN=64  BK=256 splitK=4 ", MPAD, 64, 2048, D, sk=4),
          gemm("BN=32  BK=256 splitK=4 ", MPAD, 32, 2048, D, sk=4),
          gemm("BM=16 BN=32 (shipped)  ", 16, 32, 2048, D)]:
    print(r)
print(f"{'MEASURED (PLAN 4.9)':<26}{'':>41}{5.42:>8.2f}")

print("\n=== attention   400 Q rows x 1018 keys x Dh=256   KV 1.04 MB ===\n" + HDR)
for r in [attn("BMq=128 S=1 pure flash  ", 128, 1),
          attn("BMq=64  S=1 pure flash  ", 64, 1),
          attn("BMq=32  S=1             ", 32, 1),
          attn("BMq=64  S=2             ", 64, 2),
          attn("BMq=64  S=4             ", 64, 4),
          attn("BMq=64  S=8  (shipped)  ", 64, 8),
          attn("BMq=64  S=16            ", 64, 16),
          attn("BMq=400 S=8  1 CTA/head ", 400, 8)]:
    print(r)
print(f"{'MEASURED (PLAN 4.9)':<26}{'':>41}{10.25:>8.2f}")
