"""Does the activation transpose actually pay for itself here?

Row-major activation caps boxDim[0] at 64 bf16 elems (128 B, SW128), so BK<=64
and the frame is 8 KB.  M-major lifts BK to 256 (frame to 32 KB).  The FFN
needed that.  This asks whether qkv / o_proj do, given they can raise splitK
instead -- which shrinks per-CTA K and therefore the transaction count too.
"""
CTA_CEIL, BW, L2, TXN, SMS = 132.7e9, 2.77e12, 6447e9, 248e-9, 132
MPAD, BF = 64, 2


def cfg(label, bn, k, n, sk, bk, nwarp=2):
    ctas = -(-n // bn) * sk
    kc = k // sk
    per = MPAD * kc * BF + kc * bn * BF
    txns = -(-kc // bk)
    frame_a, frame_w = MPAD * bk * BF, bk * bn * BF
    rate = min(nwarp * max(frame_a, frame_w) / TXN, CTA_CEIL)
    t_cta, t_ring = per / rate, txns * TXN
    t_hbm, t_l2 = k * n * BF / BW, ctas * per / L2
    floor = max(t_cta, t_ring, t_hbm, t_l2)
    bind = {t_cta: "cta", t_ring: "tma", t_hbm: "hbm", t_l2: "l2"}[floor]
    smem = (frame_a + frame_w) * 4          # depth 4
    return (f"{label:<34}{ctas:>5}{ctas/SMS:>6.2f}{smem/1024:>7.0f}"
            f"{t_ring*1e6:>7.2f}{t_cta*1e6:>7.2f}{t_hbm*1e6:>7.2f}{t_l2*1e6:>7.2f}"
            f"{floor*1e6:>8.2f}  {bind}")


H = (f"{'config':<34}{'CTAs':>5}{'wave':>6}{'smemKB':>7}"
     f"{'tma':>7}{'cta':>7}{'hbm':>7}{'l2':>7}{'floor':>8}  binds")

print("=== qkv   K=1024 N=2560  (weights 5.24 MB, hbm floor 1.89 us) ===\n" + H)
print(cfg("M-major x : BN64 BK128 splitK2 ", 64, 1024, 2560, 2, 128))
print(cfg("row-major : BN64 BK64  splitK2 ", 64, 1024, 2560, 2, 64))
print(cfg("row-major : BN64 BK64  splitK4 ", 64, 1024, 2560, 4, 64))
print(cfg("row-major : BN32 BK64  splitK2 ", 32, 1024, 2560, 2, 64))

print("\n=== o_proj  K=2048 N=1024  (weights 4.19 MB, hbm floor 1.51 us) ===\n" + H)
print(cfg("M-major A : BN64 BK128 splitK4 ", 64, 2048, 1024, 4, 128))
print(cfg("row-major : BN64 BK64  splitK4 ", 64, 2048, 1024, 4, 64))
print(cfg("row-major : BN64 BK64  splitK8 ", 64, 2048, 1024, 8, 64))
print(cfg("row-major : BN32 BK64  splitK8 ", 32, 2048, 1024, 8, 64))

print("\nPer layer-step, x180 layer-steps for the stage total.")
