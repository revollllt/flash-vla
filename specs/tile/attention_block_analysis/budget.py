"""Step 1 (TMA descriptor legality) + Step 2 (smem / register union) for the
Pi0.5 action-expert attention block, at the tiles chosen by wave.py.

Rules, all from hardware-unit-test/sm90/constants.yaml:
  tma.bytes.txn.max  boxDim[0]*elem <= swizzle width (SW128:128B SW64:64B SW32:32B
                 none:unbounded); boxDim[1] <= 256; max SW128 frame = 32 KB.
                 "Check this BEFORE choosing BK."
  tma.bw.cta.dram   knee at 36 KB of n_warps x frame; 2 producers ~= +10% over 1,
                 3rd buys nothing.  1 warp @32KB frame = 121.3 of 132.7 GB/s.
  tma.stages.warp.knee      ring depth saturates at 4.
  wgmma.stages.wg.knee      >=4 wgmma in flight, never wait_group 0 (costs 20-30%).
  wgmma.ratio.sm.wg2 a 2nd math WG buys NO throughput; add it only to hold more
                 accumulator registers than 255/thread allows.
  sched.ctas.sm.knee  grid must be >= 3x SM count before extra CTAs/SM become warps.
  smem_per_cta_b 232448 (227 KB);  register file 65536 regs/SM.
"""
SMEM_CAP, REGFILE, MAXREG = 232448, 65536, 255
BF = 2

SW = {"SW128": 128, "SW64": 64, "SW32": 32, "none": 1 << 30}


def legal(box0_elems, box1, swizzle="SW128"):
    return box0_elems * BF <= SW[swizzle] and box1 <= 256


def desc(name, gmem, box, swizzle, note):
    """box = (boxDim[0] elems, boxDim[1], [boxDim[2]]) innermost first."""
    ok = legal(box[0], box[1], swizzle)
    frame = BF
    for b in box:
        frame *= b
    ok = ok and frame <= 32768
    return (f"{name:<20}{gmem:<22}{str(box):<18}{swizzle:<8}"
            f"{frame/1024:>7.0f}  {'OK ' if ok else 'ILLEGAL'}  {note}")


print("=== Step 1: TMA descriptor legality "
      "(box dims innermost-first, bf16) ===\n")
print(f"{'tensor':<20}{'gmem shape':<22}{'box':<18}{'swz':<8}{'KB':>7}  {'':7}  why")
print(desc("x  row-major", "(64, 1024)", (128, 64), "SW128", "BK=128 row is 256 B -> must drop to BK<=64, frame 8 KB"))
print(desc("x  M-major  *", "(1024, 64)", (64, 128), "SW128", "M_PAD=64 elems = 128 B exactly; BK free to 256"))
print(desc("W_qkv       *", "(1024, 2560)", (64, 128), "SW128", "BN=64 -> 128 B row; legal AS STORED, no pre-block"))
print(desc("W_qkv BN=128", "(1024, 2560)", (128, 128), "SW128", "256 B row; legal only unswizzled"))
print(desc("W_o         *", "(2048, 1024)", (64, 128), "SW128", "same shape of constraint as W_qkv"))
print(desc("Q  row-major*", "(400, 4, 64)", (64, 4, 64), "SW128", "3-D box over Dh=4x64; also what wgmma A wants"))
print(desc("Q  transposed", "(256, 400)", (64, 256), "SW128", "legal 2-D, but forces a strided store in qkv"))
print(desc("K$ row-major*", "(1024, 4, 64)", (64, 4, 32), "SW128", "3-D; B operand K-major (contract over Dh)"))
print(desc("V$ row-major*", "(1024, 4, 64)", (64, 4, 32), "SW128", "SAME copy, but B operand MN-major -> other atom"))
print(desc("out M-major *", "(1024, 64)", (64, 128), "SW128", "= next layer's x; epilogue store is strided (priced)"))
print("\n  * = the shape to build.  Rows without * are the rejected alternative.")

print("\n\n=== Step 2a: shared-memory pool, per CTA ===\n")
DEPTH = 4


def ring(label, frame_b, depth=DEPTH, n=1):
    return label, frame_b * depth * n, frame_b


def body(name, parts):
    tot = sum(p[1] for p in parts)
    detail = " + ".join(f"{p[0]} {p[1]/1024:.0f}" for p in parts)
    return name, tot, detail


bodies = [
    body("qkv    BN=64 BK=128", [ring("A ring", 64 * 128 * BF), ring("W ring", 128 * 64 * BF)]),
    body("o_proj BN=64 BK=128", [ring("A ring", 64 * 128 * BF), ring("W ring", 128 * 64 * BF)]),
    body("attn   BMq=64 BKk=32", [("Q resident", 64 * 256 * BF, 0),
                                  ring("K ring", 32 * 256 * BF), ring("V ring", 32 * 256 * BF)]),
]
print(f"{'task body':<22}{'KB':>6}   breakdown (KB)")
for name, tot, detail in bodies:
    print(f"{name:<22}{tot/1024:>6.0f}   {detail}")
union = max(b[1] for b in bodies)
print(f"\n{'UNION (static pool)':<22}{union/1024:>6.0f}   of {SMEM_CAP/1024:.0f} KB cap, "
      f"{(SMEM_CAP-union)/1024:.0f} KB spare -> ONE pool, NO paging")
print(f"{'n_warps x frame':<22}{2*16:>6}   KB, vs the 36 KB tma.bw.cta.dram knee "
      f"-> 121.3 of 132.7 GB/s (91%)")

print("\n\n=== Step 2b: accumulator registers per thread ===\n")


def acc(name, elems_f32, elems_bf16=0):
    return name, elems_f32, elems_bf16


accs = [
    acc("qkv     C 64x64", 64 * 64),
    acc("o_proj  C 64x64", 64 * 64),
    acc("attn    S 64x32 + O 64x256", 64 * 32 + 64 * 256, 64 * 32),
]
print(f"{'task body':<28}{'1 math WG':>11}{'2 math WG':>11}   (regs/thread)")
for name, f32, bf16 in accs:
    r1 = f32 / 128 + bf16 / 128 / 2
    r2 = f32 / 256 + bf16 / 256 / 2
    print(f"{name:<28}{r1:>11.0f}{r2:>11.0f}")
u1 = max(a[1] / 128 + a[2] / 256 for a in accs)
u2 = max(a[1] / 256 + a[2] / 512 for a in accs)
print(f"\n{'UNION (accumulators)':<28}{u1:>11.0f}{u2:>11.0f}")
print(f"{'+ addressing / m,l / ring':<28}{30:>11}{30:>11}")
print(f"{'= per-thread total':<28}{u1+30:>11.0f}{u2+30:>11.0f}   cap {MAXREG}")

for label, math_thr in (("1 math WG", 128), ("2 math WG", 256)):
    thr = math_thr + 64                      # + 2 producer warps
    regs = (u1 if math_thr == 128 else u2) + 30
    print(f"\n{label}: {thr} threads, {regs:.0f} regs/thread"
          f" -> {thr*regs:.0f} of {REGFILE} regs/SM"
          f"  ({REGFILE/(thr*regs):.1f} CTA/SM by registers)")
print("\nGrid is 56-80 CTAs = 0.4-0.6x SM count, so sched.ctas.sm.knee says the")
print("operating point is 1 CTA/SM regardless -- register headroom buys nothing")
print("but spill margin.  2 math WG is a comfort call, not a requirement.")
