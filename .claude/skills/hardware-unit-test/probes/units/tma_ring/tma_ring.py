"""TMA delivery rate -- what does the copy engine actually sustain?

The FFN task-loop confounds six things at once (CTA count, stage count, bytes
per box, box geometry, wgmma-retirement coupling, counter polling), so no number
taken from inside it is a machine constant. This unit strips everything but the
copy engine and sweeps the axes separately.

Questions, in the order they change the design:

Q0  Is the copy engine delivering the bytes the descriptor claims? `verify()`
    replays the rate kernel's own coordinate walk at one stage and compares each
    delivered box against the source. It runs FIRST and aborts the job on
    failure: a wrong stride, box or coordinate delivers the wrong bytes at the
    right speed, so every rate below would look reasonable and be meaningless.

Q1  Is TMA limited by in-flight BYTES or by in-flight TRANSACTIONS? Sweep A
    holds `stages * box_bytes` fixed while varying the split.

Q2  What does box geometry cost at equal bytes and equal transaction count?

Q3  Is one producer warp's serial issue loop the limit? Sweep C varies warps.

Q4  Which box widths and how many box rows does the driver accept? `describe()`
    ENUMERATES cuTensorMapEncodeTiled return codes rather than assuming them.

Q5  How FEW CTAs still saturate? Q6  How SMALL can one box be and still saturate?
    E and F are the same frontier along its two axes; measuring both is what
    makes it falsifiable rather than a rearrangement of one fit.

REGIME (protocol.md rule 6b). Two regimes are machine constants and one is not:

    --regime dram   large walk, L2 FLUSHED between timed iterations
    --regime l2     footprint SMALLER than L2, no flush
    (neither)       a large walk without the flush is a partly-cached DRAM read
                    whose value depends on the sweep, and it inflated every
                    absolute constant recorded before 2026-08-29

Run:
    python3 tma_ring.py --sweeps A,E,F --regime dram --json /tmp/tma.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hut import abi, harness, regime  # noqa: E402
from hut.tma import (DTYPES, DT_BF16, SW_128B, SW_32B, SW_64B, SW_NAME,  # noqa: E402
                     SW_NONE, Geom, encode, geoms)

_SRC = Path(__file__).resolve().with_suffix(".cu")

# 256 MB is >2x this H100's 50 MB L2, so a walk covering it cannot become
# cache-resident however the sweep wraps -- but see touched_bytes: covering it
# is not the same as reaching it.
BUF_MB = 256
BUF_B = BUF_MB * 1024 * 1024

# Bytes moved per measured launch. 64 MB keeps every config in the tens of
# microseconds, where CUPTI is well above its own noise -- but it also caps how
# far the WALK reaches, and at 64 MB the touched set is only ~1.3x L2, which is
# not cold. --regime dram raises it.
TARGET_BYTES = 64 * 1024 * 1024
FLUSH_L2 = False

N_SM = 132
PEAK_TBS = 3.35              # datasheet. NOT a floor denominator.
MAX_SMEM = 227 * 1024
SMEM_PER_SM = 233472
BW_CEIL_GBS = 2770.0         # [ld.bw.dev.dram], the measured plain-load ceiling

GEOMS = geoms()

# Unit trap sites beyond the library's reserved range.
SITES = {}


def measure(unit, geom, *, n_ctas, num_producers, stages, box_dim_1,
            buf_bytes) -> dict:
    box_bytes = geom.box_bytes(box_dim_1)
    smem = num_producers * stages * box_bytes + num_producers * stages * 8
    if smem > MAX_SMEM:
        return dict(skipped=f"smem {smem} > {MAX_SMEM}")
    plan = geom.plan(box_dim_1, buf_bytes)
    # k_tile_count must be well past the ring fill so the measurement is steady
    # state: 8x stages holds the fill under 13%.
    per_issue = n_ctas * num_producers * box_bytes
    k = max(8 * stages, TARGET_BYTES // per_issue)
    return dict(plan=plan, box_bytes=box_bytes, k_tile_count=k,
                total_b=per_issue * k, smem=smem,
                touched_b=geom.touched_bytes(plan, n_ctas, num_producers, k))


def make_params(plan, *, n_ctas, num_producers, stages, box_bytes,
                k_tile_count, tensor_map, mode=0, n_check=0) -> abi.HutParams:
    p = abi.HutParams()
    p.mode = mode
    p.n_ctas, p.num_producers, p.stages = n_ctas, num_producers, stages
    p.k_tile_count, p.box_bytes = k_tile_count, box_bytes
    p.txn_bytes = p.stage_bytes = box_bytes    # one box per barrier, per stage
    for f in ("mask0", "shift0", "step0", "mask1", "step1"):
        setattr(p, f, plan[f])
    p.opt[0] = n_check
    p.tensor_map = tensor_map
    return p


def run_point(unit, buf, geom, *, n_ctas, num_producers, stages, box_dim_1,
              buf_bytes, dbg, sm_id=None) -> dict:
    pre = measure(unit, geom, n_ctas=n_ctas, num_producers=num_producers,
                  stages=stages, box_dim_1=box_dim_1, buf_bytes=buf_bytes)
    base = dict(geom=geom.name, n_ctas=n_ctas, num_producers=num_producers,
                stages=stages, box_bytes=geom.box_bytes(box_dim_1))
    if "skipped" in pre:
        return dict(**base, skipped=pre["skipped"])
    mapbuf, rc = encode(unit, buf.data_ptr(), pre["plan"])
    if rc != 0:
        return dict(**base, skipped=f"encode rc={rc}")

    regime.guard(unit, regime.DRAM if FLUSH_L2 else regime.L2,
                 pre["touched_b"], where=f"{geom.name} {n_ctas}x{num_producers}")

    p = make_params(pre["plan"], n_ctas=n_ctas, num_producers=num_producers,
                    stages=stages, box_bytes=pre["box_bytes"],
                    k_tile_count=pre["k_tile_count"], tensor_map=mapbuf[1])
    bufs = abi.buffers(dbg=dbg, sm_id=sm_id)
    dbg.zero_()
    us, spread = harness.time_us(lambda: unit.launch(p, bufs),
                                 flush_l2=FLUSH_L2)
    torch.cuda.synchronize()
    harness.check_watchdog(dbg, SITES)

    gbs = pre["total_b"] / (us * 1e-6) / 1e9
    # The constant this unit exists to produce. Every warp issues exactly
    # k_tile_count transactions over the same us, so the per-warp issue interval
    # is a division that never passes through the byte accounting -- a
    # miscounted footprint cannot produce it. [protocol.md rule 4]
    row = dict(**base, k_tile_count=pre["k_tile_count"],
               total_mb=pre["total_b"] / 1e6, us=us, spread=spread, gbs=gbs,
               ns_per_txn=us * 1000.0 / pre["k_tile_count"],
               pct_peak=100.0 * gbs / (PEAK_TBS * 1000),
               inflight_kb=n_ctas * num_producers * stages * pre["box_bytes"] / 1024,
               smem_kb=pre["smem"] / 1024,
               addressable_mb=pre["plan"]["addressable_bytes"] / 1e6,
               **regime.stamp(regime.DRAM if FLUSH_L2 else regime.L2,
                              pre["touched_b"], FLUSH_L2))
    row["footprint_mb"] = row["touched_mb"]
    if sm_id is not None:
        ids = sm_id[:n_ctas].tolist()
        per_sm = {}
        for i in ids:
            per_sm[i] = per_sm.get(i, 0) + 1
        counts = sorted(per_sm.values())
        row.update(sms_used=len(per_sm), cta_per_sm_max=counts[-1],
                   cta_per_sm_min=counts[0])
        sms = len(per_sm) or N_SM
        row["per_sm_kb"] = n_ctas * num_producers * row["box_bytes"] / sms / 1024
        row["gbs_per_sm"] = gbs / sms
        row["resident_cap"] = max(1, SMEM_PER_SM //
                                  (num_producers * stages * row["box_bytes"] +
                                   num_producers * stages * 8))
        if row["cta_per_sm_max"] > row["resident_cap"]:
            # Cumulative placement, not concurrent residency: this grid ran in
            # more than one wave and the wave boundary is inside the window.
            row["note"] = "MULTIWAVE"
    return row


N_CHECK_BOXES = 16
CHECK_BOX_DIM_1 = 32


def verify(unit, buf: torch.Tensor, dbg, buf_bytes) -> list[dict]:
    """Q0. Does a TMA box actually carry the bytes the descriptor claims?

    Every rate below is a measurement of an unknown until this passes. A wrong
    stride, box or coordinate walk delivers the WRONG bytes at the RIGHT speed,
    so no row of any sweep would look suspicious and the constants extracted
    from them would be unfalsifiable. This is the copy engine's analogue of the
    ``mma`` unit's ``--check``, which gates every wgmma rate on a torch compare.

    Two passes per geometry, because the swizzle cannot be inverted here without
    hardcoding the very layout the unit would then be asserting rather than
    testing:

    * ``SW_NONE`` -- the box lands row-major in smem, so the comparison is
      BYTE-EXACT. This is what pins the strides, the box geometry and the walk.
    * ``SW_128B`` -- the descriptor the sweeps actually use. Its bytes are
      permuted WITHIN the box, so the comparison is on the multiset of elements.
      That still fails on a wrong coordinate, a wrong box or a wrong stride; it
      does not check the intra-box permutation, and does not claim to.
    """
    rows = []
    for geom in GEOMS.values():
        p = geom.plan(CHECK_BOX_DIM_1, buf_bytes)
        box_bytes = geom.box_bytes(CHECK_BOX_DIM_1)
        src = buf[: p["global_dim_1"] * p["global_dim_0"]].view(
            p["global_dim_1"], p["global_dim_0"])
        for sw, mode in ((SW_NONE, "exact"), (SW_128B, "multiset")):
            row = dict(geom=geom.name, swizzle=SW_NAME[sw], mode=mode,
                       box_bytes=box_bytes, boxes=N_CHECK_BOXES)
            mapbuf, rc = encode(unit, buf.data_ptr(), p, swizzle=sw)
            if rc != 0:
                rows.append({**row, "skipped": f"encode rc={rc}"})
                continue
            out = torch.zeros(N_CHECK_BOXES * box_bytes // 2,
                              dtype=torch.bfloat16, device=buf.device)
            coords = torch.zeros(N_CHECK_BOXES * 2, dtype=torch.int32,
                                 device=buf.device)
            dbg.zero_()
            params = make_params(p, n_ctas=1, num_producers=1, stages=1,
                                 box_bytes=box_bytes,
                                 k_tile_count=N_CHECK_BOXES,
                                 tensor_map=mapbuf[1], mode=1,
                                 n_check=N_CHECK_BOXES)
            unit.check(params, abi.buffers(out=out, sm_id=coords, dbg=dbg))
            torch.cuda.synchronize()
            harness.check_watchdog(dbg, SITES)
            got = out.view(N_CHECK_BOXES, CHECK_BOX_DIM_1, geom.box_dim_0)
            bad_coord = bad_data = 0
            for g, (c0, c1) in enumerate(coords.view(-1, 2).tolist()):
                # The walk the HOST believes in. A mismatch here is a
                # host/device disagreement about the sweep, not a hardware
                # finding -- and it would silently corrupt the reference below.
                if (c0 != (g & p["mask0"]) * p["step0"]
                        or c1 != ((g >> p["shift0"]) & p["mask1"]) * p["step1"]):
                    bad_coord += 1
                    continue
                ref = src[c1:c1 + CHECK_BOX_DIM_1, c0:c0 + geom.box_dim_0]
                if mode == "exact":
                    ok = torch.equal(got[g], ref)
                else:
                    ok = torch.equal(
                        torch.sort(got[g].reshape(-1).float()).values,
                        torch.sort(ref.reshape(-1).float()).values)
                bad_data += not ok
            rows.append({**row, "bad_coord": bad_coord, "bad_data": bad_data,
                         "ok": bad_coord == 0 and bad_data == 0})
    return rows



def describe(unit, buf: torch.Tensor) -> list[dict]:
    """Enumerate cuTensorMapEncodeTiled legality: which box widths each
    swizzle mode accepts. Settles whether a TMA box row can exceed the swizzle
    width -- i.e. whether 'make each TMA bigger' can grow the contiguous run or
    only the number of strips."""
    rows = []
    ptr = buf.data_ptr()
    for sw in (SW_NONE, SW_32B, SW_64B, SW_128B):
        for box_dim_0 in (8, 16, 32, 64, 128, 256):
            inner = max(box_dim_0, 4096)
            _, rc = encode(unit, ptr, dict(global_dim_0=inner, global_dim_1=1024,
                                           box_dim_0=box_dim_0, box_dim_1=8,
                                           swizzle=sw, tensor_data_type=DT_BF16,
                                           elem_bytes=2))
            rows.append(dict(swizzle=SW_NAME[sw], box_inner_elem=box_dim_0,
                             box_dim_0_b=box_dim_0 * 2, rc=rc, ok=rc == 0))
    return rows



def describe_rows(unit, buf: torch.Tensor) -> list[dict]:
    """Enumerate the accepted range of boxDim[1] -- the number of box ROWS.

    The width table above caps a box ROW at the swizzle width, so the only way
    to a bigger TMA is more rows. That makes `max boxDim[1] x swizzle width`
    the hard ceiling on bytes-per-TMA, and therefore the ceiling on the whole
    "bigger box" lever. It is worth one call per candidate rather than one
    sentence of recollection.
    """
    rows = []
    ptr = buf.data_ptr()
    for box_dim_1 in (8, 64, 128, 192, 256, 257, 512):
        _, rc = encode(unit, ptr, dict(global_dim_0=4096, global_dim_1=1024,
                                           box_dim_0=64, box_dim_1=box_dim_1,
                                           swizzle=SW_128B, tensor_data_type=DT_BF16,
                                           elem_bytes=2))
        rows.append(dict(box_dim_1=box_dim_1, box_bytes=128 * box_dim_1,
                         rc=rc, ok=rc == 0))
    return rows


# -------------------------------------------------------------- Q0 / M0 gate
# Frames per geometry per swizzle mode. 16 is enough to walk every strip of the
# widest descriptor here (stride8k has nc0 = 64, but the first 16 already cover
# four distinct c0 and the c1 hop), and small enough that the gate costs
# milliseconds on a job that runs for minutes.
N_CHECK_BOXES = 16
CHECK_BOX_DIM_1 = 32


def sweep_A(unit, buf, dbg, buf_bytes):
    """Q1: bytes vs transactions. stages x box_bytes at fixed CTA/warp count.

    The decisive pairs are equal-in-flight-bytes, different transaction count:
    (stages=16, 2 KB) vs (stages=2, 16 KB) both hold 32 KB per warp.
    """
    out = []
    for stages in (2, 4, 8, 16):
        for box_dim_1 in (16, 32, 64, 128, 256):     # 2/4/8/16/32 KB boxes
            out.append(run_point(unit, buf, GEOMS["stride8k"], n_ctas=32,
                                 num_producers=1, stages=stages, box_dim_1=box_dim_1,
                                 buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_B(unit, buf, dbg, buf_bytes):
    """Q2: box geometry at equal bytes and equal transaction count."""
    out = []
    for name in ("contig", "stride2k", "stride8k"):
        for n_ctas in (32, 132):
            out.append(run_point(unit, buf, GEOMS[name], n_ctas=n_ctas,
                                 num_producers=1, stages=4, box_dim_1=64,
                                 buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_C(unit, buf, dbg, buf_bytes):
    """Q3: does one producer warp's serial issue loop cap the rate?"""
    out = []
    for num_producers in (1, 2, 4):
        for n_ctas in (32, 132):
            out.append(run_point(unit, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                                 num_producers=num_producers, stages=4, box_dim_1=64,
                                 buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_D(unit, buf, dbg, buf_bytes):
    """CTA scaling at the task-loop's own ring shape (stages 4, 8 KB boxes)."""
    out = []
    for n_ctas in (16, 32, 64, 128, 132, 264):
        out.append(run_point(unit, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                             num_producers=1, stages=4, box_dim_1=64,
                             buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_E(unit, buf, dbg, buf_bytes):
    """Q5: how FEW CTAs saturate? CTA scaling at 8 / 16 / 32 KB boxes.

    Read it as the frontier along the CTA axis: for each box size, the
    smallest CTA count whose row reaches ~95% of the 2.77 TB/s ceiling. A
    bigger box should move that count down proportionally -- if it does not,
    the per-warp issue interval is not box-independent and every floor
    derived from a single `ns/txn` is wrong.
    """
    out = []
    for box_dim_1 in (64, 128, 256):              # 8 / 16 / 32 KB
        for n_ctas in (8, 16, 24, 32, 48, 64, 96, 128, 132, 264):
            out.append(run_point(unit, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                                 num_producers=1, stages=4, box_dim_1=box_dim_1,
                                 buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_F(unit, buf, dbg, buf_bytes):
    """Q6: how SMALL can one CTA's TMA be at full occupancy? Frame scaling.

    Three producer configurations at 1 KB..32 KB boxes: one CTA per SM, two
    CTAs per SM, and one CTA per SM with two producer warps. If the frontier is
    `n_cta x n_warp x box`, the (264,1) and (132,2) rows must saturate at the
    same box size as each other and at half the (132,1) one. That equality is
    the falsifiable form of "warps and CTAs enter the copy budget identically".
    """
    out = []
    for n_ctas, num_producers in ((132, 1), (264, 1), (132, 2)):
        for box_dim_1 in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
            out.append(run_point(unit, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                                 num_producers=num_producers, stages=4, box_dim_1=box_dim_1,
                                 buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_G(unit, buf, dbg, buf_bytes):
    """Q7: the CTA scan again with TWO producer warps, and the smem wall.

    Sweep E answers "how few CTAs" for one producer warp per CTA. A second warp
    should halve that count -- the product law says CTAs and warps are the same
    currency, and sweep F showed (264,1) and (132,2) agreeing at one CTA count.
    This walks the CTA axis so the equality is tested across the whole range,
    G at N CTAs matching E at 2N.

    It also probes where the trade STOPS being free. Shared memory bounds
    `num_producers x stages x box` per CTA, so a second warp cannot keep the 32 KB
    box: 2 x 4 x 32 KB = 256 KB exceeds the 227 KB cap and the largest box
    two warps can hold at stages 4 is 28 KB. The 4-warp rows at 7 and 14 KB test
    the consequence -- if the per-CTA in-flight budget is really smem/stages
    however it is split, 4x14 KB and 2x28 KB must deliver the same.
    """
    out = []
    plan = [(2, (64, 128, 192, 224)),        # 8 / 16 / 24 / 28 KB
            (4, (56, 112))]                  # 7 / 14 KB
    for num_producers, boxes in plan:
        for box_dim_1 in boxes:
            for n_ctas in (8, 16, 24, 32, 48, 64, 96, 128, 132):
                out.append(run_point(unit, buf, GEOMS["stride8k"],
                                     n_ctas=n_ctas, num_producers=num_producers, stages=4,
                                     box_dim_1=box_dim_1,
                                     buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_H(unit, buf, dbg, buf_bytes):
    """Q8: where does a CTA stop absorbing more? The per-CTA delivery knee.

    Sweep G found that per-CTA delivery is linear in `num_producers x box` up to
    32 KB and pinned at ~133 GB/s by 48 KB, with three different splits landing
    on the same number. That brackets a knee but does not locate it, and the
    knee IS the answer to "what does a second producer warp buy" -- above it,
    nothing.

    So: walk the per-CTA product from 32 to 48 KB in 4 KB steps, by two
    different splits (2 warps with a growing box, 4 warps with a smaller one)
    at two CTA counts. If the knee is a per-CTA property it lands at the same
    product for both splits and both grids; if it tracks the box or the warp
    count instead, the two curves separate.
    """
    out = []
    # Extended past 48 KB after job 565686 read 200 GB/s per CTA at a 56 KB
    # product where this sweep's own knee said 133. Either the knee is not a
    # knee or one of the two runs is wrong, and the only way to tell is to walk
    # the gap in ONE job on ONE set of SMs.
    for n_ctas in (8, 16):
        for num_producers, boxes in ((1, (256,)),                              # 32 KB ref
                               (2, (128, 144, 160, 176, 192, 208, 224)),  # 32..56 KB
                               (4, (64, 72, 80, 88, 96, 104, 112))):      # 32..56 KB
            for box_dim_1 in boxes:
                out.append(run_point(unit, buf, GEOMS["stride8k"],
                                     n_ctas=n_ctas, num_producers=num_producers, stages=4,
                                     box_dim_1=box_dim_1,
                                     buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_I(unit, buf, dbg, buf_bytes):
    """Q9: the transition region, where neither limit binds cleanly.

    Between roughly 16 and 96 CTAs, with a per-CTA product above the 36 KB
    knee, the two-term model min(n_ctas x per_cta, curve(product)) measured up
    to 15% optimistic, and the CTA ladder is not even monotone there (48 CTAs
    came in BELOW 32 at 2 warps x 24 KB). Coarse steps cannot tell a real dip
    from a noisy point, so walk the band finely at three per-CTA products, with
    the 1-warp column as a control that stays below the knee.
    """
    out = []
    for num_producers, box_dim_1 in ((1, 256), (2, 192), (2, 224)):
        for n_ctas in (16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 80, 96):
            out.append(run_point(unit, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                                 num_producers=num_producers, stages=4, box_dim_1=box_dim_1,
                                 buf_bytes=buf_bytes, dbg=dbg))
    return out


def sweep_J(unit, buf, dbg, buf_bytes):
    """Q10: TMA latency as a function of box size, measured not inferred.

    tma.stages.warp.knee infers ~680 ns of latency at 8 KB from the stages-2 interval, and
    says the 16 and 32 KB stages-2 rows cannot supply the same number because
    they are bandwidth-contaminated. At 8 CTAs nothing aggregate binds, so the
    contamination is gone and DEPTH 1 measures latency directly: with one box
    outstanding the warp waits for each arrival before issuing the next, so the
    interval IS the round k_tile_count.

    Reading it: stages 1 is latency, and the stages at which the interval stops
    falling is where the ring covers that latency. If that stages grows with the
    box, `stages: 4` is a box-dependent recommendation and tma.stages.warp.knee needs a
    bound rather than a number.
    """
    out = []
    for box_dim_1 in (64, 128, 256):              # 8 / 16 / 32 KB
        for stages in (1, 2, 3, 4, 6, 8):
            out.append(run_point(unit, buf, GEOMS["stride8k"], n_ctas=8,
                                 num_producers=1, stages=stages, box_dim_1=box_dim_1,
                                 buf_bytes=buf_bytes, dbg=dbg))
    return out



# E1/E2 live here. Both need a source axis the other sweeps take from the CLI,
# so each sets its own footprint and says which source the row came from.
L2_FOOTPRINT_B = 4 * 1024 * 1024        # << 50 MB L2, so the walk stays resident
DRAM_FOOTPRINT_B = BUF_B                # 256 MB, >2x L2, so it cannot be


def sweep_K(unit, buf, dbg, buf_bytes):
    """E1: is the ~133 GB/s ceiling per CTA, or per SM?

    Matched pairs at EQUAL per-SM in-flight bytes, equal box size and equal
    smem per SM -- only the CTA split moves:

        132 CTAs x 2 warps x F   (1 CTA/SM, each pinned AT the per-CTA cap)
        264 CTAs x 1 warp  x F   (2 CTA/SM, each BELOW the per-CTA cap)

    If the ceiling is per-SM the two match. If it is per-CTA the two-CTA row
    wins, because two sub-cap CTAs on one SM out-deliver one capped CTA.

    Why a full grid, when the per-CTA reading was taken at 8 CTAs where nothing
    else could bind: 264 is the SMALLEST grid that puts two CTAs on any SM at
    all. Below the SM count the scheduler spreads and the pair cannot be built.
    That is the tension this experiment is stuck with, and it is why every row
    reports its measured placement -- a row whose cta_per_sm_max is 1 did not
    test what it was built to test.
    """
    rows = []
    for src, fp in (("l2", L2_FOOTPRINT_B), ("dram", DRAM_FOOTPRINT_B)):
        for box_dim_1 in (32, 64, 128, 160, 192, 224):
            box_bytes = GEOMS["stride8k"].box_bytes(box_dim_1)
            for n_ctas, num_producers in ((N_SM, 2), (2 * N_SM, 1), (3 * N_SM, 1)):
                # Only ask for a residency the smem budget can actually hold.
                # A grid that wants 3/SM where 2 fit runs in TWO WAVES, and the
                # wave transition lands inside the measured window -- which
                # would look like a per-SM effect and is not one.
                want = n_ctas // N_SM
                if want * num_producers * 4 * box_bytes > SMEM_PER_SM:
                    continue
                smid = torch.full((n_ctas,), -1, dtype=torch.int32,
                                  device=buf.device)
                r = run_point(unit, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                              num_producers=num_producers, stages=4, box_dim_1=box_dim_1,
                              buf_bytes=fp, dbg=dbg, smid=smid)
                r["src"] = src
                # The quantity the pair is about: bytes in flight per SM, and
                # delivery per SM. Both are what the CTA split is supposed to
                # leave unchanged.
                if "skipped" not in r:
                    sms = r.get("sms_used") or N_SM
                    r["per_sm_kb"] = (n_ctas * num_producers * r["box_bytes"]) / sms / 1024
                    r["gbs_per_sm"] = r["gbs"] / sms
                    r["resident_cap"] = resident_cap(num_producers, 4, r["box_bytes"])
                    if r["cta_per_sm_max"] > r["resident_cap"]:
                        r["note"] = "MULTIWAVE"
                rows.append(r)
    return rows


def sweep_L(unit, buf, dbg, buf_bytes):
    """E2: does L2 have a ceiling of its own, and where?

    tma.bw.dev.l2 records 6.45 TB/s with no cap observed. That is a LOWER BOUND,
    not a constant, and quoting it as a ceiling is the mistake rule 5 exists to
    catch. This walks the product L2-resident from where DRAM saturates to
    several times past it, with the matching DRAM row beside each point so the
    divergence is visible rather than inferred.

    Flat to the end of the range -> the bound stands, and it is recorded as a
    bound with the range tried. A bend -> that is the ceiling, and it has points
    on both sides of it.
    """
    rows = []
    for n_ctas in (N_SM, 2 * N_SM, 4 * N_SM):
        for box_dim_1 in (64, 128, 160, 192, 224):
            for src, fp in (("l2", L2_FOOTPRINT_B), ("dram", DRAM_FOOTPRINT_B)):
                smid = torch.full((n_ctas,), -1, dtype=torch.int32,
                                  device=buf.device)
                r = run_point(unit, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                              num_producers=1, stages=4, box_dim_1=box_dim_1,
                              buf_bytes=fp, dbg=dbg, smid=smid)
                r["src"] = src
                if "skipped" not in r:
                    sms = r.get("sms_used") or N_SM
                    r["per_sm_kb"] = n_ctas * r["box_bytes"] / sms / 1024
                    r["gbs_per_sm"] = r["gbs"] / sms
                    r["resident_cap"] = resident_cap(1, 4, r["box_bytes"])
                    if r["cta_per_sm_max"] > r["resident_cap"]:
                        r["note"] = "MULTIWAVE"
                rows.append(r)
    return rows



def sweep_M(unit, buf, dbg, buf_bytes):
    """E3: how FEW SMs saturate each source?

    tma.bw.sm.dram was written as 23.5 GB/s per SM, which is just
    tma.bw.dev.dram / 132 -- an arithmetic restatement of a FULL grid, not a
    machine property. Saturating DRAM does not need every SM: at the per-SM
    ceiling of ~135 GB/s, 23 SMs already carry 3.1 TB/s. Whether per-SM
    delivery actually HOLDS at 135 as SMs are added is the question, and it is
    the one that decides how many SMs a copy has to occupy -- which is the same
    as asking how many are left for anything else.

    Fixed per-CTA product at 56 KB (2 warps x 28 KB, the configuration that
    reached the per-SM ceiling), one CTA per SM, walking the grid. Decisive
    shape: if per-SM delivery is flat, the aggregate rises linearly to the
    device ceiling and the knee is at ceiling/135 SMs. If it sags, the machine
    needs more SMs than the arithmetic predicts and the per-SM constant does
    not compose.
    """
    rows = []
    for src, fp in (("dram", DRAM_FOOTPRINT_B), ("l2", L2_FOOTPRINT_B)):
        for n_ctas in (4, 8, 16, 23, 32, 48, 64, 96, 132):
            smid = torch.full((n_ctas,), -1, dtype=torch.int32,
                              device=buf.device)
            r = run_point(unit, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                          num_producers=2, stages=4, box_dim_1=224,
                          buf_bytes=fp, dbg=dbg, smid=smid)
            r["src"] = src
            if "skipped" not in r:
                sms = r.get("sms_used") or n_ctas
                r["per_sm_kb"] = (n_ctas * 2 * r["box_bytes"]) / sms / 1024
                r["gbs_per_sm"] = r["gbs"] / sms
                r["resident_cap"] = resident_cap(2, 4, r["box_bytes"])
                if r["cta_per_sm_max"] > r["resident_cap"]:
                    r["note"] = "MULTIWAVE"
            rows.append(r)
    return rows



def sweep_N(unit, buf, dbg, buf_bytes):
    """E6: does the element WIDTH move the caps, or are they in bytes?

    Everything this unit measured was bf16. The two caps that decide tile size
    -- box row <= swizzle width, and boxDim[1] <= 256 -- are stated in bytes,
    and if they really are byte-based then halving the element width halves the
    ELEMENTS a legal box can carry, not the bytes. Ported naively that is a
    trap: a 64-element contiguous tile is a 128 B row in bf16 (SW128 legal,
    32 KB boxes) and a 64 B row in fp8 (SW128 illegal, 16 KB boxes), so the
    same tile in fp8 needs TWICE the transactions per byte and the precision
    saving cancels.

    N1 enumerates legality per (dtype, swizzle, box row BYTES). Decisive: if the
    accept/reject boundary lands at the same byte width for every element
    width, the caps are byte-based and the trap is real. If it tracks elements,
    it is not.

    N2 times the issue rate at BYTE-IDENTICAL geometry across element widths --
    same row bytes, same rows, same box, so the same transaction count and
    only the element width differs. tma.issue.warp is a per-transaction cost, so
    the rows must agree; a disagreement would mean the copy engine cares about
    the element width and every constant here is bf16-only.

    TMA has no fp8 type: fp8 is encoded as UINT8 and the tensor core does the
    interpreting. The 4-bit entries are Blackwell's; sm90's answer is measured
    here rather than assumed.
    """
    rows = []
    ptr = buf.data_ptr()
    for name, dt, eb in DTYPES:
        for sw in (SW_NONE, SW_32B, SW_64B, SW_128B):
            for row_b in (16, 32, 64, 128, 256):
                box_dim_0 = max(1, row_b // eb)
                # The tensor must be several boxes wide so the request is about
                # the box, not about an out-of-range extent.
                _, rc = encode(unit, ptr, dict(
                    global_dim_0=box_dim_0 * 16, global_dim_1=1024,
                    box_dim_0=box_dim_0, box_dim_1=8, swizzle=sw,
                    tensor_data_type=dt, elem_bytes=eb))
                rows.append(dict(kind="legality", dtype=name, elem_b=eb,
                                 swizzle=SW_NAME[sw], row_b=row_b,
                                 box_dim_0_elem=box_dim_0, rc=rc, ok=rc == 0))
    return rows


def sweep_O(unit, buf, dbg, buf_bytes):
    """E6 part 2: issue rate at byte-identical geometry across element widths.

    Row stride 8192 B and box row 128 B in EVERY case, so the descriptor
    describes the same bytes and the same transaction count; only the element
    width, and hence the element counts, differ. See sweep_N's docstring for
    what a disagreement would mean.
    """
    rows = []
    for name, dt, eb in DTYPES:
        if eb not in (1, 2, 4):
            continue
        geom = Geom(name, global_dim_0=8192 // eb,
                    box_dim_0=128 // eb, swizzle=SW_128B,
                    tensor_data_type=dt, elem_bytes=eb)
        for box_dim_1 in (64, 128, 224):
            r = run_point(unit, buf, geom, n_ctas=32, num_producers=1, stages=4,
                          box_dim_1=box_dim_1, buf_bytes=buf_bytes, dbg=dbg)
            r["dtype"] = name
            r["elem_b"] = eb
            rows.append(r)
    return rows



SWEEPS = {"A": sweep_A, "B": sweep_B, "C": sweep_C, "D": sweep_D,
          "E": sweep_E, "F": sweep_F, "G": sweep_G, "H": sweep_H,
          "I": sweep_I, "J": sweep_J, "K": sweep_K, "L": sweep_L, "M": sweep_M,
          "N": sweep_N, "O": sweep_O}



def render(rows: list[dict]) -> str:
    """One row per config. `ns/txn` is the per-warp issue interval -- the
    constant -- and `%ceil` measures against the MEASURED 2.77 TB/s ceiling,
    not the 3.35 datasheet peak, so a row at 100% means saturated rather than
    impossible.

    `MB` is what the walk TOUCHES, not the addressable range, and `xL2` is that
    over the 50 MB L2. A row marked `!` is within COLD_MIN_L2_RATIO of L2 and is
    NOT reliably cold: its result depends on what the preceding row left
    resident. Two byte-identical configurations once read 1.51x apart for
    exactly this reason."""
    hdr = (f"{'geom':>9} {'CTA':>4} {'wrp':>3} {'dep':>3} {'box':>7} "
           f"{'inflight':>9} {'MB':>7} {'us':>8} {'GB/s':>8} {'ns/txn':>7} "
           f"{'%peak':>6} {'%ceil':>6} {'xL2':>5}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        if "skipped" in r:
            lines.append(f"{r['geom']:>9} {r['n_ctas']:>4} {r['num_producers']:>3} "
                         f"{r['stages']:>3} {r['box_bytes']:>7} "
                         f"{'-- ' + r['skipped']:>40}")
            continue
        lines.append(
            f"{r['geom']:>9} {r['n_ctas']:>4} {r['num_producers']:>3} {r['stages']:>3} "
            f"{r['box_bytes']:>7} {r['inflight_kb']:>8.0f}K {r['total_mb']:>7.1f} "
            f"{r['us']:>8.2f} {r['gbs']:>8.1f} {r['ns_per_txn']:>7.1f} "
            f"{r['pct_peak']:>5.1f}% {100.0 * r['gbs'] / (BW_CEIL_GBS):>5.1f}% "
            f"{r.get('l2_ratio', 0):>4.1f}"
            + ("!" if r.get('l2_ratio', 9) < regime.COLD_MIN_L2_RATIO else " "))
    return "\n".join(lines)



def render_place(rows: list[dict]) -> str:
    """Render for E1/E2: source, measured placement, and the per-SM columns.

    `SM` and `/SM` are what the pair is about, and they are READ from %smid --
    a row whose `/SM` is 1 where 2 was intended did not test what it was built
    to test, and the number beside it must not be read as a per-SM result.
    """
    hdr = (f"{'src':>5} {'CTA':>4} {'wrp':>3} {'box':>7} {'SMs':>4} "
           f"{'/SM':>3} {'perSM_KB':>9} {'MB':>7} {'us':>8} {'GB/s':>8} "
           f"{'GB/s/SM':>8} {'ns/txn':>7} {'note':>9}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        if "skipped" in r:
            lines.append(f"{r.get('src','?'):>5} {r['n_ctas']:>4} "
                         f"{r['num_producers']:>3} {r['box_bytes']:>7} "
                         f"{'-- ' + r['skipped']:>44}")
            continue
        lines.append(
            f"{r.get('src','?'):>5} {r['n_ctas']:>4} {r['num_producers']:>3} "
            f"{r['box_bytes']:>7} {r.get('sms_used', 0):>4} "
            f"{r.get('cta_per_sm_max', 0):>3} {r.get('per_sm_kb', 0):>9.1f} "
            f"{r['total_mb']:>7.1f} {r['us']:>8.2f} {r['gbs']:>8.1f} "
            f"{r.get('gbs_per_sm', 0):>8.1f} {r['ns_per_txn']:>7.1f} "
            f"{r.get('note', ''):>9}")
    return "\n".join(lines)



def render_dtype(rows: list[dict]) -> str:
    """Legality as a (dtype x swizzle) grid over box row BYTES.

    If the caps are byte-based, every dtype block is IDENTICAL -- that identity
    is the result, and it is what makes the fp8 tile trap real.
    """
    if rows and rows[0].get("kind") != "legality":
        # sweep_O rows: a rate table keyed by element width
        hdr = (f"{'dtype':>15} {'B/el':>4} {'box_el':>7} {'box':>7} {'MB':>7} "
               f"{'us':>8} {'GB/s':>8} {'ns/txn':>7}")
        out = [hdr, "-" * len(hdr)]
        for r in rows:
            if "skipped" in r:
                out.append(f"{r['dtype']:>15} {r['elem_b']:>4}  -- {r['skipped']}")
                continue
            out.append(f"{r['dtype']:>15} {r['elem_b']:>4} "
                       f"{r['box_bytes'] // r['elem_b'] // 1:>7} {r['box_bytes']:>7} "
                       f"{r['total_mb']:>7.1f} {r['us']:>8.2f} {r['gbs']:>8.1f} "
                       f"{r['ns_per_txn']:>7.1f}")
        return "\n".join(out)
    widths = sorted({r["row_b"] for r in rows})
    out = [f"  {'dtype':>15} {'swizzle':>8}" + "".join(f"{w:>7}B" for w in widths)]
    out.append("  " + "-" * (24 + 8 * len(widths)))
    for name, _, _ in DTYPES:
        blk = [r for r in rows if r["dtype"] == name]
        if not blk:
            continue
        for sw in ("none", "32B", "64B", "128B"):
            cells = []
            for w in widths:
                m = next((r for r in blk if r["swizzle"] == sw and r["row_b"] == w),
                         None)
                cells.append("ok" if m and m["ok"] else
                             (f"rc{m['rc']}" if m else "-"))
            out.append(f"  {name if sw == 'none' else '':>15} {sw:>8}"
                       + "".join(f"{c:>8}" for c in cells))
    return "\n".join(out)


PLACE_SWEEPS = {"K", "L", "M"}
DTYPE_SWEEPS = {"N", "O"}



PLACE_SWEEPS = {"K", "L", "M"}
DTYPE_SWEEPS = {"N", "O"}




def main(argv=None):
    global TARGET_BYTES, FLUSH_L2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", default="A,B,C,D,E,F,G")
    ap.add_argument("--regime", choices=("dram", "l2"), default=None,
                    help="preset for the two regimes that ARE machine "
                         "constants. dram = large walk + L2 flushed. l2 = "
                         "footprint SMALLER than L2. A large walk WITHOUT the "
                         "flush is neither.")
    ap.add_argument("--flush-l2", action="store_true")
    ap.add_argument("--target-mb", type=int,
                    default=TARGET_BYTES // 1024 // 1024)
    ap.add_argument("--footprint-mb", type=int, default=BUF_MB)
    ap.add_argument("--json", default="")
    ap.add_argument("--reps", type=int, default=harness.REPS,
                    help="timed samples per point; the median is taken over "
                         "these. Recorded in the JSON -- a constant is only "
                         "comparable with one taken at the same count.")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    TARGET_BYTES = a.target_mb * 1024 * 1024
    FLUSH_L2 = a.flush_l2
    if a.regime == "dram":
        TARGET_BYTES = max(TARGET_BYTES, 256 * 1024 * 1024)
        FLUSH_L2, a.footprint_mb = True, BUF_MB
    elif a.regime == "l2":
        FLUSH_L2, a.footprint_mb = False, 4

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}",
          flush=True)

    harness.REPS = a.reps
    unit = harness.load(_SRC, verbose=a.verbose)
    print(f"[unit] {unit.name}  flags=0x{unit.flags:x}  "
          f"{unit.n_cfg} configs", flush=True)
    print(f"[regime] {a.regime or 'custom'}: footprint {a.footprint_mb} MB, "
          f"{TARGET_BYTES // 1024 // 1024} MB moved per launch, "
          f"L2 flush {'ON' if FLUSH_L2 else 'off'}", flush=True)

    buf = torch.empty(BUF_B // 2, dtype=torch.bfloat16, device="cuda")
    buf.normal_()
    dbg = torch.zeros(600 * 2, dtype=torch.int64, device="cuda")
    sm_id = torch.full((600,), -1, dtype=torch.int32, device="cuda")

    print("\n[Q4a] cuTensorMapEncodeTiled legality (box row bytes vs swizzle)",
          flush=True)
    legal = describe(unit, buf)
    widths = sorted({r["box_dim_0_b"] for r in legal})
    print("  " + "swizzle".rjust(8) + "".join(f"{w:>7}B" for w in widths))
    for sw in ("none", "32B", "64B", "128B"):
        cells = [next(("ok" if m["ok"] else f"rc{m['rc']}") for m in legal
                      if m["swizzle"] == sw and m["box_dim_0_b"] == w)
                 for w in widths]
        print("  " + sw.rjust(8) + "".join(f"{c:>8}" for c in cells))

    print("\n[Q4b] accepted box_dim[1] at 128 B rows, SW128 -- caps bytes/TMA",
          flush=True)
    rowlegal = describe_rows(unit, buf)
    for r in rowlegal:
        print(f"  box_dim[1]={r['box_dim_1']:>4}  box={r['box_bytes']:>6} B  "
              f"{'ok' if r['ok'] else 'rc%d' % r['rc']}")

    buf_bytes = a.footprint_mb * 1024 * 1024

    print("\n[Q0] descriptor / box / walk correctness -- gates every rate below",
          flush=True)
    checks = verify(unit, buf, dbg, buf_bytes)
    for r in checks:
        tail = (r["skipped"] if "skipped" in r else
                f"{'ok' if r['ok'] else 'FAIL'}  "
                f"coord_mismatch={r['bad_coord']} data_mismatch={r['bad_data']}")
        print(f"  {r['geom']:>8}  sw={r['swizzle']:>4}  {r['mode']:>8}  "
              f"box={r['box_bytes']:>6} B x {r['boxes']:>2}  {tail}", flush=True)
    if not all(r.get("ok", False) for r in checks if "skipped" not in r):
        raise SystemExit(
            "TMA correctness check FAILED -- the descriptor, the box geometry "
            "or the coordinate walk is wrong. Every rate below would be the "
            "right speed for the wrong bytes; refusing to measure.")

    results = {"unit": unit.name, "legality": legal, "row_legality": rowlegal,
               "checks": checks, "regime": a.regime, "flush_l2": FLUSH_L2,
               "target_mb": TARGET_BYTES // 1024 // 1024,
               "footprint_mb": a.footprint_mb, "sweeps": {}}
    for key in [s.strip().upper() for s in a.sweeps.split(",") if s.strip()]:
        if key not in SWEEPS:
            raise SystemExit(f"unknown sweep {key!r}; have {sorted(SWEEPS)}")
        print(f"\n[sweep {key}] {SWEEPS[key].__doc__.splitlines()[0]}",
              flush=True)
        rows = SWEEPS[key](unit, buf, dbg, buf_bytes)
        results["sweeps"][key] = rows
        renderer = (render_dtype if key in DTYPE_SWEEPS else
                    render_place if key in PLACE_SWEEPS else render)
        print(renderer(rows), flush=True)

    results["timer"] = harness.TIMER_USED
    results["reps"] = a.reps
    if harness.TIMER_USED != "cupti":
        print(f"\n[timer] {harness.TIMER_USED} -- includes launch overhead; "
              f"not comparable with constants recorded under CUPTI", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
        print(f"\n[json] {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
