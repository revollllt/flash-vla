#!/usr/bin/env python3
"""Turn the measured machine constants into the two numbers a tiling decision asks.

    "The tile is large -- how few SMs still saturate HBM?"
    "Every SM is busy -- how small can one CTA's TMA be and still saturate?"

Two measured facts answer both.

  [tma.bw.dev.curve]  Aggregate delivery is a function of the PRODUCT
                  `n_ctas * num_producers * box_bytes` alone -- 22 bins, +-6.9%.
  [tma.issue.warp]   ONE CTA delivers linearly in bytes in flight, at
                  `num_producers * box_bytes`. Beyond that a CTA absorbs no faster.

so   delivered = min( n_ctas * per_cta(num_producers * box_bytes),  curve(product) )

The two terms bind in different places: the per-warp issue rate on small grids with
fat CTAs, the curve everywhere else. This script reads both out of `constants/`
and says which one is binding, because the answer changes with it.

    python3 frontier.py --table
    python3 frontier.py --min-ctas  --box-bytes 32768 [--warps 2] [--target 0.90]
    python3 frontier.py --min-box --ctas 132 [--warps 2]
    python3 frontier.py --copy-floor --txns-per-warp 32 --bytes 4194304 [--launches 1]

Design-time only: pure python plus PyYAML, no torch and no CUDA.
"""

from __future__ import print_function

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as K  # noqa: E402

FRAME_CAP = 32768   # [tma.bytes.txn.max]: 128 B row x 256 rows under SW128


def _sz(b):
    """Bytes as KB or MiB. In-flight budgets are powers of two, so a decimal
    'MB' here would silently disagree with the tile arithmetic upstream."""
    return "%.0f KB" % (b / 1024.0) if b < 2**20 else "%.2f MiB" % (b / 2.0**20)


def get(doc, tag):
    """A constant by tag, refusing anything not fit to compute with.

    A `status` other than measured means the entry is not fit to compute with.
    An entry that has been retracted still resolves by tag, so nothing stops a
    model reading it unless this check does.
    """
    for c in doc.get("constants", []):
        if c["tag"] == tag:
            st = c.get("status")
            if st == "retracted":
                sys.exit("%s is RETRACTED and must not be computed with: %s"
                         % (tag, c.get("rule", "")))
            if st:
                print("# note: %s is %s, not a direct measurement" % (tag, st))
            return c
    sys.exit("constants file has no %s" % tag)


class Model(object):
    def __init__(self, doc):
        self.doc = doc
        self.m = doc["machine"]
        self.t_issue = float(get(doc, "tma.issue.warp")["value"])          # ns
        # GB/s, as the constant states its units. The x1000 that used to be
        # here dated from when this entry was recorded in TB/s (3.02); it now
        # reads 3172 GB/s, and the stale multiplier put every target three
        # orders of magnitude above the measured curve -- which the tool then
        # reported as "above the measured curve" rather than as its own bug.
        self.ceil = float(get(doc, "tma.bw.dev.dram")["value"])
        # NO per-CTA ceiling term: delivery is linear at the per-warp issue
        # interval to at least 40 KB. What bounds a configuration is the DEVICE
        # curve below, which applies whatever the per-CTA split is.
        cur = [c for c in doc.get("curves", []) if c["id"] == "tma-bw-vs-product"]
        if not cur:
            sys.exit("constants file has no tma-bw-vs-product curve")
        self.spread = float(cur[0].get("spread_pct_worst", 0.0))
        # Running maximum: more in flight cannot deliver less, so a bin below a
        # smaller-product bin is noise. Applied here rather than smoothed into
        # the constants file, which keeps the measured medians.
        self.pts, best = [], 0.0
        for x, y in cur[0]["points"]:
            best = max(best, float(y))
            self.pts.append((float(x) * 1024.0, best))
        self.lo, self.hi = self.pts[0][0], self.pts[-1][0]

    # ---------------------------------------------------------------- forward
    def per_cta(self, warps, box_bytes):
        # Linear in bytes in flight, with no plateau -- see the note in
        # __init__ about the absent per-CTA ceiling.
        return warps * box_bytes / self.t_issue

    def curve(self, product):
        if product <= self.pts[0][0]:
            return self.pts[0][1] * product / self.pts[0][0]
        for (x0, y0), (x1, y1) in zip(self.pts, self.pts[1:]):
            if product <= x1:
                return y0 + (y1 - y0) * (product - x0) / (x1 - x0)
        return self.pts[-1][1]

    def bw(self, ctas, warps, box_bytes):
        a = ctas * self.per_cta(warps, box_bytes)
        b = self.curve(ctas * warps * box_bytes)
        return (min(a, b), "issue rate" if a < b else "product curve")

    # ---------------------------------------------------------------- inverse
    def product_for(self, target):
        if target > self.pts[-1][1]:
            return None
        if target <= self.pts[0][1]:
            return self.pts[0][0] * target / self.pts[0][1]
        for (x0, y0), (x1, y1) in zip(self.pts, self.pts[1:]):
            if target <= y1:
                return x0 + (x1 - x0) * (target - y0) / (y1 - y0)
        return None

    # ----------------------------------------------------------------- caveats
    def notes(self, ctas=None, warps=None, box_bytes=None, product=None):
        out = []
        if product and product > self.hi:
            out.append("%s in flight is ABOVE the measured curve (<= %s): "
                       "extrapolation, not a reading" % (_sz(product), _sz(self.hi)))
        if ctas and warps and box_bytes and warps * box_bytes > 32768 and ctas < 96:
            out.append("TRANSITION REGION: a large per-CTA budget on a small "
                       "grid. Measured up to 15% BELOW this model there "
                       "(48 CTAs x 2 warps x 28 KB predicts 2747 GB/s, "
                       "measures 2535) -- treat it as an upper bound")
        out.append("the curve's own error bar is +-%.1f%%, and its residual "
                   "favours more CTAs at equal product [tma.bw.dev.curve]"
                   % self.spread)
        return out


def _target(mo, frac, strict=True):
    """The product needed for `frac` of the device ceiling, or (t, None).

    `strict` exits, which is right for a single-answer query. The table asks for
    several fractions at once and one being off the top of the measured curve is
    a fact about that column, not a reason to print nothing -- which is what it
    did before, killing the whole table on the 99% column.
    """
    t = frac * mo.ceil
    p = mo.product_for(t)
    if p is None:
        if strict:
            sys.exit("%.0f GB/s (%.0f%% of tma.bw.dev.dram) is above the "
                     "measured curve" % (t, frac * 100))
        return t, None
    return t, p


def cmd_min_ctas(mo, box_bytes, warps, frac):
    t, prod = _target(mo, frac)
    by_cta = t / mo.per_cta(warps, box_bytes)      # issue-rate term
    by_curve = prod / (warps * box_bytes)          # product term
    need = max(by_cta, by_curve)
    print("min CTAs for %.0f%% of tma.bw.dev.dram (%.0f GB/s), %d producer warp(s) x "
          "%d B:" % (frac * 100, t, warps, box_bytes))
    print("    %.0f CTAs   (%.0f%% of this machine's %d SMs%s)"
          % (need + 0.5, 100.0 * need / mo.m["sms"], mo.m["sms"],
             "" if need <= mo.m["sms"] else " -- MORE THAN ONE CTA PER SM"))
    print("    %-18s %6.0f CTAs   (one CTA issues %.0f GB/s at this config)"
          % ("issue rate:", by_cta, mo.per_cta(warps, box_bytes)))
    print("    %-18s %6.0f CTAs   (needs %s in flight)"
          % ("product curve:", by_curve, _sz(prod)))
    print("    binding: %s"
          % ("issue rate" if by_cta > by_curve else "product curve"))
    if box_bytes > FRAME_CAP:
        print("    ! box_bytes exceeds the %d B descriptor maximum "
              "[tma.bytes.txn.max]" % FRAME_CAP)
    for n in mo.notes(need, warps, box_bytes, prod):
        print("    ! %s" % n)


def cmd_min_box(mo, ctas, warps, frac):
    t, prod = _target(mo, frac)
    print("min bytes per TMA for %.0f%% of tma.bw.dev.dram (%.0f GB/s), at %d CTAs x "
          "%d producer warp(s):" % (frac * 100, t, ctas, warps))
    need = prod / (ctas * warps)
    print("    %.0f B  (%.1f KB)  = %d box rows of 128 B under SW128"
          % (need, need / 1024.0, -(-need // 128)))
    if need > FRAME_CAP:
        best, _ = mo.bw(ctas, warps, FRAME_CAP)
        print("    ! above the %d B descriptor maximum [tma.bytes.txn.max]. At the "
              "cap this configuration delivers ~%.0f GB/s (%.0f%%); use %d "
              "producer warps of %.1f KB instead."
              % (FRAME_CAP, best, 100.0 * best / mo.ceil,
                 int(-(-need // FRAME_CAP)) * warps,
                 need / (-(-need // FRAME_CAP))/ 1024.0))
    if warps * need > 49152:
        print("    ! this puts %.1f KB per CTA in flight. Past ~48 KB per CTA "
              "nothing improves at any grid size (measured 3018 -> 3006 GB/s "
              "for 48 -> 56 KB at 132 CTAs) -- this budget is not reachable by "
              "spending more smem" % (warps * need / 1024.0))
    for n in mo.notes(ctas, warps, need, prod):
        print("    ! %s" % n)


def cmd_copy_floor(mo, txns, bytes_, launches):
    issue_us = txns * mo.t_issue / 1000.0
    bw_us = bytes_ / mo.ceil / 1000.0
    floor = max(issue_us, bw_us)
    print("copy-column floor:  (issue term is PER WARP; bandwidth term is "
          "GRID-TOTAL -- check you passed --bytes for the whole grid)")
    print("    issue-limited  %8.2f us  = %d txns/warp x %.0f ns   [tma.issue.warp]"
          % (issue_us, txns, mo.t_issue))
    print("    bandwidth-lim  %8.2f us  = %.2f MB / %.2f TB/s      [tma.bw.dev.dram]"
          % (bw_us, bytes_ / 1e6, mo.ceil / 1000.0))
    print("    floor          %8.2f us  (%s binds)"
          % (floor, "issue rate" if issue_us > bw_us else "bandwidth"))
    if launches:
        ramp = float(get(mo.doc, "launch.lat.dev.ramp")["value"]) * launches
        print("    + grid ramp    %8.2f us  = %d x %.2f us          "
              "[launch.lat.dev.ramp]" % (ramp, launches, ramp / launches))
        print("    TOTAL          %8.2f us" % (floor + ramp))
    print("    ! the bandwidth term uses tma.bw.dev.dram (steady state). For a SHORT "
          "kernel whose grid ramp is a real share of its time, use ld.bw.dev.dram's "
          "1.85 + MB/2.77 instead -- 9% more pessimistic, and right there.")
    print("    ! txns/warp = K_per_CTA / BK. It moves with BK, split-K and "
          "producer-warp count; it does NOT move with CTA count, because every "
          "CTA still walks its own K. [tma.issue.warp]")


def cmd_table(mo):
    print("# Saturation frontier on %s -- from the MEASURED curve" % mo.m["id"])
    print("# delivered = min(n_ctas x per_cta,  curve(product))")
    print("# per_cta   = num_producers x box_bytes / %.0f ns  [tma.issue.warp], "
          "linear, no plateau" % mo.t_issue)
    print()
    print("what ONE CTA can pull  [tma.issue.warp]")
    print("    %-12s %-11s %s" % ("num_producers x box", "GB/s", "note"))
    for pc in (8192, 16384, 32768, 36864, 49152):
        note = ("1 warp reaches this" if pc <= FRAME_CAP else
                "needs 2+ warps (one warp caps at the 32 KB box)")
        print("    %-12s %-11.1f %s" % ("%.0f KB" % (pc / 1024),
                                        mo.per_cta(1, pc), note))
    print("    -> delivery is LINEAR in bytes in flight: a second producer warp "
          "roughly doubles it. There is no per-CTA plateau.")
    print()
    print("Q1  large tile: min CTAs to reach a fraction of tma.bw.dev.dram")
    print("    %-22s %-12s %-12s %s"
          % ("producer config", "90% of ceil", "95%", "99%"))
    for warps, box_bytes in ((1, 8192), (1, 16384), (1, 32768),
                         (2, 8192), (2, 16384), (2, 18432)):
        cells = []
        for frac in (0.90, 0.95, 0.99):
            t, p = _target(mo, frac, strict=False)
            if p is None:
                cells.append("off curve")
                continue
            n = max(t / mo.per_cta(warps, box_bytes), p / (warps * box_bytes))
            cells.append("%.0f%s" % (n + 0.5, "" if n <= mo.m["sms"] else "*"))
        print("    %-22s %-12s %-12s %s"
              % ("%d warp x %.0f KB" % (warps, box_bytes / 1024), *cells))
    print("    * = more CTAs than this machine has SMs, so it needs >1 CTA/SM")
    print("    'off curve' = that fraction of the ceiling is above the highest "
          "product the curve measured; it is not reachable by adding CTAs")
    print()
    print("Q2  full grid: min bytes per TMA per warp")
    print("    %-18s %-14s %-14s %s"
          % ("producers", "90% of ceil", "95%", "99%"))
    for ctas, warps in ((66, 1), (132, 1), (132, 2), (264, 1), (264, 2)):
        cells = []
        for frac in (0.90, 0.95, 0.99):
            t, p = _target(mo, frac, strict=False)
            if p is None:
                cells.append("off curve")
                continue
            f = p / (ctas * warps)
            cells.append("%.1f KB%s" % (f / 1024, "" if f <= FRAME_CAP else " X"))
        print("    %-18s %-14s %-14s %s"
              % ("%d CTA x %d warp" % (ctas, warps), *cells))
    print("    X = above the descriptor maximum at this warp count; split it "
          "across more producer warps")
    print("    'off curve' = above the highest product the curve measured")
    print()
    print("! %s" % mo.notes()[-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default="sm90")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--min-ctas", action="store_true")
    ap.add_argument("--min-box", action="store_true")
    ap.add_argument("--copy-floor", action="store_true")
    ap.add_argument("--box-bytes", type=int, default=8192, help="bytes per TMA")
    ap.add_argument("--ctas", type=int, default=132)
    ap.add_argument("--warps", type=int, default=1, help="producer warps per CTA")
    ap.add_argument("--target", type=float, default=0.90,
                    help="fraction of tma.bw.dev.dram to reach (default 0.90)")
    ap.add_argument("--txns-per-warp", type=int, default=0)
    ap.add_argument("--bytes", type=int, default=0,
                    help="TOTAL bytes the whole grid moves, not per CTA -- the "
                         "bandwidth term is a grid-wide ceiling while the issue "
                         "term is per warp, and mixing the two scopes is the "
                         "easy mistake here")
    ap.add_argument("--launches", type=int, default=0)
    a = ap.parse_args(argv)

    mo = Model(K.load(a.machine)[0][1])
    if a.min_ctas:
        cmd_min_ctas(mo, a.box_bytes, a.warps, a.target)
    elif a.min_box:
        cmd_min_box(mo, a.ctas, a.warps, a.target)
    elif a.copy_floor:
        if not a.txns_per_warp or not a.bytes:
            sys.exit("--copy-floor needs --txns-per-warp and --bytes")
        cmd_copy_floor(mo, a.txns_per_warp, a.bytes, a.launches)
    else:
        cmd_table(mo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
