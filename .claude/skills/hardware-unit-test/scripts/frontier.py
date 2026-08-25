#!/usr/bin/env python3
"""Turn the measured machine constants into the two numbers a tiling decision asks.

    "The tile is large -- how few SMs still saturate HBM?"
    "Every SM is busy -- how small can one CTA's TMA be and still saturate?"

Two measured facts answer both.

  [TMA-FRONTIER]  Aggregate delivery is a function of the PRODUCT
                  `n_ctas * n_warps * frame_bytes` alone -- 22 bins, +-6.9%.
  [TMA-CTA-CEIL]  ONE CTA saturates at ~133 GB/s, reached at ~36 KB of
                  `n_warps * frame`. Beyond that a CTA absorbs no faster.

so   delivered = min( n_ctas * per_cta(n_warps * frame),  curve(product) )

The two terms bind in different places: the per-CTA ceiling on small grids with
fat CTAs, the curve everywhere else. This script reads both out of `constants/`
and says which one is binding, because the answer changes with it.

    python3 frontier.py --table
    python3 frontier.py --min-ctas  --frame 32768 [--warps 2] [--target 0.90]
    python3 frontier.py --min-frame --ctas 132 [--warps 2]
    python3 frontier.py --copy-floor --txns-per-warp 32 --bytes 4194304 [--launches 1]

Design-time only: pure python plus PyYAML, no torch and no CUDA.
"""

from __future__ import print_function

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as K  # noqa: E402

FRAME_CAP = 32768   # [TMA-FRAME-CAP]: 128 B row x 256 rows under SW128


def _sz(b):
    """Bytes as KB or MiB. In-flight budgets are powers of two, so a decimal
    'MB' here would silently disagree with the tile arithmetic upstream."""
    return "%.0f KB" % (b / 1024.0) if b < 2**20 else "%.2f MiB" % (b / 2.0**20)


def get(doc, tag):
    for c in doc.get("constants", []):
        if c["tag"] == tag:
            return c
    sys.exit("constants file has no %s" % tag)


class Model(object):
    def __init__(self, doc):
        self.doc = doc
        self.m = doc["machine"]
        self.t_issue = float(get(doc, "TMA-ISSUE")["value"])          # ns
        self.ceil = float(get(doc, "TMA-CEIL")["value"]) * 1000.0     # GB/s
        self.cta_ceil = float(get(doc, "TMA-CTA-CEIL")["value"])      # GB/s
        self.knee = self.cta_ceil * self.t_issue          # bytes per CTA
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
    def per_cta(self, warps, frame):
        return min(warps * frame / self.t_issue, self.cta_ceil)

    def curve(self, product):
        if product <= self.pts[0][0]:
            return self.pts[0][1] * product / self.pts[0][0]
        for (x0, y0), (x1, y1) in zip(self.pts, self.pts[1:]):
            if product <= x1:
                return y0 + (y1 - y0) * (product - x0) / (x1 - x0)
        return self.pts[-1][1]

    def bw(self, ctas, warps, frame):
        a = ctas * self.per_cta(warps, frame)
        b = self.curve(ctas * warps * frame)
        return (min(a, b), "per-CTA ceiling" if a < b else "product curve")

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
    def notes(self, ctas=None, warps=None, frame=None, product=None):
        out = []
        if product and product > self.hi:
            out.append("%s in flight is ABOVE the measured curve (<= %s): "
                       "extrapolation, not a reading" % (_sz(product), _sz(self.hi)))
        if ctas and warps and frame and warps * frame > self.knee and ctas < 96:
            out.append("TRANSITION REGION: per-CTA budget is above the %.0f KB "
                       "knee and the grid is small. Measured up to 15%% BELOW "
                       "this model there (48 CTAs x 2 warps x 28 KB predicts "
                       "2747 GB/s, measures 2535) -- treat it as an upper bound"
                       % (self.knee / 1024))
        out.append("the curve's own error bar is +-%.1f%%, and its residual "
                   "favours more CTAs at equal product [TMA-FRONTIER]"
                   % self.spread)
        return out


def _target(mo, frac):
    t = frac * mo.ceil
    p = mo.product_for(t)
    if p is None:
        sys.exit("%.0f GB/s (%.0f%% of TMA-CEIL) is above the measured curve"
                 % (t, frac * 100))
    return t, p


def cmd_min_ctas(mo, frame, warps, frac):
    t, prod = _target(mo, frac)
    by_cta = t / mo.per_cta(warps, frame)      # per-CTA ceiling term
    by_curve = prod / (warps * frame)          # product term
    need = max(by_cta, by_curve)
    print("min CTAs for %.0f%% of TMA-CEIL (%.0f GB/s), %d producer warp(s) x "
          "%d B:" % (frac * 100, t, warps, frame))
    print("    %.0f CTAs   (%.0f%% of this machine's %d SMs%s)"
          % (need + 0.5, 100.0 * need / mo.m["sms"], mo.m["sms"],
             "" if need <= mo.m["sms"] else " -- MORE THAN ONE CTA PER SM"))
    print("    %-18s %6.0f CTAs   (one CTA carries %.0f GB/s)"
          % ("per-CTA ceiling:", by_cta, mo.per_cta(warps, frame)))
    print("    %-18s %6.0f CTAs   (needs %s in flight)"
          % ("product curve:", by_curve, _sz(prod)))
    print("    binding: %s"
          % ("per-CTA ceiling" if by_cta > by_curve else "product curve"))
    if warps * frame > mo.knee and by_cta > by_curve:
        # Only wasted when the per-CTA ceiling is what binds. On a large grid
        # the curve binds instead and surplus per-CTA bytes still raise the
        # product -- measured +4.0% for 32 -> 48 KB per CTA at 132 CTAs.
        print("    ! %.0f KB per CTA is above the %.0f KB knee AND the per-CTA "
              "ceiling is binding: %.0f KB of that smem buys no bandwidth here "
              "[TMA-CTA-CEIL]"
              % (warps * frame / 1024.0, mo.knee / 1024.0,
                 (warps * frame - mo.knee) / 1024.0))
    if frame > FRAME_CAP:
        print("    ! frame exceeds the %d B descriptor maximum "
              "[TMA-FRAME-CAP]" % FRAME_CAP)
    for n in mo.notes(need, warps, frame, prod):
        print("    ! %s" % n)


def cmd_min_frame(mo, ctas, warps, frac):
    t, prod = _target(mo, frac)
    print("min bytes per TMA for %.0f%% of TMA-CEIL (%.0f GB/s), at %d CTAs x "
          "%d producer warp(s):" % (frac * 100, t, ctas, warps))
    if t / ctas > mo.cta_ceil:
        print("    ! UNREACHABLE AT THIS GRID, at any frame or warp count: it "
              "needs %.0f GB/s per CTA and a CTA saturates at %.0f "
              "[TMA-CTA-CEIL]. Add CTAs." % (t / ctas, mo.cta_ceil))
        return
    need = prod / (ctas * warps)
    print("    %.0f B  (%.1f KB)  = %d box rows of 128 B under SW128"
          % (need, need / 1024.0, -(-need // 128)))
    if need > FRAME_CAP:
        best, _ = mo.bw(ctas, warps, FRAME_CAP)
        print("    ! above the %d B descriptor maximum [TMA-FRAME-CAP]. At the "
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
    print("copy-column floor:")
    print("    issue-limited  %8.2f us  = %d txns/warp x %.0f ns   [TMA-ISSUE]"
          % (issue_us, txns, mo.t_issue))
    print("    bandwidth-lim  %8.2f us  = %.2f MB / %.2f TB/s      [TMA-CEIL]"
          % (bw_us, bytes_ / 1e6, mo.ceil / 1000.0))
    print("    floor          %8.2f us  (%s binds)"
          % (floor, "issue rate" if issue_us > bw_us else "bandwidth"))
    if launches:
        ramp = float(get(mo.doc, "LAUNCH-RAMP")["value"]) * launches
        print("    + grid ramp    %8.2f us  = %d x %.2f us          "
              "[LAUNCH-RAMP]" % (ramp, launches, ramp / launches))
        print("    TOTAL          %8.2f us" % (floor + ramp))
    print("    ! the bandwidth term uses TMA-CEIL (steady state). For a SHORT "
          "kernel whose grid ramp is a real share of its time, use BW-CEIL's "
          "1.85 + MB/2.77 instead -- 9%% more pessimistic, and right there.")
    print("    ! txns/warp = K_per_CTA / BK. It moves with BK, split-K and "
          "producer-warp count; it does NOT move with CTA count, because every "
          "CTA still walks its own K. [TMA-ISSUE]")


def cmd_table(mo):
    print("# Saturation frontier on %s -- from the MEASURED curve" % mo.m["id"])
    print("# delivered = min(n_ctas x per_cta,  curve(product))")
    print("# per_cta   = min(n_warps x frame / %.0f ns, %.0f GB/s), knee at "
          "%.0f KB per CTA" % (mo.t_issue, mo.cta_ceil, mo.knee / 1024))
    print()
    print("what ONE CTA can pull  [TMA-CTA-CEIL]")
    print("    %-12s %-11s %s" % ("n_warps x frame", "GB/s", "note"))
    for pc in (8192, 16384, 32768, 36864, 49152):
        note = ("1 warp reaches this" if pc <= FRAME_CAP else
                "needs 2+ warps (one warp caps at 32 KB)")
        if pc > mo.knee:
            note += "; surplus smem buys nothing"
        print("    %-12s %-11.1f %s" % ("%.0f KB" % (pc / 1024),
                                        mo.per_cta(1, pc), note))
    print("    -> the second producer warp is worth %.0f%%; the third is worth "
          "nothing" % (100 * (mo.cta_ceil / mo.per_cta(1, FRAME_CAP) - 1)))
    print()
    print("Q1  large tile: min CTAs to reach a fraction of TMA-CEIL")
    print("    %-22s %-12s %-12s %s"
          % ("producer config", "90% of ceil", "95%", "99%"))
    for warps, frame in ((1, 8192), (1, 16384), (1, 32768),
                         (2, 8192), (2, 16384), (2, 18432)):
        cells = []
        for frac in (0.90, 0.95, 0.99):
            t, p = _target(mo, frac)
            n = max(t / mo.per_cta(warps, frame), p / (warps * frame))
            cells.append("%.0f%s" % (n + 0.5, "" if n <= mo.m["sms"] else "*"))
        print("    %-22s %-12s %-12s %s"
              % ("%d warp x %.0f KB" % (warps, frame / 1024), *cells))
    print("    * = more CTAs than this machine has SMs, so it needs >1 CTA/SM")
    print()
    print("Q2  full grid: min bytes per TMA per warp")
    print("    %-18s %-14s %-14s %s"
          % ("producers", "90% of ceil", "95%", "99%"))
    for ctas, warps in ((66, 1), (132, 1), (132, 2), (264, 1), (264, 2)):
        cells = []
        for frac in (0.90, 0.95, 0.99):
            t, p = _target(mo, frac)
            if t / ctas > mo.cta_ceil:
                cells.append("none G")
                continue
            f = p / (ctas * warps)
            cells.append("%.1f KB%s" % (f / 1024, "" if f <= FRAME_CAP else " X"))
        print("    %-18s %-14s %-14s %s"
              % ("%d CTA x %d warp" % (ctas, warps), *cells))
    print("    X = above the descriptor maximum at this warp count; split it "
          "across more producer warps")
    print("    G = unreachable at this GRID at any frame or warp count -- the "
          "CTAs saturate first [TMA-CTA-CEIL]")
    print()
    print("! %s" % mo.notes()[-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default="sm90")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--min-ctas", action="store_true")
    ap.add_argument("--min-frame", action="store_true")
    ap.add_argument("--copy-floor", action="store_true")
    ap.add_argument("--frame", type=int, default=8192, help="bytes per TMA")
    ap.add_argument("--ctas", type=int, default=132)
    ap.add_argument("--warps", type=int, default=1, help="producer warps per CTA")
    ap.add_argument("--target", type=float, default=0.90,
                    help="fraction of TMA-CEIL to reach (default 0.90)")
    ap.add_argument("--txns-per-warp", type=int, default=0)
    ap.add_argument("--bytes", type=int, default=0)
    ap.add_argument("--launches", type=int, default=0)
    a = ap.parse_args(argv)

    mo = Model(K.load(a.machine)[0][1])
    if a.min_ctas:
        cmd_min_ctas(mo, a.frame, a.warps, a.target)
    elif a.min_frame:
        cmd_min_frame(mo, a.ctas, a.warps, a.target)
    elif a.copy_floor:
        if not a.txns_per_warp or not a.bytes:
            sys.exit("--copy-floor needs --txns-per-warp and --bytes")
        cmd_copy_floor(mo, a.txns_per_warp, a.bytes, a.launches)
    else:
        cmd_table(mo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
