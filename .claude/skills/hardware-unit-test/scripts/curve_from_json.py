#!/usr/bin/env python3
"""Reduce a tma_ring JSON to the bandwidth-vs-product curve, for <arch>/constants.yaml.

The probe's own claim is that delivered bandwidth is a function of the PRODUCT
`n_ctas * num_producers * box_bytes` alone. If that is true, every configuration at
a given product must land on one curve, and the spread within a product bin is
the law's own error bar -- which is the number that decides whether the curve
can be used to answer "how few CTAs" and "how small a frame" interchangeably.

    python3 curve_from_json.py profiles/hardware-unit-test/tma_frontier.json
    python3 curve_from_json.py <json> --yaml     # the block to paste into constants/

Emits the binned curve, the worst within-bin spread, and the configurations in
each bin so a reader can see the trade being asserted.

One filter is not optional. The product law holds only while a single CTA is
still absorbing everything sent to it: past ~36 KB of `num_producers x box_bytes` per CTA
a configuration above 32 KB per CTA sits in a
bin drags the curve down by up to 26% while looking like an ordinary point. So
rows above `--max-per-cta` are EXCLUDED and counted, never silently binned.
"""

from __future__ import print_function

import argparse
import json
import sys


# Rows written before the vocabulary migration carry the retired field names.
# Archived JSON is still worth plotting, so read the current name and fall back
# to the retired one rather than silently matching nothing -- which is what this
# did after the rename: `r.get("depth")` returned None for every migrated row
# and the curve came out empty with no error. [references/vocabulary.md]
_RETIRED = {"stages": "depth", "num_producers": "n_warps", "box_bytes": "frame_b"}


def field(r, name):
    if name in r:
        return r[name]
    old = _RETIRED.get(name)
    if old is not None and old in r:
        return r[old]
    raise KeyError(f"row has neither {name!r} nor its retired name {old!r}")


def rows(doc, stages=4, geom="stride8k", max_per_cta=32768):
    out, dropped = [], []
    for _, rs in sorted(doc.get("sweeps", {}).items()):
        for r in rs:
            if "skipped" in r or field(r, "stages") != stages \
                    or r.get("geom") != geom:
                continue
            if field(r, "num_producers") * field(r, "box_bytes") > max_per_cta:
                dropped.append(r)
                continue
            out.append(r)
    return out, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json")
    ap.add_argument("--yaml", action="store_true")
    ap.add_argument("--tol", type=float, default=0.06,
                    help="bin width as a fraction of product (default: the "
                         "6%% noise floor)")
    ap.add_argument("--max-per-cta", type=int, default=32768,
                    help="exclude rows whose num_producers x box_bytes exceeds this; "
                         "past the per-CTA ceiling the product law does not "
                         "hold (default 32768 B)")
    a = ap.parse_args(argv)

    doc = json.load(open(a.json))
    kept, dropped = rows(doc, max_per_cta=a.max_per_cta)
    if dropped:
        combos = sorted({"%dwx%dK" % (field(r, "num_producers"), field(r, "box_bytes") // 1024)
                         for r in dropped})
        print("excluded %d rows above %d B per CTA (%s) -- above "
              "the product law does not hold\n"
              % (len(dropped), a.max_per_cta, " ".join(combos)))
    pts = []
    for r in kept:
        prod = r["n_ctas"] * field(r, "num_producers") * field(r, "box_bytes")
        pts.append((prod, r["gbs"], r["ns_per_txn"],
                    "%dx%dx%dK" % (r["n_ctas"], field(r, "num_producers"),
                                   field(r, "box_bytes") // 1024)))
    pts.sort()

    # Bin on the product axis: anything within `tol` of the bin's first point
    # is the same product as far as this machine's noise floor can tell.
    bins, cur = [], []
    for p in pts:
        if cur and p[0] > cur[0][0] * (1 + a.tol):
            bins.append(cur)
            cur = []
        cur.append(p)
    if cur:
        bins.append(cur)

    print("%-12s %-9s %-9s %-8s %-7s %s"
          % ("product", "GB/s med", "spread", "ns/txn", "n", "configs"))
    print("-" * 96)
    curve, worst = [], 0.0
    for b in bins:
        gs = sorted(x[1] for x in b)
        med = gs[len(gs) // 2]
        spread = 100.0 * (gs[-1] - gs[0]) / med if len(gs) > 1 else 0.0
        worst = max(worst, spread)
        nsm = sorted(x[2] for x in b)[len(b) // 2]
        prod_kb = b[0][0] / 1024.0
        curve.append((round(prod_kb), round(med, 1)))
        print("%-12s %-9.1f %-9s %-8.1f %-7d %s"
              % ("%.0f KB" % prod_kb, med, "%.1f%%" % spread, nsm, len(b),
                 " ".join(x[3] for x in b[:6])))
    print("\nworst within-bin spread: %.1f%%  (noise floor is ~6%%)" % worst)

    if a.yaml:
        print("\n# --- paste into <arch>/constants.yaml ---")
        print("  - id: tma-bw-vs-product")
        print("    x: product_kb   # n_ctas * num_producers * box_bytes / 1024")
        print("    y: gbs")
        print("    points:")
        for k, v in curve:
            print("      - [%d, %.1f]" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
