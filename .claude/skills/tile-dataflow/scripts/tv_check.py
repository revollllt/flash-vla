#!/usr/bin/env python3
"""Compute an L4 per-thread access table instead of asserting one.

Every number a spec's PER-THREAD ACCESS block claims -- bits/thread, vector
width, transactions, bank-conflict ways -- is a function of two maps: the
buffer's layout and the thread-value map of the access. This tool takes those
two maps and enumerates a warp. 32 lanes times a handful of values is a few
hundred points, so the answer is exact by construction, not inferred.

    python3 tv_check.py accesses.yaml            # human table
    python3 tv_check.py accesses.yaml --markdown # paste into the spec's L4
    python3 tv_check.py --self-test              # the shipped known answers

An `expect` block turns an access into a regression test: it says "I already
know this number". Exit status is 1 when a computed value contradicts an
`expect`, or when an access WITHOUT an `expect` fails its bank/sector check --
so a deliberately-bad counterfactual can be pinned down without failing the run,
while an unannotated access still gates `status: review`.

THE BANK MODEL, stated because "N-way conflict" is ambiguous for wide accesses.
Shared memory serves one distinct 4 B word per bank per cycle, broadcasting to
every lane that wants that same word. So for one instruction issued by one warp:

    wavefronts = max over banks of (distinct words landing in that bank)
    ideal      = ceil(distinct words in the whole request / 32)
    serialisation = wavefronts / ideal

`wavefronts` is the number usually written as "N-way", but N-way is only a
defect relative to `ideal`. A 64-bit access moves 2 words per lane, so 2 words
per bank is optimal -- reporting it as "2-way conflict" is the mistake this tool
exists to stop. The DeepGEMM epilogue below is 8 against an ideal of 2: a real
4x, and the reason `sm90.hpp` keeps BLOCK_N off multiples of 32.

For global memory the granule is a 32 B sector rather than a bank:

    sectors = distinct (byte_addr >> 5) over the warp
    ideal   = ceil(distinct bytes touched / 32)
"""

from __future__ import division, print_function

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvlayout import (Layout, SwizzledTile, expression, linear,  # noqa: E402
                      swizzle_for, wgmma_acc)

WARP = 32
BANKS = 32
WORD_B = 4
SECTOR_B = 32

DTYPE_BYTES = {
    "f64": 8, "fp64": 8, "double": 8,
    "f32": 4, "fp32": 4, "float": 4, "tf32": 4, "i32": 4, "u32": 4,
    "bf16": 2, "f16": 2, "fp16": 2, "half": 2, "i16": 2,
    "e4m3": 1, "e5m2": 1, "fp8": 1, "i8": 1, "u8": 1, "int8": 1,
}

# Units that move data without any per-thread address. Naming them is the point:
# a Hopper spec that describes a TMA load per-thread has described a kernel it
# did not write.
NO_PER_THREAD_UNITS = {
    "tma": ("cp.async.bulk.tensor", "one elected thread issues the whole tile; "
            "the copy engine writes smem"),
    "wgmma-desc": ("wgmma.mma_async", "operands are read through a matrix descriptor, "
                   "not by ld.shared"),
    "tcgen05": ("tcgen05.mma / tcgen05.cp", "CTA-pair MMA reads smem/TMEM directly"),
}


# ------------------------------------------------------------------ analysis

def _lane_words(offset_fn, tv, *, tid, vid, dtype_bytes):
    """Byte span and covered 4 B words for one lane's single instruction.

    Returns (base_byte, span_bytes, words, contiguous). `contiguous` is False
    when the `vec` elements the TV map claims are not actually adjacent after
    swizzling -- the check that catches a vector width a swizzle cannot support.
    """
    addrs = [offset_fn(tv.coord(tid, vid, e)) * dtype_bytes for e in range(tv.vec)]
    base = addrs[0]
    contiguous = all(a == base + i * dtype_bytes for i, a in enumerate(addrs))
    span = tv.vec * dtype_bytes
    lo, hi = base // WORD_B, (base + span + WORD_B - 1) // WORD_B
    return base, span, tuple(range(lo, hi)), contiguous


def analyse_instruction(offset_fn, tv, *, vid, warp, dtype_bytes, space):
    """Bank/sector behaviour of one instruction issued by one warp.

    Pure: takes the two maps, returns a dict of counted facts. `warp` indexes
    warps inside `tv.threads`, because warp 0 and warp 3 of a warp group do not
    always behave the same and the worst one is what matters.
    """
    lanes = range(warp * WARP, min((warp + 1) * WARP, tv.threads))
    pairs, byte_set, misaligned, noncontig = [], set(), [], []
    for tid in lanes:
        base, span, words, contiguous = _lane_words(
            offset_fn, tv, tid=tid, vid=vid, dtype_bytes=dtype_bytes)
        if not contiguous:
            noncontig.append(tid)
        if span in (2, 4, 8, 16) and base % span:
            misaligned.append(tid)
        for w in words:
            pairs.append((tid, w))
        byte_set.update(range(base, base + span))

    distinct = set(w for _, w in pairs)
    per_bank = {}
    for w in distinct:
        per_bank.setdefault(w % BANKS, set()).add(w)

    out = {
        "requests": len(pairs),
        "distinct_words": len(distinct),
        "banks_touched": len(per_bank),
        "broadcast": len(pairs) / len(distinct) if distinct else 0.0,
        "noncontiguous_lanes": tuple(noncontig),
        "misaligned_lanes": tuple(misaligned),
    }
    if space == "smem":
        out["wavefronts"] = max((len(v) for v in per_bank.values()), default=0)
        out["ideal"] = int(math.ceil(len(distinct) / BANKS)) if distinct else 0
    else:
        sectors = set(b // SECTOR_B for b in byte_set)
        out["sectors"] = len(sectors)
        out["ideal"] = int(math.ceil(len(byte_set) / SECTOR_B)) if byte_set else 0
        out["bytes_touched"] = len(byte_set)
    out["serialisation"] = (
        (out.get("wavefronts", out.get("sectors", 0)) / out["ideal"]) if out["ideal"] else 1.0)
    return out


def analyse_access(offset_fn, tv, *, dtype_bytes, space):
    """Worst-case over every instruction and every warp, plus the totals.

    Returns (worst, totals). `worst` is the instruction/warp with the highest
    serialisation -- that is the one a reviewer needs to see. `totals` sums the
    real cost over the whole access so two designs can be compared.
    """
    n_warps = max(1, tv.threads // WARP)
    worst, total_cost, total_ideal = None, 0, 0
    for vid in range(tv.vals):
        for warp in range(n_warps):
            r = analyse_instruction(offset_fn, tv, vid=vid, warp=warp,
                                    dtype_bytes=dtype_bytes, space=space)
            cost = r.get("wavefronts", r.get("sectors", 0))
            total_cost += cost
            total_ideal += r["ideal"]
            if worst is None or r["serialisation"] > worst[0]["serialisation"]:
                worst = (r, vid, warp)
    # Distinct key names: `ideal` already means the per-instruction ideal, and
    # letting the totals overwrite it silently turned every expect into a lie.
    totals = {"total_cost": total_cost, "total_ideal": total_ideal,
              "warps": n_warps, "instructions": tv.vals}
    return worst, totals


# --------------------------------------------------------------- spec inputs

def build_buffer(cfg, *, dtype_bytes):
    """Build the element-offset map from an access file's `buffer` block."""
    kind = cfg.get("kind", "tile" if "rows" in cfg else "layout")
    if kind == "tile":
        sw = cfg.get("swizzle", "none")
        tile = SwizzledTile(int(cfg["rows"]), int(cfg["cols"]), dtype_bytes, sw)
        desc = "tile %dx%d, swizzle %s, row stride %d B" % (
            cfg["rows"], cfg["cols"], sw, int(cfg["cols"]) * dtype_bytes)
        return tile, desc
    layout = Layout(tuple(cfg["shape"]), tuple(cfg["stride"]))
    sw = cfg.get("swizzle", "none")
    if sw in (None, "none", 0):
        return layout, "layout %s" % layout
    swz = swizzle_for(sw, dtype_bytes) if not isinstance(sw, (list, tuple)) else None
    if swz is None:
        swz = __import__("tvlayout").Swizzle(*sw)

    def fn(*crd):
        return swz(layout(*crd))

    return fn, "layout %s o %s" % (layout, swz)


def build_tv(cfg):
    """Build the (tid, vid) -> coord map from an access file's `tv` block."""
    threads = int(cfg.get("threads", 128))
    if "atom" in cfg:
        name = cfg["atom"]
        if name == "wgmma_acc":
            return wgmma_acc(int(cfg["n"]), vec=int(cfg.get("vec", 1)),
                             project=cfg.get("project", "both"), threads=threads)
        if name == "linear":
            return linear(int(cfg["cols"]), threads, int(cfg["vals"]),
                          vec=int(cfg.get("vec", 1)), rows=cfg.get("rows"))
        raise ValueError("unknown tv atom %r" % name)
    return expression(cfg["expr"], threads, int(cfg["vals"]),
                      vec=int(cfg.get("vec", 1)), name=cfg.get("name", "expr"))


def dtype_bytes_of(name):
    if name not in DTYPE_BYTES:
        raise ValueError("unknown dtype %r; known: %s" % (name, sorted(DTYPE_BYTES)))
    return DTYPE_BYTES[name]


# ------------------------------------------------------------------- report

def check_access(acc):
    """Run one access entry.

    Returns (lines, status, results). `status` separates the two questions a
    reviewer asks: does the access behave well (`perf`), and does it behave the
    way the spec claimed (`expect`). A counterfactual pinned with an `expect`
    block is allowed to be slow; an access nobody annotated is not.
    """
    aid = acc.get("id", "?")
    unit = acc.get("unit", "thread")
    if unit in NO_PER_THREAD_UNITS:
        inst, why = NO_PER_THREAD_UNITS[unit]
        return (["%-24s NO per-thread access -- %s." % (aid, why),
                 "%-24s   %s; nothing to count here, and the spec should say so"
                 % ("", inst)],
                {"perf": True, "expect": True, "has_expect": False}, {"unit": unit})

    dtb = dtype_bytes_of(acc["dtype"])
    offset_fn, bdesc = build_buffer(acc["buffer"], dtype_bytes=dtb)
    tv = build_tv(dict(acc["tv"], threads=acc.get("threads", acc["tv"].get("threads", 128))))
    space = acc.get("space", "smem")
    worst, totals = analyse_access(offset_fn, tv, dtype_bytes=dtb, space=space)
    r, vid, warp = worst

    bits = tv.vec * dtb * 8
    ok = True                      # the bank/sector and lowering verdict
    ok_expect = True               # whether the computed values match the spec's claims
    lines = ["=== %s ===" % aid,
             "  buffer   %s" % bdesc,
             "  tv       %s, %d threads x %d inst x %d elem" % (
                 tv.name, tv.threads, tv.vals, tv.vec),
             "  access   %s %d b/thread (%d B)" % (
                 acc.get("op", "ld"), bits, tv.vec * dtb)]

    if r["noncontiguous_lanes"]:
        ok = False
        lines.append("           NOT CONTIGUOUS at vec=%d (lanes %s...) -- the %d elements this "
                     "TV map\n           claims per instruction are not adjacent after swizzling; "
                     "vec must drop" % (tv.vec, list(r["noncontiguous_lanes"])[:4], tv.vec))
    if r["misaligned_lanes"]:
        ok = False
        lines.append("           MISALIGNED base for a %d B access (lanes %s...)"
                     % (tv.vec * dtb, list(r["misaligned_lanes"])[:4]))

    if space == "smem":
        verdict = "PASS" if r["serialisation"] <= 1.0 else "FAIL"
        ok = ok and r["serialisation"] <= 1.0
        lines.append("  banks    wavefronts %d   ideal %d   -> %.3gx serialisation   %s"
                     % (r["wavefronts"], r["ideal"], r["serialisation"], verdict))
        lines.append("           worst inst %d, warp %d: %d distinct words over %d banks, "
                     "broadcast %.3gx" % (vid, warp, r["distinct_words"],
                                          r["banks_touched"], r["broadcast"]))
        lines.append("  total    %d inst x %d warps -> %d bank cycles (ideal %d)"
                     % (totals["instructions"], totals["warps"],
                    totals["total_cost"], totals["total_ideal"]))
    else:
        verdict = "PASS" if r["serialisation"] <= 1.0 else "FAIL"
        ok = ok and r["serialisation"] <= 1.0
        lines.append("  gmem     sectors %d   ideal %d   -> %.3gx over-fetch   %s"
                     % (r["sectors"], r["ideal"], r["serialisation"], verdict))
        lines.append("           worst inst %d, warp %d: %d bytes touched, broadcast %.3gx"
                     % (vid, warp, r["bytes_touched"], r["broadcast"]))
        lines.append("  total    %d inst x %d warps -> %d sectors (ideal %d)"
                     % (totals["instructions"], totals["warps"],
                    totals["total_cost"], totals["total_ideal"]))

    got = dict(r)
    got.update(totals)
    for key, want in (acc.get("expect") or {}).items():
        have = got.get(key)
        if have != want:
            ok_expect = False
            lines.append("  EXPECT   %s: spec says %r, computed %r  <-- MISMATCH" % (key, want, have))
        else:
            lines.append("  expect   %s == %r  ok" % (key, want))
    status = {"perf": ok, "expect": ok_expect, "has_expect": bool(acc.get("expect"))}
    return lines, status, got


def markdown_row(acc, got):
    """One row of the spec's L4 PER-THREAD ACCESS table, ready to paste."""
    if got.get("unit") in NO_PER_THREAD_UNITS:
        inst, why = NO_PER_THREAD_UNITS[got["unit"]]
        return "| %s | n/a | n/a | NO per-thread access -- %s |" % (acc.get("id", "?"), why)
    dtb = dtype_bytes_of(acc["dtype"])
    vec = int(acc["tv"].get("vec", 1))
    if acc.get("space", "smem") == "smem":
        note = ("%d-wavefront, ideal %d -> %.3gx" %
                (got["wavefronts"], got["ideal"], got["serialisation"]))
    else:
        note = "%d sectors, ideal %d -> %.3gx" % (got["sectors"], got["ideal"], got["serialisation"])
    return "| %s | %d b/thread | %d inst x %d warps | %s |" % (
        acc.get("id", "?"), vec * dtb * 8, got["instructions"], got["warps"], note)


def evaluate_file(path):
    """Run every access in a file. Returns (all_ok, report_lines, table_rows, failed_ids).

    Split out from `run_file` so `budget.py` can fold an L4 check into the spec's
    check block without capturing stdout.
    """
    import yaml
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    accesses = doc["accesses"] if isinstance(doc, dict) else doc
    all_ok, rows, out, bad = True, [], [], []
    for acc in accesses:
        lines, status, got = check_access(acc)
        # A pinned counterfactual may be slow on purpose; an unannotated access
        # may not. Either way a contradicted `expect` is always a failure.
        entry_ok = status["expect"] and (status["perf"] or status["has_expect"])
        if not entry_ok:
            bad.append(acc.get("id", "?"))
        all_ok = all_ok and entry_ok
        out.extend(lines + [""])
        rows.append(markdown_row(acc, got))
    return all_ok, out, rows, bad


def run_file(path, *, markdown=False):
    all_ok, out, rows, bad = evaluate_file(path)
    accesses = rows
    if markdown:
        print("| touch | width | count | banks / sectors |")
        print("|---|---|---|---|")
        for row in rows:
            print(row)
    else:
        print("\n".join(out))
    if all_ok:
        print("ALL PASS  %d access(es)" % len(accesses))
    else:
        print("FAIL  %d of %d access(es): %s" % (len(bad), len(accesses), ", ".join(bad)))
    return 0 if all_ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("access_file", nargs="?", help="YAML access file; see references/l4-access.md")
    p.add_argument("--markdown", action="store_true", help="emit the L4 table row form")
    p.add_argument("--self-test", action="store_true", help="run the shipped known answers")
    args = p.parse_args(argv)
    if args.self_test:
        here = os.path.dirname(os.path.abspath(__file__))
        return run_file(os.path.join(here, "tests", "known_answers.yaml"))
    if not args.access_file:
        p.error("give an access file, or --self-test")
    return run_file(args.access_file, markdown=args.markdown)


if __name__ == "__main__":
    sys.exit(main())
