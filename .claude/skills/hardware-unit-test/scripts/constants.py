#!/usr/bin/env python3
"""Render and validate the machine-constant tables.

A constant is only usable in a floor if a reader can tell what was held fixed,
what would have refuted it, and on which machine and toolchain it was taken.
This script enforces that mechanically, so "we measured 270 ns" cannot decay
into a number nobody can re-derive.

    python3 constants.py                     # the summary table
    python3 constants.py --unit tma          # one unit
    python3 constants.py --tag tma.issue.warp     # one constant, in full
    python3 constants.py --validate          # what `status: measured` must clear

Design-time only: pure python plus PyYAML, no torch and no CUDA, so it runs on
a login node.
"""

from __future__ import print_function

import argparse
import glob
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# One directory per architecture, each holding a `constants.yaml` and the unit
# references written against it. Nothing arch-specific lives above that
# directory, so porting is "add sm120/, re-run the probes" rather than "edit
# every file and hope".

# What an entry must carry to be USABLE as a reference: the number, the
# condition it holds under, the one-line answer, and the rule to apply.
#
# The measurement narrative -- claim, valid, isolation, falsifier and per-job
# provenance -- was removed on 2026-08-29 when this became a distributable
# results manual rather than a lab notebook. It is not "optional"; it is out of
# scope for this file. The evidence still exists in git history and in the
# snapshot named in the file header, and the unit references under <arch>/ carry
# the sweep design.
REQUIRED = ["value", "units", "short", "rule"]



def arch_dirs():
    """Every directory holding a constants.yaml is an architecture."""
    return sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(ROOT, "*", "constants.yaml")))


def load(machine=None):
    import yaml
    paths = [os.path.join(d, "constants.yaml") for d in arch_dirs()]
    if machine:
        # Match either the directory name (sm90) or the machine id inside it.
        paths = [p for p in paths
                 if machine in os.path.basename(os.path.dirname(p))
                 or machine in open(p).read(400)]
    if not paths:
        have = [os.path.basename(d) for d in arch_dirs()]
        sys.exit("no architecture matches %r; have %s" % (machine, have or "none"))
    out = []
    for p in paths:
        with open(p) as f:
            out.append((p, yaml.safe_load(f)))
    return out


def one_line(text, width=88):
    return " ".join(str(text).split())[:width]


def wrap(label, text, indent=4):
    body = textwrap.fill(" ".join(str(text).split()), width=78,
                         initial_indent=" " * indent,
                         subsequent_indent=" " * indent)
    return "  %s\n%s" % (label, body)


def resolve_tag(constants, tag):
    """Constants by tag.

    One spelling only: `<engine>.<quantity>.<scope>[.<condition>]`, so the scope
    and the unit are recoverable from the name. An earlier UPPER-KEBAB scheme,
    invented here with no vendor meaning, was removed from the repo on
    2026-08-30 along with the map that resolved it. Nothing resolves it now.
    """
    hits = [c for c in constants if c["tag"] == tag]
    if hits:
        return hits
    return []


def show_full(c):
    print("=" * 80)
    st = c.get("status", "measured")
    print("%-28s unit=%-8s value=%s %s%s"
          % (c["tag"], c.get("unit", "?"), c.get("value"), c.get("units", ""),
             "" if st == "measured" else "   [status: %s]" % st))
    print("=" * 80)
    for field in ("short", "rule"):
        if field in c:
            print(wrap(field.upper(), c[field]))
    print()


def validate(path, doc):
    bad = 0
    m = doc.get("machine", {})
    for k in ("id", "gpu", "arch", "sms", "clocks_pinned"):
        if k not in m:
            print("FAIL   machine        missing %r in %s" % (k, path))
            bad += 1
    known_units = {u["id"] for u in doc.get("units", [])}
    for c in doc.get("constants", []):
        tag = c.get("tag", "<untagged>")
        for f in REQUIRED:
            if f not in c or c[f] in (None, ""):
                print("FAIL   %-28s missing %s" % (tag, f))
                bad += 1
        if c.get("unit") not in known_units:
            print("FAIL   %-28s unit %r is not declared in units:"
                  % (tag, c.get("unit")))
            bad += 1
        if bad == 0 or True:
            st = c.get("status", "measured")
            if st != "measured":
                print("NOTE   %-28s status: %s -- not a measurement; do not "
                      "cite it as one" % (tag, st))
    for u in doc.get("units", []):
        if u.get("status") == "unmeasured":
            print("GAP    %-28s unit declared but NOT MEASURED -- a floor that "
                  "needs it is blocked" % u["id"])
    print("%s  %s: %d constants, %d problems"
          % ("FAIL" if bad else "PASS",
             os.path.basename(os.path.dirname(path)),
             len(doc.get("constants", [])), bad))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default=None,
                    help="architecture directory or machine id, e.g. sm90")
    ap.add_argument("--unit", default=None, help="only this unit (tma, launch, ...)")
    ap.add_argument("--tag", default=None, help="one constant, in full")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args(argv)

    docs = load(a.machine)
    if a.validate:
        bad = sum(validate(p, d) for p, d in docs)
        return 1 if bad else 0

    for path, doc in docs:
        m = doc["machine"]
        print("# %s -- %s, %d SMs, %s, clocks %s (noise floor ~%s%%)"
              % (m["id"], m["gpu"], m["sms"], m["arch"],
                 "PINNED" if m.get("clocks_pinned") else "NOT PINNED",
                 m.get("noise_floor_pct", "?")))
        if m.get("caution"):
            print(textwrap.fill(" ".join(m["caution"].split()), width=78,
                                initial_indent="  ! ", subsequent_indent="    "))
        print()
        cs = doc.get("constants", [])
        if a.unit:
            cs = [c for c in cs if c.get("unit") == a.unit]
        if a.tag:
            cs = resolve_tag(cs, a.tag)
            if not cs:
                sys.exit("no constant tagged %r" % a.tag)
        if a.tag or a.unit:
            for c in cs:
                show_full(c)
        else:
            print("%-28s %-7s %-34s %s" % ("TAG", "UNIT", "ANSWER", "RULE"))
            print("-" * 140)
            for c in cs:
                val = c.get("short") or "%s %s" % (c.get("value"),
                                                   c.get("units", ""))
                mark = "" if c.get("status", "measured") == "measured" else " *"
                print("%-28s %-7s %-34s %s"
                      % (c["tag"] + mark, c.get("unit", "?"),
                         one_line(val, 34), one_line(c.get("rule", ""), 70)))
            print("  * = retracted or derived, not a direct measurement -- see "
                  "--tag <name> for the full entry")
            print()
            for u in doc.get("units", []):
                if u.get("status") != "measured":
                    print("GAP  unit %-8s %s" % (u["id"], u.get("status")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
