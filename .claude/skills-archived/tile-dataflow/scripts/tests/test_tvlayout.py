#!/usr/bin/env python3
"""Cross-validate `tvlayout` against CUTLASS's own pycute, then run the known answers.

`tvlayout` reimplements CuTe's layout and swizzle algebra so the checker can run
on a login node with no CUDA and no torch. A reimplementation nobody checks is
worse than no tool at all -- a confidently wrong bank count is exactly the
failure the checker exists to remove -- so every function here is compared
against pycute over a spread of inputs whenever that checkout is present.

    python3 tests/test_tvlayout.py

Set PYCUTE_PATH to point somewhere else; the pycute comparisons skip (loudly)
when it cannot be imported, and the rest of the suite still runs.
"""

from __future__ import division, print_function

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import tvlayout as tvl  # noqa: E402

PYCUTE_PATH = os.environ.get(
    "PYCUTE_PATH", "/data/user/jzou521/codes/cuda/cutlass/python")

FAILURES = []


def check(cond, what):
    if cond:
        print("  ok   %s" % what)
    else:
        print("  FAIL %s" % what)
        FAILURES.append(what)


def load_pycute():
    if PYCUTE_PATH not in sys.path:
        sys.path.append(PYCUTE_PATH)
    try:
        import pycute
        return pycute
    except ImportError:
        return None


# ------------------------------------------------------------ cross-validation

def test_swizzle_vs_pycute(pycute):
    """The XOR layer, over every legal parameter triple we would ever emit."""
    rng = random.Random(20260822)
    bad = 0
    for bits in range(0, 4):
        for base in range(0, 6):
            for shift in range(max(bits, 1), 6):
                mine = tvl.Swizzle(bits, base, shift)
                theirs = pycute.Swizzle(bits, base, shift)
                for _ in range(200):
                    off = rng.randrange(0, 1 << 16)
                    if mine(off) != theirs(off):
                        bad += 1
    check(bad == 0, "Swizzle matches pycute over 4x6x5 params x 200 offsets")


def test_layout_vs_pycute(pycute):
    """Offsets for flat, nested and linear-coordinate lookups."""
    cases = [
        ((8, 128), (128, 1)),
        ((64, 512), (520, 1)),
        (((8, 8), 2), ((1, 16), 128)),
        (((4, 2), (8, 4)), ((1, 64), (4, 128))),
        ((128,), (1,)),
    ]
    rng = random.Random(7)
    bad = 0
    for shape, stride in cases:
        mine = tvl.Layout(shape, stride)
        theirs = pycute.Layout(shape, stride)
        if mine.size() != pycute.size(theirs) or mine.cosize() != pycute.cosize(theirs):
            bad += 1
        for _ in range(400):
            i = rng.randrange(0, mine.size())
            if mine(i) != theirs(i):          # linear coordinate
                bad += 1
    check(bad == 0, "Layout offset/size/cosize matches pycute over %d shapes" % len(cases))


# ------------------------------------------------------------- self-consistency

def test_swizzle_atoms():
    """The three parameters are forced by dtype and mode; pin the known atoms.

    These are the spellings CUTLASS uses for `Layout_K_SW*_Atom`, and they are
    the values every Hopper smem tile is built from.
    """
    want = {
        (128, 2): (3, 3, 3),    # bf16, the 128 B atom
        (128, 4): (3, 2, 3),    # f32
        (128, 1): (3, 4, 3),    # e4m3
        (64, 2): (2, 3, 3),     # bf16, the 64 B atom
        (32, 2): (1, 3, 3),     # bf16, the 32 B atom
    }
    for (mode, eb), expect in sorted(want.items()):
        s = tvl.swizzle_for(mode, eb)
        got = (s.bits, s.base, s.shift)
        check(got == expect, "swizzle_for(%dB, %dB elem) == Swizzle<%d,%d,%d>"
              % ((mode, eb) + expect))


def test_swizzle_is_a_permutation():
    """A swizzle must permute a tile, never collide two elements onto one slot.

    Cheap, and it is the property that would break silently: a wrong base or
    shift can still produce plausible-looking bank counts while corrupting data.
    """
    for dtype_bytes in (1, 2, 4):
        for mode in (32, 64, 128):
            cols = (mode // dtype_bytes) * 2
            tile = tvl.SwizzledTile(16, cols, dtype_bytes, mode)
            offs = [tile(r, c) for r in range(16) for c in range(cols)]
            check(len(set(offs)) == len(offs) and max(offs) == len(offs) - 1,
                  "SwizzledTile(16x%d, %dB elem, SW%d) is a bijection"
                  % (cols, dtype_bytes, mode))


def test_vector_granule_survives_swizzle():
    """16 B stays contiguous under every atom -- that is what `base` buys.

    If this ever fails, every 128-bit access in every spec built on the atom is
    silently illegal, so it is worth its own test rather than being implied.
    """
    for dtype_bytes in (1, 2, 4):
        elems_16B = 16 // dtype_bytes
        tile = tvl.SwizzledTile(16, elems_16B * 8, dtype_bytes, 128)
        ok = True
        for r in range(16):
            for c0 in range(0, elems_16B * 8, elems_16B):
                base = tile(r, c0)
                if base % elems_16B:
                    ok = False
                if any(tile(r, c0 + e) != base + e for e in range(elems_16B)):
                    ok = False
        check(ok, "16 B granule stays contiguous and aligned under SW128, %dB elems" % dtype_bytes)


def test_wgmma_acc_covers_the_tile():
    """The accumulator TV map must cover 64 x N exactly once per warp group.

    This atom decides the epilogue, the softmax span and which scale a thread
    needs; a map that double-covers or misses is the kind of error that shows up
    as a wrong answer in one corner of the output.
    """
    for n in (64, 128, 256):
        for vec in (1, 2):
            tv = tvl.wgmma_acc(n, vec=vec)
            seen = []
            for tid in range(tv.threads):
                for vid in range(tv.vals):
                    for e in range(tv.vec):
                        seen.append(tv.coord(tid, vid, e))
            check(len(seen) == 64 * n and len(set(seen)) == 64 * n,
                  "wgmma_acc(m64n%d, vec=%d) covers 64x%d exactly once" % (n, vec, n))


ACCESS_FILES = [
    ("known_answers.yaml", os.path.join(HERE, "known_answers.yaml")),
    ("references/accesses-deepgemm.yaml",
     os.path.join(HERE, "..", "..", "references", "accesses-deepgemm.yaml")),
    ("references/accesses-flashmla.yaml",
     os.path.join(HERE, "..", "..", "references", "accesses-flashmla.yaml")),
]


def test_known_answers():
    """The checker's own cases plus both worked specs' live access files.

    The specs' files are run here so their L4 tables cannot drift: an `expect`
    that stops matching is a spec whose prose now disagrees with its layout.
    """
    import tv_check
    for label, path in ACCESS_FILES:
        print("  -- %s --" % label)
        check(tv_check.run_file(path) == 0, "%s all pass" % label)


def main():
    print("tvlayout self-consistency")
    test_swizzle_atoms()
    test_swizzle_is_a_permutation()
    test_vector_granule_survives_swizzle()
    test_wgmma_acc_covers_the_tile()

    print("cross-validation against pycute")
    pycute = load_pycute()
    if pycute is None:
        print("  SKIP pycute not importable from %s -- set PYCUTE_PATH" % PYCUTE_PATH)
        FAILURES.append("pycute cross-validation SKIPPED (not a pass)")
    else:
        test_swizzle_vs_pycute(pycute)
        test_layout_vs_pycute(pycute)

    test_known_answers()

    print("")
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - %s" % f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
