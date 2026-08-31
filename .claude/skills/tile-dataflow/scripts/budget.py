#!/usr/bin/env python3
"""Run a tile-dataflow spec's consistency arithmetic, instead of doing it in your head.

SKILL.md's "Consistency arithmetic" table is a list of equations that must
balance. Asking a model to evaluate two dozen of them from memory, mid-spec, is
the least reliable step in the whole process and the one most easily made exact:
every check here is division and comparison over fields the spec already states.

    python3 budget.py path/to/spec.md
    python3 budget.py path/to/spec.md --sms 132       # for occupancy/persistence
    python3 budget.py path/to/spec.md --quiet         # only failures

Verdicts:
    PASS   the equation balances
    FAIL   it does not; the spec is wrong, not the check
    TIGHT  it balances with little headroom -- the thing to lead a hand-back with
    SKIP   a field it needs is still TODO or deleted
    MANUAL this check is not arithmetic and a human still has to read it

Exit status is 1 if anything FAILs. A SKIP is not a pass: `status: review`
requires every check to be PASS, TIGHT or MANUAL, which `--gate` enforces.

Design rule for adding a check
------------------------------
A check earns a FAIL only if it evaluates an equation over DECLARED fields.
If it has to parse prose to find its operand, it is a linter, and it belongs at
MANUAL: prose regexes look rigorous, fire where the question does not arise, and
train the author to satisfy the pattern instead of thinking about the property.
When a check genuinely needs to find something mechanically, change the SCHEMA
so the field carries it -- a list instead of a sentence -- rather than making
the prose rigid. Two checks were demoted on exactly this ground (2026-08-25):
`loop_bounds` (its real content is already in `trip_count`, arithmetically) and
`loop_carried` (it substring-matched a free-text field, so "C" matched "C1" and
it reported PASS while verifying nothing).
"""

from __future__ import division, print_function

import argparse
import collections
import os
import re
import sys

Finding = collections.namedtuple("Finding", "name verdict detail")

# Per-CTA shared memory caps. The per-SM figures (167936 / 102400 / 233472) are
# a different number and mixing them up buys a stage that does not exist.
SMEM_CAP_B = {
    "sm80": 166912, "sm86": 101376, "sm89": 101376,
    "sm90": 232448, "sm90a": 232448, "sm100": 232448, "sm100a": 232448,
}
SMEM_PER_SM_B = {
    "sm80": 167936, "sm86": 102400, "sm89": 102400,
    "sm90": 233472, "sm90a": 233472, "sm100": 233472, "sm100a": 233472,
}
REGS_PER_SM = 65536
MAX_REGS_PER_THREAD = 255

DTYPE_BYTES = {
    "f64": 8, "fp64": 8, "f32": 4, "fp32": 4, "float": 4, "tf32": 4, "i32": 4,
    "bf16": 2, "f16": 2, "fp16": 2, "half": 2,
    "e4m3": 1, "e5m2": 1, "fp8": 1, "i8": 1, "u8": 1,
}

# Published dense ridge points, FLOP/byte, by (arch family, operand dtype). The
# dtype matters more than the arch: fp8 doubles the FLOPs and halves the bytes
# against bf16, so one bf16 number condemns every fp8 tile ever written.
# Marked [I] wherever used -- Phase 0 exists because datasheet denominators are
# not what a machine reaches, so a spec that measured its own should put it in
# `toolchain.measured.ridge_point` and leave this table unused.
RIDGE_PUBLISHED = {
    ("sm90", 2): 295.0, ("sm90", 1): 591.0, ("sm90", 4): 148.0,
    ("sm100", 2): 295.0, ("sm100", 1): 591.0, ("sm100", 4): 148.0,
    ("sm80", 2): 153.0, ("sm80", 4): 77.0,
}


def ridge_for(arch, dtype):
    family = arch.rstrip("a")
    return RIDGE_PUBLISHED.get((family, DTYPE_BYTES.get(dtype, 2)))


# --------------------------------------------------------------- spec loading

class DuplicateKeyError(Exception):
    pass


def _load_yaml_strict(text):
    """Parse YAML, but reject duplicate keys instead of silently keeping the last.

    Not a theoretical hazard: `assets/spec-template.md` shipped two different
    `primitive:` fields in one `non_mma` entry -- one for the named primitive,
    one for the reduction mechanism -- and stock PyYAML keeps only the second,
    so half the contract vanished at parse time with no diagnostic.
    """
    import yaml

    class StrictLoader(yaml.SafeLoader):
        pass

    def no_duplicates(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise DuplicateKeyError(
                    "duplicate key %r at line %d (an earlier value is silently discarded)"
                    % (key, key_node.start_mark.line + 1))
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates)
    return yaml.load(text, StrictLoader)


def load_spec(path):
    """Split a spec into (spec dict, prose body). Both are checked.

    Two shapes are accepted, because both are in use: a spec written from
    `assets/spec-template.md` opens with `---` front matter, while the worked
    examples in `references/` carry the same YAML in a fenced ```yaml block
    under a prose introduction.
    """
    with open(path) as fh:
        text = fh.read()
    if text.startswith("---"):
        parts = text.split("\n---", 1)
        if len(parts) != 2:
            raise ValueError("%s front matter is not terminated by a --- line" % path)
        return _load_yaml_strict(parts[0].lstrip("-\n")), parts[1]
    m = re.search(r"^```ya?ml\n(.*?)^```", text, re.S | re.M)
    if not m:
        raise ValueError("%s has neither --- front matter nor a ```yaml block" % path)
    return _load_yaml_strict(m.group(1)), text[:m.start()] + text[m.end():]


# ----------------------------------------------------------------- accessors

def get(spec, dotted, default=None):
    """Fetch `a.b.c`, treating TODO / None / deleted as absent."""
    node = spec
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    if node is None:
        return default
    if isinstance(node, str) and ("TODO" in node or not node.strip()):
        return default
    return node


def as_int(value):
    """Pull a leading integer out of `132 (= num_sms)` and friends; None if absent."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    m = re.match(r"\s*(-?\d+)", str(value))
    return int(m.group(1)) if m else None


def geti(spec, dotted):
    return as_int(get(spec, dotted))


def arch_of(spec):
    return str(get(spec, "arch", "sm90a")).strip()


def math_groups(spec):
    """Warp groups that issue an MMA, by name reference from `math[].group`."""
    groups = get(spec, "warp_groups") or []
    named = set(str(m.get("group")) for m in (get(spec, "math") or []) if m.get("group"))
    hits = [g for g in groups if str(g.get("id")) in named]
    if hits:
        return hits
    # No `math` section to cross-reference: fall back to the naming convention.
    return [g for g in groups if str(g.get("id", "")).startswith("math")]


def contraction_extent(spec, entry):
    """Extent of the axis a math entry reduces, from its `contracts` field.

    Returns (extent, name), or (None, None) when the field is absent or its axis
    cannot be resolved. `contracts` accepts an axis name resolved through
    `problem.dims`, or an inline `name=extent`.
    """
    raw = get(entry, "contracts")
    if raw is None:
        return None, None
    text = str(raw).strip()
    if "=" in text:
        name, _, rhs = text.partition("=")
        extent = as_int(rhs)
        return (extent, name.strip()) if extent is not None else (None, None)
    axis = text.split()[0]
    extent = geti(spec, "problem.dims.%s" % axis)
    return (extent, axis) if extent is not None else (None, None)


# -------------------------------------------------------------------- checks

def check_smem(spec, opts):
    depth = geti(spec, "pipeline.depth")
    staged = get(spec, "pipeline.staged_buffers") or []
    non_staged = get(spec, "pipeline.non_staged_buffers") or []
    if depth is None or not staged:
        return Finding("smem", "SKIP", "pipeline.depth or staged_buffers still TODO")
    per_stage = sum(as_int(b.get("bytes")) or 0 for b in staged)
    # Sum every buffer's stated `bytes`. An aliasing buffer carries bytes: 0
    # because its storage belongs to the buffer it overlaps -- the `aliases`
    # string names the partner and is read by a human, it is not a second way to
    # zero the cost. Dropping every buffer that mentions an alias also drops the
    # OWNER, which is how FlashMLA's 73728 B sQ went missing.
    extra = sum(as_int(b.get("bytes")) or 0 for b in non_staged)
    aliased = [b.get("name") for b in non_staged
               if str(b.get("aliases", "none")).strip() not in ("none", "None", "", "n/a")]
    total = depth * per_stage + extra
    cap = SMEM_CAP_B.get(arch_of(spec))
    if cap is None:
        return Finding("smem", "SKIP", "unknown arch %r" % arch_of(spec))
    spare = cap - total
    detail = ("%d B = %d stages x %d + %d non-staged / %d B cap, %d B spare"
              % (total, depth, per_stage, extra, cap, spare))
    if aliased:
        detail += ("; %s declare an alias -- each alias_safe_because is a correctness "
                   "argument a human still has to read" % aliased)
    stated = geti(spec, "grid.launch.smem_B")
    if stated is not None and stated != total:
        return Finding("smem", "FAIL", detail + "; grid.launch.smem_B says %d" % stated)
    if spare < 0:
        return Finding("smem", "FAIL", detail)
    return Finding("smem", "TIGHT" if spare < 4096 else "PASS", detail)


def check_staged_bytes(spec, opts):
    """Each buffer's stated `bytes` against its own shape x dtype."""
    bufs = (get(spec, "pipeline.staged_buffers") or [])
    bad, checked = [], 0
    for b in bufs:
        shape, dt, stated = b.get("shape"), b.get("dtype"), as_int(b.get("bytes"))
        if not shape or dt not in DTYPE_BYTES or stated is None:
            continue
        elems = 1
        for s in shape:
            if as_int(s) is None:
                elems = None
                break
            elems *= as_int(s)
        if elems is None:
            continue
        checked += 1
        want = elems * DTYPE_BYTES[dt]
        if want != stated:
            bad.append("%s: %s %s = %d B, spec says %d" % (b.get("name"), shape, dt, want, stated))
    if not checked:
        return Finding("staged_bytes", "SKIP", "no buffer has shape+dtype+bytes all filled")
    if bad:
        return Finding("staged_bytes", "FAIL", "; ".join(bad))
    return Finding("staged_bytes", "PASS", "%d buffer(s) match shape x dtype" % checked)


def check_threads(spec, opts):
    launch = geti(spec, "grid.launch.threads")
    groups = get(spec, "warp_groups") or []
    if launch is None or not groups:
        return Finding("threads", "SKIP", "grid.launch.threads or warp_groups still TODO")
    total, notes = 0, []
    for g in groups:
        thr, warps = as_int(g.get("threads")), as_int(g.get("warps"))
        if thr is None:
            return Finding("threads", "SKIP", "warp group %r has no thread count" % g.get("id"))
        total += thr
        if warps is not None and warps * 32 != thr:
            notes.append("%s: %d warps != %d threads" % (g.get("id"), warps, thr))
        issues = str(g.get("issues", "")) + str(g.get("role", ""))
        if "wgmma" in issues and thr % 128:
            notes.append("%s issues wgmma with %d threads (not a whole warp group)"
                         % (g.get("id"), thr))
    detail = "sum(warp_groups.threads) = %d, launch %d" % (total, launch)
    if notes:
        return Finding("threads", "FAIL", detail + "; " + "; ".join(notes))
    return Finding("threads", "PASS" if total == launch else "FAIL", detail)


def check_acc_registers(spec, opts):
    m, n = geti(spec, "grid.cta_tile.M"), geti(spec, "grid.cta_tile.N")
    groups = math_groups(spec)
    if m is None or n is None or not groups:
        return Finding("acc_registers", "SKIP", "cta_tile or math warp groups still TODO")
    math_threads = sum(as_int(g.get("threads")) or 0 for g in groups)
    if not math_threads:
        return Finding("acc_registers", "SKIP", "math warp groups have no thread counts")
    if str(get(spec, "math.0.acc.location", "")).upper() == "TMEM":
        return Finding("acc_registers", "PASS",
                       "accumulator is in TMEM; it does not compete for RF")
    per_thread = m * n / math_threads
    detail = "%g f32/thread = %dx%d / %d math threads" % (per_thread, m, n, math_threads)
    stated = as_int(get(spec, "math.0.acc.elems_per_thread"))
    if stated is not None and stated != int(per_thread):
        detail += "; math[0].acc.elems_per_thread says %d" % stated
        return Finding("acc_registers", "FAIL", detail)
    if per_thread > MAX_REGS_PER_THREAD:
        return Finding("acc_registers", "FAIL", detail + " -- exceeds 255 before operands")
    if per_thread > 200:
        return Finding("acc_registers", "TIGHT",
                       detail + " -- past the ~200 spill cliff once operands and addressing land")
    return Finding("acc_registers", "PASS", detail)


def regs_allocated_per_cta(spec):
    """Registers a CTA really holds, or None when nothing reconfigures them.

    On sm90+ `setmaxnreg` makes the launch bound's `max_regs_per_thread` a
    pre-redistribution ceiling rather than the allocation, so
    `threads x max_regs x cta_per_sm` is the wrong product for any
    warp-specialized kernel: DeepGEMM's 384 x 255 "needs" 97920 registers while
    the kernel actually allocates 24/240/240 -> 64512 and fits.
    """
    groups = [g for g in (get(spec, "warp_groups") or []) if as_int(g.get("regs")) is not None]
    if not groups:
        return None
    return sum(as_int(g["regs"]) * (as_int(g.get("threads")) or 0) for g in groups)


def check_register_budget(spec, opts):
    thr = geti(spec, "grid.launch.threads")
    regs = geti(spec, "grid.launch.max_regs_per_thread")
    per_sm = geti(spec, "grid.launch.cta_per_sm")
    if per_sm is None:
        return Finding("register_budget", "SKIP", "grid.launch.cta_per_sm still TODO")
    if regs is not None and regs > MAX_REGS_PER_THREAD:
        return Finding("register_budget", "FAIL", "max_regs_per_thread %d > 255" % regs)
    allocated = regs_allocated_per_cta(spec)
    if allocated is not None:
        return Finding("register_budget", "PASS",
                       "%d regs/CTA from setmaxnreg, not %s x %s -- see the setmaxnreg check, "
                       "which is the binding one here" % (allocated, thr, regs))
    if None in (thr, regs):
        return Finding("register_budget", "SKIP", "grid.launch fields still TODO")
    used = thr * regs * per_sm
    detail = "%d threads x %d regs x %d CTA/SM = %d / %d" % (thr, regs, per_sm, used, REGS_PER_SM)
    if used > REGS_PER_SM:
        return Finding("register_budget", "FAIL",
                       detail + " -- cta_per_sm cannot be %d at this register count" % per_sm)
    return Finding("register_budget", "TIGHT" if used > REGS_PER_SM * 0.94 else "PASS", detail)


def check_setmaxnreg(spec, opts):
    """Post-setmaxnreg allocation, which is a per-CTA budget the launch bound hides."""
    groups = [g for g in (get(spec, "warp_groups") or []) if as_int(g.get("regs")) is not None]
    if not groups:
        return Finding("setmaxnreg", "SKIP", "no warp group states a post-setmaxnreg `regs`")
    per_sm = geti(spec, "grid.launch.cta_per_sm") or 1
    total = sum(as_int(g["regs"]) * (as_int(g.get("threads")) or 0) for g in groups)
    budget = REGS_PER_SM // per_sm
    parts = ", ".join("%s %dx%d" % (g.get("id"), as_int(g.get("threads")) or 0, as_int(g["regs"]))
                      for g in groups)
    detail = "%d regs allocated (%s) / %d available at %d CTA/SM" % (total, parts, budget, per_sm)
    if total > budget:
        return Finding("setmaxnreg", "FAIL", detail)
    return Finding("setmaxnreg", "TIGHT" if total > budget * 0.97 else "PASS", detail)


def check_mma(spec, opts):
    """mma_k, mma_m and mma_n_legal, which share the same fields."""
    entries = get(spec, "math") or []
    step = geti(spec, "mainloop.step")
    if not entries or step is None:
        return Finding("mma", "SKIP", "math[] or mainloop.step still TODO")
    problems, notes = [], []
    for i, e in enumerate(entries):
        ik = as_int(get(e, "inst_shape.K"))
        im = as_int(get(e, "inst_shape.M"))
        inn = as_int(get(e, "inst_shape.N"))
        cnt = as_int(e.get("count_per_stage"))
        unit = str(e.get("unit", ""))
        want, want_name = contraction_extent(spec, e)
        if want is None:
            want, want_name = step, "mainloop.step"
        if ik and cnt and ik * cnt != want:
            if e.get("contracts"):
                problems.append("math[%d]: %d iters x K=%d = %d != %s %d"
                                % (i, cnt, ik, cnt * ik, want_name, want))
            else:
                # An MMA whose contraction is not the mainloop axis -- attention's
                # QK^T reduces the head dim while the mainloop walks kv_seqlen --
                # is not a defect, it is a missing field.
                notes.append("math[%d]: %d x K=%d = %d != mainloop.step %d; set "
                             "math[%d].contracts to the axis this MMA reduces"
                             % (i, cnt, ik, cnt * ik, step, i))
        elif ik and cnt is None:
            notes.append("math[%d]: count_per_stage derivable = %d" % (i, step // ik))
        if "wgmma" in unit:
            if im is not None and im != 64:
                problems.append("math[%d]: wgmma M is 64, spec says %d" % (i, im))
            if inn is not None and (inn % 8 or not 8 <= inn <= 256):
                problems.append("math[%d]: wgmma N=%d is not a legal atom (8..256 step 8)"
                                % (i, inn))
            if ik is not None and ik not in (16, 32):
                problems.append("math[%d]: wgmma K is 16 (16-bit) or 32 (8-bit), spec says %d"
                                % (i, ik))
    if problems:
        return Finding("mma", "FAIL", "; ".join(problems))
    if notes:
        return Finding("mma", "MANUAL", "; ".join(notes))
    return Finding("mma", "PASS", "%d math entr(ies) legal" % len(entries))


def check_mma_m(spec, opts):
    m = geti(spec, "grid.cta_tile.M")
    entries = get(spec, "math") or []
    if m is None or not entries:
        return Finding("mma_m", "SKIP", "cta_tile.M or math[] still TODO")
    inst_m = as_int(get(entries[0], "inst_shape.M"))
    if inst_m is None:
        return Finding("mma_m", "SKIP", "math[0].inst_shape.M still TODO")
    n_groups = len(math_groups(spec)) or 1
    covered = inst_m * n_groups
    detail = "inst M %d x %d math group(s) = %d vs cta_tile.M %d" % (inst_m, n_groups, covered, m)
    if covered == m:
        return Finding("mma_m", "PASS", detail)
    if m % inst_m == 0:
        return Finding("mma_m", "MANUAL",
                       detail + " -- covers only with a split rule; state it in warp_groups.role")
    return Finding("mma_m", "FAIL", detail)


def check_trip_count(spec, opts):
    step = geti(spec, "mainloop.step")
    trip = geti(spec, "mainloop.trip_count")
    tail = get(spec, "mainloop.tail")
    axis = get(spec, "mainloop.axis")
    extent = geti(spec, "problem.dims.%s" % axis) if axis else None
    if step is None:
        return Finding("trip_count", "SKIP", "mainloop.step still TODO")
    if tail is None:
        return Finding("trip_count", "FAIL",
                       "mainloop.tail is unset -- the most commonly skipped field, "
                       "and a correctness bug when skipped")
    if extent is None:
        return Finding("trip_count", "MANUAL",
                       "reduction extent %r is dynamic or unstated; tail policy is %r"
                       % (axis, tail))
    want = -(-extent // step)
    detail = "%s=%d / step %d -> %d iters, tail %r" % (axis, extent, step, want, tail)
    if trip is not None and trip != want:
        return Finding("trip_count", "FAIL", detail + "; spec says trip_count %d" % trip)
    if extent % step and "none" in str(tail).lower():
        return Finding("trip_count", "FAIL",
                       detail + " -- extent %% step = %d but tail says none-needed"
                       % (extent % step))
    return Finding("trip_count", "PASS", detail)


def check_occupancy(spec, opts):
    """What smem and registers actually permit, against what the spec claims."""
    claimed = geti(spec, "grid.launch.cta_per_sm")
    smem = geti(spec, "grid.launch.smem_B")
    thr = geti(spec, "grid.launch.threads")
    regs = geti(spec, "grid.launch.max_regs_per_thread")
    if claimed is None:
        return Finding("occupancy", "SKIP", "grid.launch.cta_per_sm still TODO")
    limits = []
    if smem:
        cap = SMEM_PER_SM_B.get(arch_of(spec))
        if cap:
            limits.append(("smem", cap // smem))
    allocated = regs_allocated_per_cta(spec)
    if allocated:
        limits.append(("registers (setmaxnreg)", REGS_PER_SM // allocated))
    elif thr and regs:
        limits.append(("registers", REGS_PER_SM // (thr * regs)))
    if not limits:
        return Finding("occupancy", "SKIP", "no smem or register figures to derive a limit from")
    allowed = min(v for _, v in limits)
    detail = "claims %d CTA/SM; limits: %s" % (
        claimed, ", ".join("%s -> %d" % (k, v) for k, v in limits))
    if claimed > allowed:
        return Finding("occupancy", "FAIL", detail)
    if claimed == 1 and allowed > 1:
        return Finding("occupancy", "MANUAL",
                       detail + " -- %d would fit; a value of 1 is argued, not inherited" % allowed)
    return Finding("occupancy", "PASS", detail)


def check_persistence(spec, opts):
    if str(get(spec, "grid.mode", "")).strip() != "persistent":
        return Finding("persistence", "SKIP", "grid.mode is not persistent")
    per_sm = geti(spec, "grid.persistence.cta_per_sm") or geti(spec, "grid.launch.cta_per_sm")
    ctas = geti(spec, "grid.ctas")
    if opts.sms is None:
        return Finding("persistence", "MANUAL", "pass --sms to check grid >= SM_count x cta_per_sm")
    if per_sm is None or ctas is None:
        return Finding("persistence", "SKIP", "grid.ctas or cta_per_sm still TODO")
    need = opts.sms * per_sm
    detail = "grid %d vs %d SMs x %d CTA/SM = %d" % (ctas, opts.sms, per_sm, need)
    if ctas < need:
        # The canonical row (spec-schema §7) is "grid >= SM_count x cta_per_sm
        # OR the shortfall is named" -- grid_realises_it is where it gets named,
        # and a named shortfall is a claim for the reviewer, not arithmetic.
        named = str(get(spec, "grid.persistence.grid_realises_it") or "").strip()
        if len(named) > 10 and "todo" not in named.lower():
            return Finding("persistence", "MANUAL",
                           detail + " -- shortfall named, reviewer reads it: %r"
                           % named[:90])
        return Finding("persistence", "FAIL",
                       detail + " -- a grid this size is spread one CTA per SM; "
                                "the extra capacity is never used")
    return Finding("persistence", "PASS", detail)


def check_cooperative_cluster(spec, opts):
    coop = get(spec, "grid.cooperative")
    shape = get(spec, "grid.cluster.shape")
    size = 1
    for s in (shape or []):
        size *= as_int(s) or 1
    if coop in (None, False, "false"):
        return Finding("cooperative", "PASS", "not cooperative; no residency guarantee needed")
    if size > 1:
        return Finding("cooperative", "MANUAL",
                       "cooperative launch with cluster %s: the ceiling is "
                       "cudaOccupancyMaxActiveClusters x %d, not SM_count x cta_per_sm. "
                       "Measure it -- deep pipelines hit this routinely" % (shape, size))
    return Finding("cooperative", "MANUAL",
                   "cooperative launch: check the grid against what is co-resident, "
                   "and say which CTAs block on each other")


def check_arithmetic_intensity(spec, opts):
    m, n = geti(spec, "grid.cta_tile.M"), geti(spec, "grid.cta_tile.N")
    step = geti(spec, "mainloop.step")
    ops = get(spec, "mainloop.operands_per_iter") or []
    if None in (m, n, step) or not ops:
        return Finding("arithmetic_intensity", "SKIP", "cta_tile, step or operands still TODO")
    per_iter = sum(as_int(o.get("bytes")) or 0 for o in ops)
    if not per_iter:
        return Finding("arithmetic_intensity", "SKIP", "operands_per_iter states no bytes")
    ai = (2.0 * m * n * step) / per_iter
    ridge = opts.ridge or as_int(get(spec, "toolchain.measured.ridge_point"))
    src = "measured" if ridge else "[I] published"
    ridge = ridge or ridge_for(arch_of(spec), str(get(spec, "problem.dtypes.a", "bf16")))
    if ridge is None:
        return Finding("arithmetic_intensity", "MANUAL",
                       "%.0f FLOP/byte; no ridge point for %s/%s to compare against"
                       % (ai, arch_of(spec), get(spec, "problem.dtypes.a")))
    detail = "%.0f FLOP/byte per CTA tile vs ridge %.0f (%s)" % (ai, ridge, src)
    if ai >= ridge:
        return Finding("arithmetic_intensity", "PASS",
                       detail + " -- compute-bound from DRAM alone, with no reuse assumed")
    # A single tile's operands are read by every CTA sharing its row or column,
    # so this figure is the NO-REUSE lower bound and the shortfall is exactly the
    # L2 hit rate the rasterization has to deliver. That makes it an argument to
    # check against grid.rasterization, not a verdict on the tile.
    return Finding("arithmetic_intensity", "MANUAL",
                   detail + " -- needs >= %.2gx reuse of A/B out of L2 to reach peak. "
                            "grid.rasterization must argue it delivers that; if it cannot, "
                            "the tile is memory-bound by construction" % (ridge / ai))


def _carried_names(spec):
    """mainloop.loop_carried as (set_of_names, exact).

    Preferred form is a LIST, which can be compared exactly. A free-text string
    is accepted for older specs, but then only crude tokens are available and
    the caller must degrade to MANUAL rather than claim a match -- a substring
    test over prose passes for the wrong reason ("C" matches "C1", "task"
    matches any sentence containing it).
    """
    raw = get(spec, "mainloop.loop_carried")
    if isinstance(raw, (list, tuple)):
        return set(str(x).strip() for x in raw if str(x).strip()), True
    if raw is None:
        return set(), True
    return set(t for t in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", str(raw))), False


def check_loop_bounds(spec, body, opts):
    """ADVISORY: prose `range()` calls that omit start or step.

    Deliberately not a gate. The tail question this used to police is already
    decided arithmetically by check_trip_count, from mainloop.{axis,step,tail}
    and problem.dims -- declared fields, not prose shape. A regex over markdown
    stacked on top of that trains the author to satisfy the regex instead of
    thinking about the tail, and it fires on loops where the question does not
    arise at all (a parallel tile loop at L1 has no step semantics to spell
    out). Where a check needs to find something mechanically, the schema should
    carry it as a field; parsing prose is the fallback that looks rigorous and
    is not.
    """
    if not re.search(r"\bL1\b|### L1", body):
        return Finding("loop_bounds", "SKIP", "no L1-L4 nest found in the prose body")
    bare = []
    for m in re.finditer(r"range\(([^)]*)\)", body):
        args = [a for a in m.group(1).split(",") if a.strip()]
        if len(args) < 3:
            bare.append(m.group(0))
    if bare:
        return Finding("loop_bounds", "MANUAL",
                       "%d range() omit start/step: %s -- fine on a parallel tile "
                       "loop, worth a look on the contraction (trip_count owns "
                       "the actual tail check)"
                       % (len(bare), ", ".join(sorted(set(bare))[:4])))
    return Finding("loop_bounds", "PASS", "every range() states start, stop and step")


def check_loop_carried(spec, opts):
    """Every non_mma loop_carried name also appears in mainloop.loop_carried."""
    entries = get(spec, "non_mma") or []
    if not entries:
        return Finding("loop_carried", "SKIP", "non_mma is empty or still TODO")
    declared, exact = _carried_names(spec)
    if not declared:
        return Finding("loop_carried", "SKIP", "mainloop.loop_carried still TODO")
    claimed = []
    for e in entries:
        for name in (e.get("loop_carried") or []):
            if isinstance(name, str) and name.strip():
                claimed.append((e.get("id"), name.strip()))
    if not claimed:
        return Finding("loop_carried", "PASS",
                       "%d non_mma entr(ies), none declares a carried name -- "
                       "nothing to cross-check" % len(entries))
    if not exact:
        return Finding("loop_carried", "MANUAL",
                       "mainloop.loop_carried is prose, so %d claimed name(s) (%s) "
                       "cannot be matched exactly -- make it a LIST and this "
                       "becomes arithmetic"
                       % (len(claimed), ", ".join(n for _, n in claimed[:4])))
    missing = ["%s carries %r" % (i, n) for i, n in claimed if n not in declared]
    if missing:
        return Finding("loop_carried", "FAIL",
                       "%s -- absent from mainloop.loop_carried %s"
                       % ("; ".join(missing), sorted(declared)))
    return Finding("loop_carried", "PASS",
                   "%d carried name(s) trace to mainloop.loop_carried" % len(claimed))


def check_non_mma_present(spec, opts):
    entries = get(spec, "non_mma")
    if entries is None:
        return Finding("non_mma", "FAIL",
                       "non_mma is absent. `[]` is a legal answer for a plain GEMM "
                       "and an illegal omission everywhere else")
    def costed(e):
        if e.get("cost") and "TODO" not in str(e.get("cost")):
            return True
        # Work that runs in another launch has no slot in this kernel's L3
        # timeline, so it has no cost to state here -- it needs its own spec.
        return "separate" in str(e.get("where", "")).lower()

    missing = [e.get("id") for e in entries if not costed(e)]
    if not missing:
        return Finding("non_mma", "PASS", "%d entr(ies), all costed or deferred to their own spec"
                       % len(entries))
    if str(get(spec, "status", "draft")).strip() == "draft":
        # An unfilled field in a draft is not yet a defect. --gate promotes every
        # SKIP to a failure, which is what `status: review` has to clear.
        return Finding("non_mma", "SKIP", "no `cost` yet on %s" % missing)
    if str(get(spec, "status", "")).strip() == "reference":
        return Finding("non_mma", "MANUAL",
                       "no `cost` on %s; legal in a reverse-engineered spec only if "
                       "open_questions says what would settle it" % missing)
    return Finding("non_mma", "FAIL",
                   "no `cost` on %s -- without it L3's CUDA-core column is decorative" % missing)


def check_l4_accesses(spec, opts):
    """Run the spec's L4 access file, so the table below it is computed not claimed.

    This is the join between the two tools: `budget.py` checks the arithmetic the
    YAML states, `tv_check.py` checks the lowering the layouts imply, and a spec
    is only self-consistent when both hold.
    """
    raw = get(spec, "l4_accesses")
    if raw is None:
        return Finding("l4_accesses", "SKIP",
                       "no l4_accesses field -- the L4 table is unverified prose. "
                       "See references/l4-access.md")
    if str(raw).strip().lower() == "none":
        return Finding("l4_accesses", "MANUAL",
                       "declared none: nothing in this kernel is touched per-thread. "
                       "True for a pure TMA+wgmma mainloop, and worth a reviewer's eye")
    path = str(raw).strip()
    if not os.path.isabs(path):
        path = os.path.join(opts.spec_dir, path)
    if not os.path.exists(path):
        return Finding("l4_accesses", "FAIL", "access file not found: %s" % path)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import tv_check
    try:
        ok, _lines, rows, bad = tv_check.evaluate_file(path)
    except Exception as e:                       # a malformed access file is a spec defect
        return Finding("l4_accesses", "FAIL", "%s: %s" % (path, e))
    detail = "%d access(es) in %s" % (len(rows), os.path.basename(path))
    if not ok:
        return Finding("l4_accesses", "FAIL", detail + "; failing: %s" % ", ".join(bad))
    return Finding("l4_accesses", "PASS", detail + ", all computed values hold")


def check_status(spec, body, opts):
    status = str(get(spec, "status", "draft")).strip()
    oq = get(spec, "open_questions") or []
    todos = len(re.findall(r"\bTODO\b", body))
    detail = "status %r, %d open question(s), %d TODO in the body" % (status, len(oq), todos)
    if status == "approved" and not get(spec, "approved_by"):
        return Finding("status", "FAIL", detail + "; approved with no approved_by")
    if status == "review" and oq and get(spec, "source") is None:
        return Finding("status", "FAIL",
                       detail + "; open_questions must be empty before status: review")
    if status == "review" and todos:
        return Finding("status", "FAIL", detail + "; TODO left in a spec marked review")
    return Finding("status", "PASS", detail)


MANUAL_CHECKS = [
    ("concurrency", "L3's bubble check: for a steady-state stage each engine is busy "
                    "or its idle time is named. Structural, so a reviewer can settle it."),
    ("traceability", "every bound at L4 traces to an L2 loop, every name at L2 to an L1 dim."),
    ("tile_order", "grid.rasterization carries an L2 argument, not just a name."),
    ("rounding_contract", "every non_mma.dtype says where rounding lands; the parity "
                          "reference mirrors it."),
    ("acceptance", "the spec names the ONE measurement that decides acceptance, and it "
                   "is the one the kernel ships under."),
]


def run_checks(spec, body, opts):
    findings = [
        check_smem(spec, opts),
        check_staged_bytes(spec, opts),
        check_threads(spec, opts),
        check_acc_registers(spec, opts),
        check_register_budget(spec, opts),
        check_setmaxnreg(spec, opts),
        check_mma(spec, opts),
        check_mma_m(spec, opts),
        check_trip_count(spec, opts),
        check_occupancy(spec, opts),
        check_persistence(spec, opts),
        check_cooperative_cluster(spec, opts),
        check_arithmetic_intensity(spec, opts),
        check_non_mma_present(spec, opts),
        check_loop_carried(spec, opts),
        check_loop_bounds(spec, body, opts),
        check_l4_accesses(spec, opts),
        check_status(spec, body, opts),
    ]
    findings.extend(Finding(n, "MANUAL", d) for n, d in MANUAL_CHECKS)
    return findings


# -------------------------------------------------------------------- output

def report(findings, *, quiet=False, width_limit=150):
    width = max(len(f.name) for f in findings)
    order = {"FAIL": 0, "TIGHT": 1, "SKIP": 2, "MANUAL": 3, "PASS": 4}
    for f in sorted(findings, key=lambda f: (order[f.verdict], f.name)):
        if quiet and f.verdict not in ("FAIL", "TIGHT"):
            continue
        # Prose fields (a `tail` policy, a `rasterization` argument) run to
        # paragraphs. One line per check is what makes the block scannable, and
        # the spec is where the full text is read.
        detail = " ".join(str(f.detail).split())
        if len(detail) > width_limit:
            detail = detail[:width_limit - 4] + " ..."
        print("%-6s %-*s  %s" % (f.verdict, width, f.name, detail))
    tally = collections.Counter(f.verdict for f in findings)
    print("")
    print("  ".join("%s %d" % (k, tally[k]) for k in ("FAIL", "TIGHT", "SKIP", "MANUAL", "PASS")
                    if tally[k]))
    return tally


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("spec", help="path to a spec written from assets/spec-template.md")
    p.add_argument("--sms", type=int, default=None, help="SM count, for persistence/occupancy")
    p.add_argument("--ridge", type=float, default=None,
                   help="measured ridge point in FLOP/byte; overrides the published [I] value")
    p.add_argument("--quiet", action="store_true", help="only FAIL and TIGHT")
    p.add_argument("--gate", action="store_true",
                   help="also fail on SKIP -- what `status: review` requires")
    args = p.parse_args(argv)
    args.spec_dir = os.path.dirname(os.path.abspath(args.spec))

    try:
        spec, body = load_spec(args.spec)
    except DuplicateKeyError as e:
        print("FAIL   yaml           %s" % e)
        return 1

    tally = report(run_checks(spec, body, args), quiet=args.quiet)
    if tally["FAIL"] or (args.gate and tally["SKIP"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
