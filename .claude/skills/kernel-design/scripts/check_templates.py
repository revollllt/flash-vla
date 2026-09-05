#!/usr/bin/env python3
"""Compile every sm90 template and assert the PTX it was written to produce.

A template that only compiles proves nothing: a mainloop whose wgmma is
dead-code-eliminated still exits zero.  Each template therefore declares the
instructions it exists to demonstrate, and this script fails when one is
missing from the generated PTX.

Directives, anywhere in the template, one per line:

    // CHECK-GRADE: structural          what the file claims to be (required)
    // CHECK-ARCH: sm_90a               target (default sm_90a)
    // CHECK-INCLUDE: third_party/x     repo-relative -I, repeatable
    // CHECK-PTX: wgmma\\.mma_async      regex that must match the PTX
    // CHECK-PTX-COUNT: 4 ld\\.shared    regex that must match >= N times

Each template is compiled to PTX, ASSEMBLED with ptxas (which -ptx alone skips,
and which is what catches an instruction the target does not support), and then
checked against its declared assertions.

    Exit codes: 0 all templates pass, 1 a template failed, 2 no usable nvcc.

The GRADE is what a reader is entitled to believe.  A `structural` template
compiles and its instructions survive codegen; nothing in it has been run, so
it must not carry measurements.  A `reference` template is a whole machine: it
runs, checks itself against a reference, and reports its own numbers, so it
must carry the STATUS block those numbers live in and the build line that
reproduces them.  This script enforces the grade a file declares; it cannot
tell whether the numbers in a STATUS block are current, which is why the
STATUS block names the machine and the toolchain it was taken on.

What no grade proves: that the kernel computes the right values.  For a
`structural` template numerical authority is the parity harness named in its
header; for a `reference` template it is the in-file check, run on a GPU.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = SKILL_DIR / "references" / "templates"
# .claude/skills/kernel-design -> repo root
REPO_ROOT = SKILL_DIR.parent.parent.parent

DIRECTIVE_RE = re.compile(
    r"^\s*//\s*CHECK-(GRADE|ARCH|INCLUDE|PTX-COUNT|PTX)\s*:\s*(.+?)\s*$", re.MULTILINE
)

GRADES = ("structural", "reference")


MODULE_HINT = (
    "source /usr/share/Modules/init/bash && module load cuda/13.0 gcc/13.3 "
    "&& export NVCC_PREPEND_FLAGS=\"-ccbin $(command -v g++)\""
)


def find_nvcc(explicit=None):
    """PATH only: the toolchain is a module load, not a guess."""
    if explicit:
        return explicit if Path(explicit).exists() else None
    return shutil.which("nvcc")


def find_ccbin(explicit=None):
    """nvcc's default host compiler here is GCC 8, too old for CUDA 13 headers.

    Returns (path, warning).  An explicit flag wins, then $CXX, then g++ on
    PATH; a version below 9 is reported rather than silently accepted.
    """
    candidate = explicit or os.environ.get("CXX") or shutil.which("g++")
    if not candidate:
        return None, "no g++ found; " + MODULE_HINT
    try:
        first = subprocess.run(
            [candidate, "-dumpfullversion", "-dumpversion"],
            capture_output=True, text=True, timeout=20,
        ).stdout.split()
        major = int(first[0].split(".")[0]) if first else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return candidate, None
    if major < 9:
        return candidate, f"{candidate} is GCC {major}, too old for CUDA 13; " + MODULE_HINT
    return candidate, None


def parse_directives(text):
    grade, arch, includes, checks, counts = None, "sm_90a", [], [], []
    for kind, value in DIRECTIVE_RE.findall(text):
        if kind == "GRADE":
            grade = value
        elif kind == "ARCH":
            arch = value
        elif kind == "INCLUDE":
            includes.append(value)
        elif kind == "PTX":
            checks.append(value)
        elif kind == "PTX-COUNT":
            n, _, pattern = value.partition(" ")
            counts.append((int(n), pattern.strip()))
    return grade, arch, includes, checks, counts


def check_grade(grade, text):
    """The grade is a promise to the reader; this is the promise's terms.

    Both directions matter.  A `reference` file without a STATUS block claims
    to have been run and shows nothing for it.  A `structural` file WITH one
    carries numbers no harness in the file can reproduce, which is how a
    template starts quoting a measurement nobody can re-take.
    """
    if grade is None:
        return [f"declares no CHECK-GRADE ({' | '.join(GRADES)})"]
    if grade not in GRADES:
        return [f"CHECK-GRADE is {grade!r}, not one of {GRADES}"]

    has_status = re.search(r"^//\s*STATUS\b", text, re.M) is not None
    notes = []
    if grade == "reference":
        if "int main(" not in text:
            notes.append("graded reference but has no `int main(` to run")
        if not has_status:
            notes.append("graded reference but carries no `// STATUS` block")
        if not re.search(r"^//.*\bnvcc\b", text, re.M):
            notes.append("graded reference but documents no nvcc build line")
    else:
        if has_status:
            notes.append("graded structural but carries a `// STATUS` block; "
                         "a file that reports numbers has to be able to produce them")
        if not re.search(r"structural only", text, re.I):
            notes.append("graded structural but does not say so in its header")
    return notes


def check_pdl_annotations(text):
    """A PDL site without a declared placement is a placement nobody can audit."""
    notes = []
    if "pdl_wait(" in text and "PDL-WAIT:" not in text:
        notes.append("calls pdl_wait but declares no `// PDL-WAIT:` placement")
    if "pdl_trigger" in text and "PDL-TRIGGER:" not in text:
        notes.append("calls pdl_trigger but declares no `// PDL-TRIGGER:` placement")
    return notes


def check_one(path, nvcc, ccbin, keep_dir):
    text = path.read_text()
    grade, arch, includes, checks, counts = parse_directives(text)
    failures = check_grade(grade, text) + check_pdl_annotations(text)
    if not checks and not counts:
        failures.append("declares no CHECK-PTX assertion")

    out_dir = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp())
    ptx_path = out_dir / (path.stem + ".ptx")
    cmd = [nvcc, f"-arch={arch}", "-std=c++17", "-ptx", str(path), "-o", str(ptx_path)]
    if ccbin:
        cmd[1:1] = ["-ccbin", ccbin]
    for inc in includes:
        cmd += ["-I", str(REPO_ROOT / inc)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, ["compile failed:"] + [f"    {line}" for line in tail[:12]], cmd, grade

    # -ptx stops before ptxas, so it accepts instructions the assembler will
    # reject -- setmaxnreg on a plain sm_90 target, for one.  Assemble for real
    # before trusting the file.
    asm = subprocess.run(
        [str(Path(nvcc).parent / "ptxas"), f"-arch={arch}", str(ptx_path),
         "-o", os.devnull],
        capture_output=True, text=True,
    )
    if asm.returncode != 0:
        tail = (asm.stderr or asm.stdout).strip().splitlines()
        return False, ["ptxas failed:"] + [f"    {line}" for line in tail[:8]], cmd, grade

    ptx = ptx_path.read_text()
    for pattern in checks:
        if not re.search(pattern, ptx):
            failures.append(f"PTX missing /{pattern}/")
    for n, pattern in counts:
        hits = len(re.findall(pattern, ptx))
        if hits < n:
            failures.append(f"PTX has {hits} of >= {n} /{pattern}/")
    return not failures, failures, cmd, grade


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nvcc", help="path to nvcc (default: the one on PATH)")
    ap.add_argument("--ccbin", help="host compiler (default: $CXX, then g++ on PATH)")
    ap.add_argument("--only", help="substring filter on template file name")
    ap.add_argument("--keep", metavar="DIR", help="keep generated PTX in DIR")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the nvcc line")
    args = ap.parse_args()

    nvcc = find_nvcc(args.nvcc)
    if not nvcc:
        print("no nvcc on PATH. Run:\n  " + MODULE_HINT, file=sys.stderr)
        return 2
    ccbin, warning = find_ccbin(args.ccbin)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    if args.keep:
        Path(args.keep).mkdir(parents=True, exist_ok=True)

    templates = sorted(TEMPLATE_DIR.glob("*.cu"))
    if args.only:
        templates = [t for t in templates if args.only in t.name]
    if not templates:
        print(f"no templates under {TEMPLATE_DIR}", file=sys.stderr)
        return 1

    print(f"nvcc {nvcc}" + (f"  ccbin {ccbin}" if ccbin else ""))
    failed, graded = 0, {}
    for path in templates:
        ok, notes, cmd, grade = check_one(path, nvcc, ccbin, args.keep)
        graded[grade] = graded.get(grade, 0) + 1
        print(f"{'PASS' if ok else 'FAIL'}  {(grade or 'ungraded')[:10]:10s}  {path.name}")
        if args.verbose:
            print("      " + " ".join(cmd))
        if not ok:
            failed += 1
            for note in notes:
                print(f"      {note}")
    tally = ", ".join(f"{n} {g or 'ungraded'}" for g, n in sorted(graded.items()))
    print(f"\n{len(templates) - failed}/{len(templates)} templates pass ({tally})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
