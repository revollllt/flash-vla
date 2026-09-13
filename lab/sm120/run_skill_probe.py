#!/usr/bin/env python3
"""Run a `hardware-unit-test` probe against sm_120 without editing the skill.

Three of the skill's six probes contain no sm90-only construct and measure
mechanisms that survive on this part -- `tma_ring`, `gmem_atomic` and
`coop_launch`. Their harness takes an `arch_flags` override
(`probes/hut/harness.py`), but each probe script calls `harness.load(_SRC)`
without one, so they compile for the harness default of `-arch=sm_90a` and fail
to run here.

The skill is a shared source of truth for every hardware axis and this is a
property of one machine, so the override belongs here rather than in it. The
probe modules bind `harness` by module object, so patching `harness.build`
before importing one reaches the call inside `harness.load`.

    CUDA_HOME=/path/to/cuda python3 lab/sm120/run_skill_probe.py coop_launch
    CUDA_HOME=/path/to/cuda python3 lab/sm120/run_skill_probe.py tma_ring -- --sweeps A

Anything after `--` is passed to the probe unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROBES = REPO / ".claude" / "skills" / "hardware-unit-test" / "probes"

#: `sm_120f` rather than `sm_120a`: identical on all 51 probed instructions and
#: covers the GB20x family rather than one die [isa.target.a_required].
DEFAULT_ARCH = ["-gencode", "arch=compute_120f,code=sm_120f"]

#: Some probes bake their machine's constants in as module globals -- tma_ring
#: carries sm90's 132 SMs, 227 KB of shared memory and 3.35 TB/s. Those are not
#: configuration, they are the wrong machine, and a stage x box grid sized
#: against 227 KB fails at launch here rather than being skipped
#: [smem.bytes.cta.max]. Overridden by name after import, and every override is
#: printed so a number is never read against the wrong denominator.
MACHINE = {
    "N_SM": 170,
    "MAX_SMEM": 101376,        # [smem.bytes.cta.max]
    "SMEM_PER_SM": 102400,
    "PEAK_TBS": 1.792,         # spec.py datasheet peak, NOT a floor denominator
    "BW_CEIL_GBS": 1598.0,     # [ld.bw.dev.dram], the measured plain-load ceiling
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", help="unit name, e.g. coop_launch, gmem_atomic, tma_ring")
    ap.add_argument("--arch", default=None,
                    help="override the arch flags (default: sm_120f)")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="arguments after `--` are passed to the probe")
    a = ap.parse_args()

    src = PROBES / "units" / a.probe / f"{a.probe}.py"
    if not src.is_file():
        have = sorted(p.name for p in (PROBES / "units").iterdir() if p.is_dir())
        sys.exit(f"no probe {a.probe!r} under {PROBES / 'units'}; have {have}")

    sys.path.insert(0, str(PROBES))
    harness = importlib.import_module("hut.harness")

    arch = a.arch.split() if a.arch else DEFAULT_ARCH
    original = harness.build

    def build(source, *, arch_flags=None, verbose=False):
        return original(source, arch_flags=arch_flags or arch, verbose=verbose)

    harness.build = build
    print(f"[run_skill_probe] {a.probe} with {' '.join(arch)}", flush=True)

    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    sys.argv = [str(src), *rest]

    # Import rather than run, so the baked-in machine constants can be replaced
    # before main() reads them.
    spec = importlib.util.spec_from_file_location(f"probe_{a.probe}", src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for name, value in MACHINE.items():
        if hasattr(module, name):
            print(f"[run_skill_probe] {name}: {getattr(module, name)} -> {value}",
                  flush=True)
            setattr(module, name, value)

    return int(module.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
