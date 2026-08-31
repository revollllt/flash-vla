"""Build, time and repeat -- one implementation, not four.

Before this, each driver rolled its own: repetition counts of 5, 7, 20 and 30
across three median implementations, and `overlap.py` shipped with a single
launch per mode and read two identical configurations 11% apart.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .abi import Unit
from .toolchain import cache_dir, cuda_home, cutlass_include

_INCLUDE = Path(__file__).resolve().parents[1] / "include"

# Enough samples that the median is not one unlucky launch, few enough that a
# sweep stays inside a Slurm allocation. Every unit uses this number, so two
# units' numbers are comparable without checking how each was taken.
REPS = 7

TIMER_USED = ""


def build(src: Path, *, arch_flags=None, verbose=False) -> Path:
    """Compile one unit .cu to a .so, cached by source hash.

    The hash covers the .cu AND every hut/ header, so editing a shared header
    invalidates every unit -- which is the point of sharing them.
    """
    h = hashlib.sha256(Path(src).read_bytes())
    for hdr in sorted(_INCLUDE.rglob("*.cuh")) + sorted(_INCLUDE.rglob("*.hpp")):
        h.update(hdr.read_bytes())
    out = cache_dir(Path(src).stem, h.hexdigest()[:16]) / f"lib{Path(src).stem}.so"
    if out.exists():
        return out
    cmd = ["nvcc", "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
           *(arch_flags or ["-arch=sm_90a"]), "--expt-relaxed-constexpr",
           f"-I{cutlass_include()}", f"-I{_INCLUDE}",
           "-o", str(out), str(src),
           f"-L{cuda_home()}/lib64/stubs", "-lcuda"]
    if verbose:
        print("[build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    # ptxas serialising wgmma still produces a number -- the serialised one --
    # so it is a build failure here, not a warning. [protocol.md rule 11]
    for code in ("C7515", "C7518"):
        if code in r.stderr:
            raise RuntimeError(
                f"ptxas serialised the wgmma pipeline ({code}); the rate would "
                f"be the serialised one:\n{r.stderr}")
    return out


def load(src, **kw) -> Unit:
    return Unit(build(Path(src), **kw))


def time_us(run, *, reps=None, flush_l2=False):
    """Median GPU time in us and the run-to-run spread, as (median, spread).

    Prefers the repo's CUPTI harness -- the one every recorded constant was
    taken under -- and falls back to CUDA events elsewhere, SAYING WHICH: the
    two disagree by the launch overhead events include, so a number from one is
    not comparable with a constant recorded under the other. [rule 8]
    """
    global TIMER_USED
    # Late-bound, not a default argument: `reps=REPS` in the signature freezes
    # the count at import time and silently ignores --reps. The sample count
    # changes what the median means, so it is a measurement parameter -- it has
    # to be settable and reported, not baked in. [rule 8]
    reps = REPS if reps is None else reps
    import torch
    try:
        from flash_vla.bench import bench_gpu_time
        samples = bench_gpu_time(run, enable_cupti=True, cold_l2_cache=flush_l2,
                                 dry_run_iters=3, repeat_iters=reps)
        TIMER_USED = "cupti"
    except ImportError:
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        beg = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            beg[i].record(); run(); end[i].record()
        torch.cuda.synchronize()
        samples = [b.elapsed_time(e) for b, e in zip(beg, end)]
        TIMER_USED = "events"
    samples = sorted(s * 1000.0 for s in samples)
    med = samples[len(samples) // 2]
    return med, (samples[-1] - samples[0]) / max(med, 1e-9)


LIB_SITES = {1: "barrier wait", 2: "barrier drain", 3: "counter poll",
             4: "producer_acquire", 5: "consumer_wait", 6: "producer_tail"}


def check_watchdog(dbg, sites=None) -> None:
    """Raise naming the trap site if any CTA hit its deadline."""
    if int(dbg.max().item()) == 0:
        return
    names = dict(LIB_SITES, **(sites or {}))
    hit = dbg.view(-1, 2)
    site = int(hit[hit[:, 0] != 0][0, 0].item())
    raise RuntimeError(
        f"watchdog fired at site {site} = {names.get(site, 'unknown')} -- a "
        f"count or a phase is wrong, not a slow measurement")
