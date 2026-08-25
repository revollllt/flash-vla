"""TMA delivery-rate probe -- what does the copy engine actually sustain?

The FFN task-loop confounds six things at once (CTA count, ring depth,
bytes/TMA, box geometry, wgmma-retirement coupling, counter polling), so no
number taken from inside it is a machine constant. This probe strips
everything but the copy engine and sweeps the axes separately.

Questions, in the order they change the design:

Q1  Is TMA limited by in-flight BYTES or by in-flight TRANSACTIONS?
    Sweep A holds ``depth * frame_b`` fixed while varying the split. If GB/s
    depends only on the product, it is byte-limited and the levers are ring
    depth and CTA count. If it tracks ``frame_b`` at a fixed product, it is
    transaction-limited and the only lever is a bigger TMA -- which would make
    "deepen the ring" and "add CTAs" the wrong fixes.

Q2  What does box geometry cost at equal bytes and equal transaction count?
    Sweep B compares a contiguous box (the pre-blocked weight pattern) against
    128 B strips at 2 KB and 8 KB stride (the x_pad and hidden patterns).

Q3  Is one producer warp's serial issue loop the limit? Sweep C varies warps.

Q4  Can ``boxDim[0] * elemSize`` exceed the swizzle width, and how many box
    ROWS does the driver accept? ``describe()`` enumerates
    ``cuTensorMapEncodeTiled`` return codes instead of assuming. Together the
    two tables fix the largest frame a single TMA can carry, which is the
    ceiling on every "make the TMA bigger" lever below.

Q5  Tile is large: how FEW CTAs still saturate HBM? Sweep E holds the frame at
    each of several sizes and walks the CTA count, so the answer is a measured
    knee rather than ``BW_ceil * t_issue / frame`` extrapolated off a fit.

Q6  Every SM is busy: how SMALL can one CTA's TMA be and still saturate? Sweep
    F pins the grid at one CTA per SM and walks the frame size. E and F are the
    same frontier approached along its two axes; measuring both is what makes
    the frontier falsifiable rather than a rearrangement of one fit.

Run (the probe lives in the skill, so it is invoked by PATH, not as a module):
    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/memory/tma_ring.py
    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/memory/tma_ring.py \
        --sweeps E,F --json profiles/hardware-unit-test/tma_frontier.json

Off this repo: any host with torch + nvcc + CUTLASS headers can run it directly
(``python3 tma_ring.py``); it falls back to a CUDA-event timer when
``flash_vla.bench`` is absent and SAYS SO in the header line, because the timer
identity changes the number.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

def _find_repo() -> Path:
    """Walk up for a repo marker rather than counting ``parents[]``.

    The probe sits under `.claude/skills/hardware-unit-test/probes/`, four levels
    down, and an index that encodes that depth breaks silently the first time
    the skill is copied somewhere else -- which is the whole point of the
    probes living inside the skill.
    """
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            return d
    return here.parent


_REPO = _find_repo()
_SRC = Path(__file__).resolve().with_suffix(".cu")
_CUTLASS = Path(os.environ.get("CUTLASS_DIR", "/data/user/jzou521/codes/cuda/cutlass"))

# CUtensorMapSwizzle
SW_NONE, SW_32B, SW_64B, SW_128B = 0, 1, 2, 3
SW_NAME = {SW_NONE: "none", SW_32B: "32B", SW_64B: "64B", SW_128B: "128B"}

# The probe buffer. 256 MB is >2x this H100's 50 MB L2, so a walk that covers
# it cannot become cache-resident no matter how the sweep wraps.
BUF_MB = 256
BUF_B = BUF_MB * 1024 * 1024

# Bytes moved per measured launch. ~64 MB keeps every config in the tens of
# microseconds, where CUPTI's per-kernel timing is well above its own noise.
TARGET_BYTES = 64 * 1024 * 1024

# H100 SXM5: 132 SMs, 3.35 TB/s HBM3 peak, 227 KB max dynamic smem per CTA.
N_SM = 132
PEAK_TBS = 3.35
MAX_SMEM = 227 * 1024
# The MEASURED cold-read ceiling on this machine (83% of PEAK_TBS), from the
# launch/bandwidth unit -- see sm90/constants.yaml [BW-CEIL]. The
# frontier sweeps are read against this, never against PEAK_TBS: a row at 83%
# of peak is saturated, and calling it "83%" invites a hunt for the other 17%.
BW_CEIL_GBS = 2770.0


# --------------------------------------------------------------------- build
def _build_dir() -> Path:
    tag = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    d = _REPO / ".cache" / "cuda_ext" / f"tma_ring_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(verbose: bool = False) -> Path:
    out = _build_dir() / "libtma_ring.so"
    if out.exists():
        return out
    cuda_home = os.environ.get("CUDA_HOME", "/data/apps/cuda/13.1")
    cmd = [
        "nvcc", "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
        "-arch=sm_90a", "--expt-relaxed-constexpr",
        f"-I{_CUTLASS}/include",
        "-o", str(out), str(_SRC),
        f"-L{cuda_home}/lib64/stubs", "-lcuda",
    ]
    if verbose:
        print("[build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    return out


class Probe:
    def __init__(self, verbose: bool = False):
        self._lib = ctypes.CDLL(str(build(verbose)))
        self._lib.tma_probe_encode.restype = ctypes.c_int
        self._lib.tma_probe_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong,
            ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
        self._lib.tma_probe_launch.restype = ctypes.c_int
        self._lib.tma_probe_launch.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 10 + [
            ctypes.c_void_p, ctypes.c_void_p]
        self._lib.tma_probe_map_bytes.restype = ctypes.c_int
        self.map_bytes = self._lib.tma_probe_map_bytes()

    def encode(self, ptr: int, inner: int, outer: int, box_inner: int,
               box_outer: int, swizzle: int):
        """Return (buffer, rc). rc != 0 means the driver rejected the geometry."""
        # cuTensorMapEncodeTiled requires a 64 B aligned destination.
        raw = ctypes.create_string_buffer(self.map_bytes + 64)
        addr = ctypes.addressof(raw)
        off = (-addr) % 64
        rc = self._lib.tma_probe_encode(
            ctypes.c_void_p(addr + off), ctypes.c_void_p(ptr),
            inner, outer, box_inner, box_outer, swizzle)
        return (raw, addr + off), rc

    def launch(self, mapbuf, n_ctas, n_warps, depth, frame_b, trip,
               mask0, shift0, step0, mask1, step1, dbg):
        rc = self._lib.tma_probe_launch(
            ctypes.c_void_p(mapbuf[1]), n_ctas, n_warps, depth, frame_b, trip,
            mask0, shift0, step0, mask1, step1,
            ctypes.c_void_p(dbg.data_ptr()), None)
        if rc != 0:
            raise RuntimeError(f"tma_probe_launch rc={rc}")


# ----------------------------------------------------------------- geometry
class Geom:
    """One descriptor plus the coordinate walk that keeps every box in bounds.

    ``inner`` is the fastest-varying extent in ELEMENTS, so the packed row is
    ``inner * 2`` bytes and that is also the stride between consecutive box
    rows. ``inner == box_inner`` therefore means the box rows are ADJACENT and
    the whole box is one contiguous run; anything larger leaves 128 B strips at
    an ``inner * 2`` byte stride. That single distinction is what Q2 measures.
    """

    def __init__(self, name, inner, box_inner=64, swizzle=SW_128B):
        self.name = name
        self.inner = inner
        self.box_inner = box_inner
        self.swizzle = swizzle
        self.row_b = inner * 2
        self.strip_b = box_inner * 2

    def frame_bytes(self, box_outer):
        return self.strip_b * box_outer

    def plan(self, box_outer, footprint_b):
        """Descriptor dims + walk masks for a target footprint."""
        outer = BUF_B // self.row_b
        nc0 = self.inner // self.box_inner            # positions along dim0
        rows_wanted = max(box_outer, footprint_b // self.row_b)
        nc1 = min(_floor_pow2(outer // box_outer), _floor_pow2(
            max(1, rows_wanted // box_outer)))
        assert nc0 & (nc0 - 1) == 0, f"nc0={nc0} must be a power of two"
        assert nc1 >= 1
        return dict(
            inner=self.inner, outer=outer,
            box_inner=self.box_inner, box_outer=box_outer,
            swizzle=self.swizzle,
            mask0=nc0 - 1, shift0=nc0.bit_length() - 1, step0=self.box_inner,
            mask1=nc1 - 1, step1=box_outer,
            # The walk covers nc1*box_outer rows; the nc0 strips tile each row
            # exactly, so the touched footprint is the same expression for both
            # the contiguous and the strided descriptor.
            footprint_b=nc1 * box_outer * self.row_b,
        )


def _floor_pow2(n: int) -> int:
    return 1 << (max(1, n).bit_length() - 1)


# The three patterns that actually appear in ffn_taskloop.cu.
GEOMS = {
    # tmwup / tmwd: pre-blocked, box_inner == inner, so the box is contiguous.
    "contig": Geom("contig", inner=64),
    # tmx: x_pad is (64, D=1024) -- 128 B strips at a 2048 B stride.
    "stride2k": Geom("stride2k", inner=1024),
    # tmh: hidden is (64, FF=4096) -- 128 B strips at an 8192 B stride.
    "stride8k": Geom("stride8k", inner=4096),
}


# ------------------------------------------------------------------ Q4 probe
def describe(probe: Probe, buf: torch.Tensor) -> list[dict]:
    """Enumerate cuTensorMapEncodeTiled legality: which box widths each
    swizzle mode accepts. Settles whether a TMA box row can exceed the swizzle
    width -- i.e. whether 'make each TMA bigger' can grow the contiguous run or
    only the number of strips."""
    rows = []
    ptr = buf.data_ptr()
    for sw in (SW_NONE, SW_32B, SW_64B, SW_128B):
        for box_inner in (8, 16, 32, 64, 128, 256):
            inner = max(box_inner, 4096)
            _, rc = probe.encode(ptr, inner, 1024, box_inner, 8, sw)
            rows.append(dict(swizzle=SW_NAME[sw], box_inner_elem=box_inner,
                             box_inner_b=box_inner * 2, rc=rc, ok=rc == 0))
    return rows


def describe_rows(probe: Probe, buf: torch.Tensor) -> list[dict]:
    """Enumerate the accepted range of boxDim[1] -- the number of box ROWS.

    The width table above caps a box ROW at the swizzle width, so the only way
    to a bigger TMA is more rows. That makes `max boxDim[1] x swizzle width`
    the hard ceiling on bytes-per-TMA, and therefore the ceiling on the whole
    "bigger frame" lever. It is worth one call per candidate rather than one
    sentence of recollection.
    """
    rows = []
    ptr = buf.data_ptr()
    for box_outer in (8, 64, 128, 192, 256, 257, 512):
        _, rc = probe.encode(ptr, 4096, 1024, 64, box_outer, SW_128B)
        rows.append(dict(box_outer=box_outer, frame_b=128 * box_outer,
                         rc=rc, ok=rc == 0))
    return rows


# ------------------------------------------------------------------- timing
def _bench_samples(run, reps):
    """Return per-iteration GPU times in ms, and the name of the timer used.

    Prefers this repo's CUPTI harness -- the one every other number in the
    repo is taken under -- and falls back to CUDA events elsewhere. The two
    disagree by the launch overhead events include (~13% at 0.18 ms, most of
    the measurement at 5 us), so the timer is reported, never assumed.
    """
    try:
        from flash_vla.bench import bench_gpu_time
    except ImportError:
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        beg = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            beg[i].record()
            run()
            end[i].record()
        torch.cuda.synchronize()
        return [b.elapsed_time(e) for b, e in zip(beg, end)], "events"
    return bench_gpu_time(run, enable_cupti=True, cold_l2_cache=False,
                          dry_run_iters=3, repeat_iters=reps), "cupti"


TIMER_USED = ""


def _time_us(probe, mapbuf, cfg, dbg, reps=30) -> float:
    global TIMER_USED

    def run():
        probe.launch(mapbuf, cfg["n_ctas"], cfg["n_warps"], cfg["depth"],
                     cfg["frame_b"], cfg["trip"], cfg["mask0"], cfg["shift0"],
                     cfg["step0"], cfg["mask1"], cfg["step1"], dbg)

    # Residency is controlled by the 256 MB footprint, not by a flush kernel:
    # an extra memset between iterations would land inside CUPTI's window.
    samples, TIMER_USED = _bench_samples(run, reps)
    samples = sorted(samples)
    return samples[len(samples) // 2] * 1000.0


def measure(probe, geom: Geom, *, n_ctas, n_warps, depth, box_outer,
            footprint_b, dbg) -> dict | None:
    frame_b = geom.frame_bytes(box_outer)
    smem = n_warps * depth * frame_b + n_warps * depth * 8
    if smem > MAX_SMEM:
        return dict(skipped=f"smem {smem} > {MAX_SMEM}")
    plan = geom.plan(box_outer, footprint_b)
    # trip must be well past the fill so the measurement is steady state, not
    # ring fill. `4 * depth` leaves the fill at 25% of the trips, which biases
    # exactly the configurations the frontier sweeps care about (132 CTAs x
    # 32 KB moves TARGET_BYTES in 16 issues); 8x holds the fill under 13%.
    per_issue = n_ctas * n_warps * frame_b
    trip = max(8 * depth, TARGET_BYTES // per_issue)
    total_b = per_issue * trip
    return dict(plan=plan, frame_b=frame_b, trip=trip, total_b=total_b,
                smem=smem, n_ctas=n_ctas, n_warps=n_warps, depth=depth,
                box_outer=box_outer)


def run_point(probe, buf, geom, *, n_ctas, n_warps, depth, box_outer,
              footprint_b, dbg) -> dict:
    pre = measure(probe, geom, n_ctas=n_ctas, n_warps=n_warps, depth=depth,
                  box_outer=box_outer, footprint_b=footprint_b, dbg=dbg)
    if "skipped" in pre:
        return dict(geom=geom.name, n_ctas=n_ctas, n_warps=n_warps,
                    depth=depth, frame_b=geom.frame_bytes(box_outer),
                    skipped=pre["skipped"])
    p = pre["plan"]
    mapbuf, rc = probe.encode(buf.data_ptr(), p["inner"], p["outer"],
                              p["box_inner"], p["box_outer"], p["swizzle"])
    if rc != 0:
        return dict(geom=geom.name, n_ctas=n_ctas, n_warps=n_warps,
                    depth=depth, frame_b=pre["frame_b"], skipped=f"encode rc={rc}")
    cfg = dict(n_ctas=n_ctas, n_warps=n_warps, depth=depth,
               frame_b=pre["frame_b"], trip=pre["trip"], mask0=p["mask0"],
               shift0=p["shift0"], step0=p["step0"], mask1=p["mask1"],
               step1=p["step1"])
    us = _time_us(probe, mapbuf, cfg, dbg)
    torch.cuda.synchronize()
    if int(dbg.max().item()) != 0:
        raise RuntimeError(f"watchdog fired: dbg={dbg[dbg != 0][:8].tolist()}")
    gbs = pre["total_b"] / (us * 1e-6) / 1e9
    inflight_b = n_ctas * n_warps * depth * pre["frame_b"]
    # The constant this probe exists to produce. Every warp issues exactly
    # `trip` transactions and they all run for the same `us`, so the per-warp
    # issue interval is a division -- it does not go through the byte
    # accounting, and so survives a wrong footprint or a miscounted frame.
    ns_per_txn = us * 1000.0 / pre["trip"]
    return dict(geom=geom.name, n_ctas=n_ctas, n_warps=n_warps, depth=depth,
                frame_b=pre["frame_b"], trip=pre["trip"],
                total_mb=pre["total_b"] / 1e6, us=us, gbs=gbs,
                ns_per_txn=ns_per_txn,
                pct_peak=100.0 * gbs / (PEAK_TBS * 1000),
                inflight_kb=inflight_b / 1024,
                footprint_mb=p["footprint_b"] / 1e6, smem_kb=pre["smem"] / 1024)


# ------------------------------------------------------------------- sweeps
def sweep_A(probe, buf, dbg, footprint_b):
    """Q1: bytes vs transactions. depth x frame_b at fixed CTA/warp count.

    The decisive pairs are equal-in-flight-bytes, different transaction count:
    (depth=16, 2 KB) vs (depth=2, 16 KB) both hold 32 KB per warp.
    """
    out = []
    for depth in (2, 4, 8, 16):
        for box_outer in (16, 32, 64, 128, 256):     # 2/4/8/16/32 KB frames
            out.append(run_point(probe, buf, GEOMS["stride8k"], n_ctas=32,
                                 n_warps=1, depth=depth, box_outer=box_outer,
                                 footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_B(probe, buf, dbg, footprint_b):
    """Q2: box geometry at equal bytes and equal transaction count."""
    out = []
    for name in ("contig", "stride2k", "stride8k"):
        for n_ctas in (32, 132):
            out.append(run_point(probe, buf, GEOMS[name], n_ctas=n_ctas,
                                 n_warps=1, depth=4, box_outer=64,
                                 footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_C(probe, buf, dbg, footprint_b):
    """Q3: does one producer warp's serial issue loop cap the rate?"""
    out = []
    for n_warps in (1, 2, 4):
        for n_ctas in (32, 132):
            out.append(run_point(probe, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                                 n_warps=n_warps, depth=4, box_outer=64,
                                 footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_D(probe, buf, dbg, footprint_b):
    """CTA scaling at the task-loop's own ring shape (depth 4, 8 KB frames)."""
    out = []
    for n_ctas in (16, 32, 64, 128, 132, 264):
        out.append(run_point(probe, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                             n_warps=1, depth=4, box_outer=64,
                             footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_E(probe, buf, dbg, footprint_b):
    """Q5: how FEW CTAs saturate? CTA scaling at 8 / 16 / 32 KB frames.

    Read it as the frontier along the CTA axis: for each frame size, the
    smallest CTA count whose row reaches ~95% of the 2.77 TB/s ceiling. A
    bigger frame should move that count down proportionally -- if it does not,
    the per-warp issue interval is not frame-independent and every floor
    derived from a single `ns/txn` is wrong.
    """
    out = []
    for box_outer in (64, 128, 256):              # 8 / 16 / 32 KB
        for n_ctas in (8, 16, 24, 32, 48, 64, 96, 128, 132, 264):
            out.append(run_point(probe, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                                 n_warps=1, depth=4, box_outer=box_outer,
                                 footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_F(probe, buf, dbg, footprint_b):
    """Q6: how SMALL can one CTA's TMA be at full occupancy? Frame scaling.

    Three producer configurations at 1 KB..32 KB frames: one CTA per SM, two
    CTAs per SM, and one CTA per SM with two producer warps. If the frontier is
    `n_cta x n_warp x frame`, the (264,1) and (132,2) rows must saturate at the
    same frame size as each other and at half the (132,1) one. That equality is
    the falsifiable form of "warps and CTAs enter the copy budget identically".
    """
    out = []
    for n_ctas, n_warps in ((132, 1), (264, 1), (132, 2)):
        for box_outer in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
            out.append(run_point(probe, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                                 n_warps=n_warps, depth=4, box_outer=box_outer,
                                 footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_G(probe, buf, dbg, footprint_b):
    """Q7: the CTA scan again with TWO producer warps, and the smem wall.

    Sweep E answers "how few CTAs" for one producer warp per CTA. A second warp
    should halve that count -- the product law says CTAs and warps are the same
    currency, and sweep F showed (264,1) and (132,2) agreeing at one CTA count.
    This walks the CTA axis so the equality is tested across the whole range,
    G at N CTAs matching E at 2N.

    It also probes where the trade STOPS being free. Shared memory bounds
    `n_warps x depth x frame` per CTA, so a second warp cannot keep the 32 KB
    frame: 2 x 4 x 32 KB = 256 KB exceeds the 227 KB cap and the largest frame
    two warps can hold at depth 4 is 28 KB. The 4-warp rows at 7 and 14 KB test
    the consequence -- if the per-CTA in-flight budget is really smem/depth
    however it is split, 4x14 KB and 2x28 KB must deliver the same.
    """
    out = []
    plan = [(2, (64, 128, 192, 224)),        # 8 / 16 / 24 / 28 KB
            (4, (56, 112))]                  # 7 / 14 KB
    for n_warps, boxes in plan:
        for box_outer in boxes:
            for n_ctas in (8, 16, 24, 32, 48, 64, 96, 128, 132):
                out.append(run_point(probe, buf, GEOMS["stride8k"],
                                     n_ctas=n_ctas, n_warps=n_warps, depth=4,
                                     box_outer=box_outer,
                                     footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_H(probe, buf, dbg, footprint_b):
    """Q8: where does a CTA stop absorbing more? The per-CTA delivery knee.

    Sweep G found that per-CTA delivery is linear in `n_warps x frame` up to
    32 KB and pinned at ~133 GB/s by 48 KB, with three different splits landing
    on the same number. That brackets a knee but does not locate it, and the
    knee IS the answer to "what does a second producer warp buy" -- above it,
    nothing.

    So: walk the per-CTA product from 32 to 48 KB in 4 KB steps, by two
    different splits (2 warps with a growing frame, 4 warps with a smaller one)
    at two CTA counts. If the knee is a per-CTA property it lands at the same
    product for both splits and both grids; if it tracks the frame or the warp
    count instead, the two curves separate.
    """
    out = []
    for n_ctas in (8, 16):
        for n_warps, boxes in ((1, (256,)),                     # 32 KB ref
                               (2, (128, 144, 160, 176, 192)),  # 32..48 KB
                               (4, (64, 72, 80, 88, 96))):      # 32..48 KB
            for box_outer in boxes:
                out.append(run_point(probe, buf, GEOMS["stride8k"],
                                     n_ctas=n_ctas, n_warps=n_warps, depth=4,
                                     box_outer=box_outer,
                                     footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_I(probe, buf, dbg, footprint_b):
    """Q9: the transition region, where neither limit binds cleanly.

    Between roughly 16 and 96 CTAs, with a per-CTA product above the 36 KB
    knee, the two-term model min(n_ctas x per_cta, curve(product)) measured up
    to 15% optimistic, and the CTA ladder is not even monotone there (48 CTAs
    came in BELOW 32 at 2 warps x 24 KB). Coarse steps cannot tell a real dip
    from a noisy point, so walk the band finely at three per-CTA products, with
    the 1-warp column as a control that stays below the knee.
    """
    out = []
    for n_warps, box_outer in ((1, 256), (2, 192), (2, 224)):
        for n_ctas in (16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 80, 96):
            out.append(run_point(probe, buf, GEOMS["stride8k"], n_ctas=n_ctas,
                                 n_warps=n_warps, depth=4, box_outer=box_outer,
                                 footprint_b=footprint_b, dbg=dbg))
    return out


def sweep_J(probe, buf, dbg, footprint_b):
    """Q10: TMA latency as a function of frame size, measured not inferred.

    TMA-DEPTH infers ~680 ns of latency at 8 KB from the depth-2 interval, and
    says the 16 and 32 KB depth-2 rows cannot supply the same number because
    they are bandwidth-contaminated. At 8 CTAs nothing aggregate binds, so the
    contamination is gone and DEPTH 1 measures latency directly: with one frame
    outstanding the warp waits for each arrival before issuing the next, so the
    interval IS the round trip.

    Reading it: depth 1 is latency, and the depth at which the interval stops
    falling is where the ring covers that latency. If that depth grows with the
    frame, `depth: 4` is a frame-dependent recommendation and TMA-DEPTH needs a
    bound rather than a number.
    """
    out = []
    for box_outer in (64, 128, 256):              # 8 / 16 / 32 KB
        for depth in (1, 2, 3, 4, 6, 8):
            out.append(run_point(probe, buf, GEOMS["stride8k"], n_ctas=8,
                                 n_warps=1, depth=depth, box_outer=box_outer,
                                 footprint_b=footprint_b, dbg=dbg))
    return out


SWEEPS = {"A": sweep_A, "B": sweep_B, "C": sweep_C, "D": sweep_D,
          "E": sweep_E, "F": sweep_F, "G": sweep_G, "H": sweep_H,
          "I": sweep_I, "J": sweep_J}


def render(rows: list[dict]) -> str:
    """One row per config. `ns/txn` is the per-warp issue interval -- the
    constant -- and `%ceil` measures against the MEASURED 2.77 TB/s ceiling,
    not the 3.35 datasheet peak, so a row at 100% means saturated rather than
    impossible."""
    hdr = (f"{'geom':>9} {'CTA':>4} {'wrp':>3} {'dep':>3} {'frame':>7} "
           f"{'inflight':>9} {'MB':>7} {'us':>8} {'GB/s':>8} {'ns/txn':>7} "
           f"{'%peak':>6} {'%ceil':>6}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        if "skipped" in r:
            lines.append(f"{r['geom']:>9} {r['n_ctas']:>4} {r['n_warps']:>3} "
                         f"{r['depth']:>3} {r['frame_b']:>7} "
                         f"{'-- ' + r['skipped']:>40}")
            continue
        lines.append(
            f"{r['geom']:>9} {r['n_ctas']:>4} {r['n_warps']:>3} {r['depth']:>3} "
            f"{r['frame_b']:>7} {r['inflight_kb']:>8.0f}K {r['total_mb']:>7.1f} "
            f"{r['us']:>8.2f} {r['gbs']:>8.1f} {r['ns_per_txn']:>7.1f} "
            f"{r['pct_peak']:>5.1f}% {100.0 * r['gbs'] / (BW_CEIL_GBS):>5.1f}%")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", default="A,B,C,D,E,F,G")
    ap.add_argument("--footprint-mb", type=int, default=BUF_MB,
                    help="target walk footprint; << 50 MB makes the source "
                         "L2-resident, >> 100 MB forces cold DRAM")
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    dev = torch.cuda.current_device()
    print(f"[env] {torch.cuda.get_device_name(dev)}  torch {torch.__version__}",
          flush=True)

    probe = Probe(a.verbose)
    buf = torch.empty(BUF_B // 2, dtype=torch.bfloat16, device="cuda")
    buf.normal_()
    dbg = torch.zeros(264 * 2, dtype=torch.int64, device="cuda")

    print("\n[Q4a] cuTensorMapEncodeTiled legality "
          "(box row bytes vs swizzle width)", flush=True)
    legal = describe(probe, buf)
    widths = sorted({r["box_inner_b"] for r in legal})
    print("  " + "swizzle".rjust(8) + "".join(f"{w:>7}B" for w in widths))
    for sw in ("none", "32B", "64B", "128B"):
        cells = []
        for w in widths:
            m = next(r for r in legal
                     if r["swizzle"] == sw and r["box_inner_b"] == w)
            cells.append("ok" if m["ok"] else f"rc{m['rc']}")
        print("  " + sw.rjust(8) + "".join(f"{c:>8}" for c in cells))

    print("\n[Q4b] accepted boxDim[1] (box ROWS) at 128 B rows, SW128 "
          "-- this is what caps bytes/TMA", flush=True)
    rowlegal = describe_rows(probe, buf)
    for r in rowlegal:
        print(f"  boxDim[1]={r['box_outer']:>4}  frame={r['frame_b']:>6} B  "
              f"{'ok' if r['ok'] else 'rc%d' % r['rc']}")

    footprint_b = a.footprint_mb * 1024 * 1024
    results = {"legality": legal, "row_legality": rowlegal,
               "footprint_mb": a.footprint_mb, "sweeps": {}}
    for key in [s.strip().upper() for s in a.sweeps.split(",") if s.strip()]:
        if key not in SWEEPS:
            raise SystemExit(f"unknown sweep {key!r}; have {sorted(SWEEPS)}")
        print(f"\n[sweep {key}] {SWEEPS[key].__doc__.splitlines()[0]}", flush=True)
        rows = SWEEPS[key](probe, buf, dbg, footprint_b)
        results["sweeps"][key] = rows
        print(render(rows), flush=True)
        results["timer"] = TIMER_USED
        if TIMER_USED != "cupti":
            print(f"  [timer] {TIMER_USED} -- includes launch overhead; not "
                  f"comparable with the recorded CUPTI constants", flush=True)

    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
        print(f"\n[json] {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
