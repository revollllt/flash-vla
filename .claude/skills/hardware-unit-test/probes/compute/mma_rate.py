"""wgmma issue-rate probe -- the other half of an L3 timeline's ratio.

The TMA unit says a producer warp delivers one box every 270 ns. Nothing so far
says how fast the tensor core retires the instructions that consume it, so any
claim that the math column covers the copy column has been an assertion. This
measures the math side.

One warpgroup (or two) issues `wgmma.mma_async.m64nNk16.f32.bf16.bf16` back to
back out of resident shared memory: no TMA, no global traffic, no barriers in
the loop.

Questions, in the order they change a design:

M0  Is the instruction even doing what we think? `--check` runs ONE wgmma on
    real data and compares D against torch. A rate measured on an unverified
    instruction is a measurement of an unknown, and both the smem descriptors
    and the accumulator register mapping are easy to get subtly wrong.

M1  What is one wgmma worth? Sweep N over 8..256 and report CYCLES PER
    INSTRUCTION -- clock-invariant, which matters because clocks are unpinnable
    here. If cycles/instruction is flat in N the cost is issue overhead and a
    bigger N is free FLOPs; if it scales with N the tensor core is the limit.

M2  How many must be in flight? Sweep instructions per commit group and the
    wait_group depth. This is the math column's version of the TMA ring depth,
    and it sets how many accumulator registers a stage must hold -- an L2
    budget decision, not a codegen one.

M3  What does wgmma.wait_group cost at each depth? Same sweep, read down the
    WAIT axis at fixed group size.

M4  Does ONE warpgroup saturate the tensor core? 128 threads against 256, same
    work per warpgroup. If one saturates, a second math warpgroup is pure
    register pressure; if it does not, the seesaw schedules are justified.

MS  What about the WARP-level `mma.sync.m16n8k16`? M1 finds wgmma useless below
    N = 64, which leaves an obvious hole: what should a small output tile use
    instead? mma.sync reads operands from registers rather than shared memory,
    so it has no minimum N to speak of. MS1 sweeps independent accumulators
    (1 chains them and measures LATENCY, >1 measures throughput), MS2 sweeps
    warps per SM, and the summary puts both instructions on one axis --
    FLOP per cycle per SM -- which is the only fair way to compare a warp
    instruction with a warpgroup one.

Run:
    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/compute/mma_rate.py
    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/compute/mma_rate.py \
        --sweeps M1 --json profiles/hardware-unit-test/mma.json
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
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            return d
    return here.parent


_REPO = _find_repo()
_SRC = Path(__file__).resolve().with_suffix(".cu")
_CUTLASS = Path(os.environ.get("CUTLASS_DIR", "/data/user/jzou521/codes/cuda/cutlass"))

M_TILE, K_TILE = 64, 16
N_SM = 132
TARGET_US = 100.0

# H100 SXM5 dense bf16 peak, 989.4 TFLOP/s at a 1755 MHz boost over 132 SMs.
# [I] datasheet-derived, and used ONLY as a yardstick for "are we at peak" --
# every constant this probe reports is a measured cycle count, which does not
# depend on it. Clocks are unpinnable here, so a FLOP/s figure would move with
# the clock and a cycle count does not.
PEAK_FLOP_PER_CYCLE_PER_SM = 989.4e12 / (1.755e9 * 132)


def build(verbose: bool = False) -> Path:
    tag = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    d = _REPO / ".cache" / "cuda_ext" / f"mma_rate_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    out = d / "libmma_rate.so"
    if out.exists():
        return out
    cuda_home = os.environ.get("CUDA_HOME", "/data/apps/cuda/13.1")
    cmd = [
        "nvcc", "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
        # NOT -arch=sm_90a. On this toolchain that resolves to virtual arch
        # compute_90, and ptxas then REJECTS every 90a-only instruction --
        # wgmma.fence, wgmma.commit_group, wgmma.wait_group. The explicit
        # gencode is what actually selects compute_90a.
        "-gencode", "arch=compute_90a,code=sm_90a",
        "--expt-relaxed-constexpr", f"-I{_CUTLASS}/include",
        "-o", str(out), str(_SRC),
        f"-L{cuda_home}/lib64/stubs", "-lcuda",
    ]
    if verbose:
        print("[build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    # C7515 means ptxas serialized the wgmma pipeline, which would make every
    # rate below the serialized one. Treat it as a build failure, not a note.
    if "C7515" in r.stderr:
        raise RuntimeError("ptxas serialized the wgmma pipeline (C7515); "
                           "accumulator fences are missing:\n" + r.stderr)
    return out


class Probe:
    def __init__(self, verbose: bool = False):
        self._lib = ctypes.CDLL(str(build(verbose)))
        self._lib.mma_probe_rate.restype = ctypes.c_int
        self._lib.mma_probe_rate.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p]
        self._lib.mma_probe_check.restype = ctypes.c_int
        self._lib.mma_probe_check.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p]
        self._lib.mma_probe_sync_rate.restype = ctypes.c_int
        self._lib.mma_probe_sync_rate.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p]
        self._lib.mma_probe_sync_check.restype = ctypes.c_int
        self._lib.mma_probe_sync_check.argtypes = [ctypes.c_void_p] * 4
        self._lib.mma_probe_cfg.restype = ctypes.c_int
        self._lib.mma_probe_cfg_count.restype = ctypes.c_int
        self.n_cfg = self._lib.mma_probe_cfg_count()

    def cfg(self, i):
        return tuple(self._lib.mma_probe_cfg(i, f) for f in (0, 1, 2))

    def rate(self, cfg, n_ctas, n_threads, a, b, trip, sink, cycles):
        rc = self._lib.mma_probe_rate(
            cfg, n_ctas, n_threads, ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()), trip,
            ctypes.c_void_p(sink.data_ptr()),
            ctypes.c_void_p(cycles.data_ptr()), None)
        if rc != 0:
            raise RuntimeError(f"mma_probe_rate rc={rc}")

    def sync_rate(self, nacc, n_ctas, n_threads, a, b, trip, sink, cycles):
        rc = self._lib.mma_probe_sync_rate(
            nacc, n_ctas, n_threads, ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b.data_ptr()), trip,
            ctypes.c_void_p(sink.data_ptr()),
            ctypes.c_void_p(cycles.data_ptr()), None)
        if rc != 0:
            raise RuntimeError(f"mma_probe_sync_rate rc={rc}")

    def sync_check(self, a, b, out):
        rc = self._lib.mma_probe_sync_check(
            ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), None)
        if rc != 0:
            raise RuntimeError(f"mma_probe_sync_check rc={rc}")

    def check(self, n, a, b, out):
        rc = self._lib.mma_probe_check(
            n, ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), None)
        if rc != 0:
            raise RuntimeError(f"mma_probe_check rc={rc}")


def _bench(run, reps):
    try:
        from flash_vla.bench import bench_gpu_time
    except ImportError:
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        beg = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            beg[i].record(); run(); end[i].record()
        torch.cuda.synchronize()
        return [b.elapsed_time(e) for b, e in zip(beg, end)], "events"
    return bench_gpu_time(run, enable_cupti=True, cold_l2_cache=False,
                          dry_run_iters=3, repeat_iters=reps), "cupti"


TIMER = ""


def _time_us(run, reps=20) -> float:
    global TIMER
    s, TIMER = _bench(run, reps)
    s = sorted(s)
    return s[len(s) // 2] * 1000.0


# ------------------------------------------------------------------- checking
def check_all(probe, dev) -> list[dict]:
    """M0. Every N the rate sweep uses, verified against torch before any
    number from it is believed."""
    rows = []
    for n in (8, 16, 32, 64, 128, 256):
        torch.manual_seed(n)
        a = torch.randn(M_TILE, K_TILE, dtype=torch.bfloat16, device=dev)
        b = torch.randn(n, K_TILE, dtype=torch.bfloat16, device=dev)
        out = torch.zeros(M_TILE, n, dtype=torch.float32, device=dev)
        probe.check(n, a.contiguous(), b.contiguous(), out)
        torch.cuda.synchronize()
        ref = (a.float() @ b.float().T)
        err = (out - ref).abs().max().item()
        scale = ref.abs().max().item()
        rows.append(dict(n=n, max_abs_err=err, ref_max=scale,
                         rel=err / max(scale, 1e-9), ok=err / max(scale, 1e-9) < 2e-2))
    return rows


def check_sync(probe, dev) -> dict:
    """MS0. The m16n8k16 fragment layout, verified rather than recalled."""
    torch.manual_seed(7)
    a = torch.randn(16, 16, dtype=torch.bfloat16, device=dev)
    b = torch.randn(8, 16, dtype=torch.bfloat16, device=dev)
    out = torch.zeros(16, 8, dtype=torch.float32, device=dev)
    probe.sync_check(a.contiguous(), b.contiguous(), out)
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    err = (out - ref).abs().max().item()
    scale = max(ref.abs().max().item(), 1e-9)
    return dict(max_abs_err=err, rel=err / scale, ok=err / scale < 2e-2)


def run_sync(probe, a, b, sink, cycles, nacc, n_ctas, n_threads) -> dict:
    """One mma.sync configuration. Reported on the same FLOP/cycle/SM axis as
    wgmma, which is the only way to compare a warp instruction with a
    warpgroup one."""
    warps = n_threads // 32
    flop_per_inst = 2.0 * 16 * 8 * 16

    def go(t):
        return lambda: probe.sync_rate(nacc, n_ctas, n_threads, a, b, t, sink,
                                       cycles)

    probe.sync_rate(nacc, n_ctas, n_threads, a, b, 8, sink, cycles)
    torch.cuda.synchronize()
    us8 = _time_us(go(8), reps=5)
    trip = max(16, min(400000, int(8 * TARGET_US / max(us8, 1e-3))))
    us = _time_us(go(trip))
    torch.cuda.synchronize()

    cyc_med = cycles[:n_ctas].float().median().item()
    inst_per_warp = trip * nacc
    cyc_per_inst = cyc_med / inst_per_warp
    flop_per_cyc_sm = warps * flop_per_inst / cyc_per_inst
    flops = n_ctas * warps * inst_per_warp * flop_per_inst
    return dict(kind="mma.sync", nacc=nacc, warps=warps, n_ctas=n_ctas,
                trip=trip, us=us, cycles=cyc_med, cyc_per_inst=cyc_per_inst,
                flop_per_cyc_sm=flop_per_cyc_sm,
                pct_peak=100.0 * flop_per_cyc_sm / PEAK_FLOP_PER_CYCLE_PER_SM,
                tflops=flops / (us * 1e-6) / 1e12,
                clock_ghz=cyc_med / (us * 1000.0))


def render_sync(rows) -> str:
    hdr = (f"{'acc':>4} {'warps':>6} {'CTAs':>5} {'us':>8} {'cyc/inst':>9} "
           f"{'FLOP/cyc/SM':>12} {'%peak':>6} {'TFLOP/s':>9} {'GHz':>5}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['nacc']:>4} {r['warps']:>6} {r['n_ctas']:>5} "
                   f"{r['us']:>8.1f} {r['cyc_per_inst']:>9.2f} "
                   f"{r['flop_per_cyc_sm']:>12.0f} {r['pct_peak']:>5.0f}% "
                   f"{r['tflops']:>9.1f} {r['clock_ghz']:>5.2f}")
    return "\n".join(out)


# -------------------------------------------------------------------- sweeps
def run_cfg(probe, a, b, sink, cycles, cfg, n_ctas, n_threads) -> dict:
    n, ngroup, wait = probe.cfg(cfg)
    wg = n_threads // 128

    def go(t):
        return lambda: probe.rate(cfg, n_ctas, n_threads, a, b, t, sink, cycles)

    probe.rate(cfg, n_ctas, n_threads, a, b, 8, sink, cycles)
    torch.cuda.synchronize()
    us8 = _time_us(go(8), reps=5)
    trip = max(16, min(200000, int(8 * TARGET_US / max(us8, 1e-3))))
    us = _time_us(go(trip))
    torch.cuda.synchronize()

    cyc = cycles[:n_ctas].float()
    cyc_med = cyc.median().item()
    inst_per_wg = trip * ngroup
    cyc_per_inst = cyc_med / inst_per_wg
    flops = (n_ctas * wg * inst_per_wg) * 2.0 * M_TILE * n * K_TILE
    tflops = flops / (us * 1e-6) / 1e12
    ideal = 2.0 * M_TILE * n * K_TILE / PEAK_FLOP_PER_CYCLE_PER_SM / wg
    return dict(cfg=cfg, n=n, ngroup=ngroup, wait=wait, n_ctas=n_ctas,
                warpgroups=wg, inflight=ngroup * (wait + 1), trip=trip, us=us,
                cycles=cyc_med, cyc_per_inst=cyc_per_inst, tflops=tflops,
                ideal_cyc=ideal, pct_peak=100.0 * ideal / cyc_per_inst,
                clock_ghz=cyc_med / (us * 1000.0))


def render(rows) -> str:
    hdr = (f"{'N':>5} {'grp':>4} {'wait':>5} {'flight':>7} {'WG':>3} "
           f"{'CTAs':>5} {'us':>8} {'cyc/inst':>9} {'ideal':>7} {'%peak':>6} "
           f"{'TFLOP/s':>9} {'GHz':>5}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['n']:>5} {r['ngroup']:>4} {r['wait']:>5} "
                   f"{r['inflight']:>7} {r['warpgroups']:>3} {r['n_ctas']:>5} "
                   f"{r['us']:>8.1f} {r['cyc_per_inst']:>9.1f} "
                   f"{r['ideal_cyc']:>7.1f} {r['pct_peak']:>5.0f}% "
                   f"{r['tflops']:>9.1f} {r['clock_ghz']:>5.2f}")
    return "\n".join(out)


def sweep_M1(probe, a, b, sink, cycles):
    """M1: what is one wgmma worth? N from 8 to 256, one warpgroup."""
    rows = []
    for n_ctas in (1, N_SM):
        for cfg in range(8):
            rows.append(run_cfg(probe, a, b, sink, cycles, cfg, n_ctas, 128))
    return rows


def sweep_M2(probe, a, b, sink, cycles):
    """M2/M3: how many in flight, and what wait_group depth costs."""
    rows = []
    for cfg in (8, 9, 10, 11, 3, 12, 13, 14, 15, 16, 17):
        rows.append(run_cfg(probe, a, b, sink, cycles, cfg, N_SM, 128))
    return rows


def sweep_M4(probe, a, b, sink, cycles):
    """M4: does one warpgroup saturate the tensor core?"""
    rows = []
    for cfg in (3, 5, 13, 16):
        for n_threads in (128, 256):
            rows.append(run_cfg(probe, a, b, sink, cycles, cfg, N_SM, n_threads))
    return rows


def sweep_MS1(probe, a, b, sink, cycles):
    """MS1: mma.sync -- independent accumulators. 1 chains them (latency)."""
    return [run_sync(probe, a, b, sink, cycles, nacc, N_SM, 128)
            for nacc in (1, 2, 4, 8)]


def sweep_MS2(probe, a, b, sink, cycles):
    """MS2: mma.sync -- warps per SM, at 4 independent accumulators."""
    return [run_sync(probe, a, b, sink, cycles, 4, N_SM, nt)
            for nt in (32, 64, 128, 256)]


SWEEPS = {"M1": sweep_M1, "M2": sweep_M2, "M4": sweep_M4,
          "MS1": sweep_MS1, "MS2": sweep_MS2}
SYNC_SWEEPS = {"MS1", "MS2"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", default="M1,M2,M4,MS1,MS2")
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    a_ = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    dev = "cuda"
    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}",
          flush=True)

    probe = Probe(a_.verbose)
    print(f"[cfg] {probe.n_cfg} compiled (N, group, wait) configurations",
          flush=True)

    print("\n[M0] correctness -- one wgmma against torch, per N", flush=True)
    checks = check_all(probe, dev)
    for r in checks:
        print(f"  n={r['n']:>4}  max|err| {r['max_abs_err']:9.4f}  "
              f"rel {r['rel']:.2e}  {'OK' if r['ok'] else 'FAIL'}", flush=True)
    if not all(r["ok"] for r in checks):
        raise SystemExit("wgmma correctness check FAILED -- descriptors or the "
                         "accumulator mapping are wrong; no rate below would "
                         "mean anything")

    print("\n[MS0] correctness -- one mma.sync.m16n8k16 against torch",
          flush=True)
    cs = check_sync(probe, dev)
    print(f"  max|err| {cs['max_abs_err']:9.4f}  rel {cs['rel']:.2e}  "
          f"{'OK' if cs['ok'] else 'FAIL'}", flush=True)
    if not cs["ok"]:
        raise SystemExit("mma.sync correctness check FAILED -- the fragment "
                         "layout is wrong; no rate below would mean anything")
    results_ms0 = cs

    a = torch.randn(M_TILE, K_TILE, dtype=torch.bfloat16, device=dev)
    b = torch.randn(256, K_TILE, dtype=torch.bfloat16, device=dev)
    sink = torch.zeros(N_SM * 256, dtype=torch.float32, device=dev)
    cycles = torch.zeros(N_SM, dtype=torch.int64, device=dev)

    results = {"checks": checks, "check_sync": results_ms0, "sweeps": {}}
    for key in [s.strip().upper() for s in a_.sweeps.split(",") if s.strip()]:
        if key not in SWEEPS:
            raise SystemExit(f"unknown sweep {key!r}; have {sorted(SWEEPS)}")
        print(f"\n[sweep {key}] {SWEEPS[key].__doc__.splitlines()[0]}", flush=True)
        rows = SWEEPS[key](probe, a, b, sink, cycles)
        results["sweeps"][key] = rows
        print(render_sync(rows) if key in SYNC_SWEEPS else render(rows),
              flush=True)

    # MS3, the question MS exists for: on one axis, which instruction wins at
    # each output-tile N? Both columns are FLOP per cycle per SM, so a warp
    # instruction and a warpgroup one are finally comparable.
    m1 = results["sweeps"].get("M1")
    ms = results["sweeps"].get("MS1", []) + results["sweeps"].get("MS2", [])
    if m1 and ms:
        best_sync = max(ms, key=lambda r: r["flop_per_cyc_sm"])
        print(f"\n[MS3] wgmma vs mma.sync, FLOP per cycle per SM "
              f"(architectural peak {PEAK_FLOP_PER_CYCLE_PER_SM:.0f})",
              flush=True)
        print(f"  best mma.sync: {best_sync['flop_per_cyc_sm']:.0f} "
              f"({best_sync['pct_peak']:.0f}% of peak) at {best_sync['warps']} "
              f"warps x {best_sync['nacc']} accumulators", flush=True)
        print(f"  {'wgmma N':>8} {'FLOP/cyc/SM':>12} {'%peak':>6}  verdict",
              flush=True)
        for r in [x for x in m1 if x["n_ctas"] == N_SM]:
            f = 2.0 * M_TILE * r["n"] * K_TILE / r["cyc_per_inst"]
            win = ("wgmma" if f > best_sync["flop_per_cyc_sm"] else "mma.sync")
            ratio = f / best_sync["flop_per_cyc_sm"]
            print(f"  {r['n']:>8} {f:>12.0f} "
                  f"{100.0 * f / PEAK_FLOP_PER_CYCLE_PER_SM:>5.0f}%  "
                  f"{win} by {max(ratio, 1/ratio):.2f}x", flush=True)
        results["ms3_best_sync"] = best_sync

    results["timer"] = TIMER
    if TIMER != "cupti":
        print(f"\n[timer] {TIMER} -- includes launch overhead", flush=True)
    if a_.json:
        Path(a_.json).write_text(json.dumps(results, indent=2))
        print(f"\n[json] {a_.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
