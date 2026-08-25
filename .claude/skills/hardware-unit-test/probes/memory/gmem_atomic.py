"""Global-atomic and gmem-counter probe -- what does ordering actually cost?

Two design decisions in this project are currently made without a number:
whether a split-K reduction should accumulate with `red.global.add` or write
partials and pay another launch, and how fine-grained a megakernel task can be
before its counter protocol costs more than the task. Both need the atomic unit
measured in isolation, which is what this does.

Questions, in the order they change a design:

A1  Does the RETURN VALUE cost anything? `red` is fire-and-forget, `atom`
    returns the old value and must round-trip. Decisive pair: identical traffic,
    uncontended, red vs atom. If red is much cheaper, every accumulate that does
    not need the old value should be one -- a mechanical, free win.

A2  What does CONTENTION cost? Sweep the number of distinct addresses at a fixed
    op count, at two placements: 4 B apart (a warp inside one 128 B sector) and
    128 B apart (a line each). The pair separates serialisation inside the L2
    atomic unit from address bandwidth.

A3  Does WIDTH pay? u32 / f32 / f16x2 / bf16x2 / v2.f32 / v4.f32 at equal op
    count. If the unit is per-transaction rather than per-byte -- as TMA turned
    out to be -- packing is free throughput and changes how a reduction is laid
    out.

A4  What does SCOPE cost? `.cta` / `.gpu` / `.sys` on the same instruction.
    Decides whether a megakernel's ordering can use a narrower scope.

A5  What is the ARRIVE -> OBSERVE latency, and does it grow with the number of
    observers? Ping-pong between two CTAs, timed from the host so the number
    never crosses two SMs' unsynchronised clocks. This is the constant the
    task-graph counter protocol is built on.

Run:
    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/memory/gmem_atomic.py
    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/memory/gmem_atomic.py \
        --sweeps A2 --json profiles/hardware-unit-test/atomic.json
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

# Must match the Op enum in the .cu, in order.
OPS = ["red.u32", "atom.u32", "red.f32", "atom.f32", "red.f16x2", "red.bf16x2",
       "red.v2f32", "red.v4f32", "atom.cas", "atom.exch",
       "red.u32.cta", "red.u32.sys", "atom.u32.cta", "atom.u32.sys"]
OP_IDX = {name: i for i, name in enumerate(OPS)}
# Bytes the instruction updates, so a width sweep can be read as bandwidth as
# well as op rate. The two readings disagree exactly when the unit is
# per-transaction, which is the point of A3.
OP_BYTES = {"red.u32": 4, "atom.u32": 4, "red.f32": 4, "atom.f32": 4,
            "red.f16x2": 4, "red.bf16x2": 4, "red.v2f32": 8, "red.v4f32": 16,
            "atom.cas": 4, "atom.exch": 4, "red.u32.cta": 4, "red.u32.sys": 4,
            "atom.u32.cta": 4, "atom.u32.sys": 4}

# 256 MiB: exactly 65536 addresses at a 4096 B page stride, which is more than
# the 33792 threads of the default grid, so "uncontended" is reachable at every
# placement rather than only at the narrow ones.
BUF_B = 256 * 1024 * 1024
N_SM = 132
N_THREADS = 256
TARGET_US = 120.0        # size trip to land here, so timing noise is small


def build(verbose: bool = False) -> Path:
    tag = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    d = _REPO / ".cache" / "cuda_ext" / f"gmem_atomic_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    out = d / "libgmem_atomic.so"
    if out.exists():
        return out
    cuda_home = os.environ.get("CUDA_HOME", "/data/apps/cuda/13.1")
    cmd = ["nvcc", "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
           "-arch=sm_90a", "--expt-relaxed-constexpr",
           "-o", str(out), str(_SRC),
           f"-L{cuda_home}/lib64/stubs", "-lcuda"]
    if verbose:
        print("[build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    return out


class Probe:
    def __init__(self, verbose: bool = False):
        self._lib = ctypes.CDLL(str(build(verbose)))
        self._lib.atom_probe_rate.restype = ctypes.c_int
        self._lib.atom_probe_rate.argtypes = [
            ctypes.c_int, ctypes.c_void_p] + [ctypes.c_int] * 5 + [
            ctypes.c_void_p, ctypes.c_void_p]
        self._lib.atom_probe_pingpong.restype = ctypes.c_int
        self._lib.atom_probe_pingpong.argtypes = [
            ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p,
                                                     ctypes.c_void_p]
        n = self._lib.atom_probe_op_count()
        assert n == len(OPS), f"OPS list has {len(OPS)}, .cu has {n}"

    def rate(self, op, base, n_addr, stride_b, n_ctas, n_threads, trip, sink):
        rc = self._lib.atom_probe_rate(
            OP_IDX[op], ctypes.c_void_p(base), n_addr, stride_b, n_ctas,
            n_threads, trip, ctypes.c_void_p(sink), None)
        if rc != 0:
            raise RuntimeError(f"atom_probe_rate rc={rc}")

    def pingpong(self, ctr, rounds, advance_atomic, n_pollers, dbg):
        rc = self._lib.atom_probe_pingpong(
            ctypes.c_void_p(ctr), rounds, int(advance_atomic), n_pollers,
            ctypes.c_void_p(dbg), None)
        if rc != 0:
            raise RuntimeError(f"atom_probe_pingpong rc={rc}")


def _bench(run, reps):
    """Per-iteration GPU times in ms, plus the timer's name. The repo's CUPTI
    harness where available; CUDA events elsewhere. The two differ by the launch
    overhead events include, so which one ran is reported, never assumed."""
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


def run_rate(probe, buf, sink, op, *, n_addr, stride_b, n_ctas=N_SM,
             n_threads=N_THREADS, trip=None) -> dict:
    """One configuration. `trip` is calibrated rather than fixed: contention
    changes the rate by orders of magnitude, and one constant trip would make
    the contended points milliseconds long and the free ones unmeasurably
    short."""
    threads = n_ctas * n_threads
    assert n_addr * stride_b <= BUF_B, f"{n_addr}x{stride_b} exceeds the buffer"

    def go(t):
        return lambda: probe.rate(op, buf.data_ptr(), n_addr, stride_b,
                                  n_ctas, n_threads, t, sink.data_ptr())

    probe.rate(op, buf.data_ptr(), n_addr, stride_b, n_ctas, n_threads, 8,
               sink.data_ptr())
    torch.cuda.synchronize()
    us8 = _time_us(go(8), reps=5)
    trip = trip or max(8, min(4096, int(8 * TARGET_US / max(us8, 1e-3))))
    us = _time_us(go(trip))
    ops = threads * trip
    return dict(op=op, n_addr=n_addr, stride_b=stride_b, n_ctas=n_ctas,
                n_threads=n_threads, trip=trip, threads=threads,
                share=threads / n_addr, us=us, gops=ops / (us * 1e-6) / 1e9,
                ns_per_op=us * 1000.0 / trip,
                gbs=ops * OP_BYTES[op] / (us * 1e-6) / 1e9)


def render(rows) -> str:
    hdr = (f"{'op':>13} {'addrs':>7} {'stride':>7} {'share':>7} {'trip':>6} "
           f"{'us':>9} {'Gop/s':>8} {'ns/op':>8} {'GB/s':>8}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['op']:>13} {r['n_addr']:>7} {r['stride_b']:>7} "
                   f"{r['share']:>7.0f} {r['trip']:>6} {r['us']:>9.2f} "
                   f"{r['gops']:>8.2f} {r['ns_per_op']:>8.1f} {r['gbs']:>8.1f}")
    return "\n".join(out)


# --------------------------------------------------------------------- sweeps
FREE = 65536          # more addresses than threads: no two threads share one


def sweep_A1(probe, buf, sink, ctr, dbg):
    """A1: does the return value cost anything? red vs atom, uncontended."""
    rows = []
    for op in ("red.u32", "atom.u32", "red.f32", "atom.f32", "atom.exch",
               "atom.cas"):
        rows.append(run_rate(probe, buf, sink, op, n_addr=FREE, stride_b=128))
    return rows


def sweep_A2(probe, buf, sink, ctr, dbg):
    """A2: what does contention cost? distinct addresses, two placements."""
    rows = []
    for stride_b in (4, 128):
        for n_addr in (1, 4, 32, 256, 2048, 16384, FREE):
            for op in ("red.u32", "atom.u32"):
                rows.append(run_rate(probe, buf, sink, op, n_addr=n_addr,
                                     stride_b=stride_b))
    return rows


def sweep_A3(probe, buf, sink, ctr, dbg):
    """A3: does width pay? equal op count, uncontended."""
    rows = []
    for op in ("red.u32", "red.f32", "red.f16x2", "red.bf16x2", "red.v2f32",
               "red.v4f32"):
        rows.append(run_rate(probe, buf, sink, op, n_addr=FREE, stride_b=128))
    return rows


def sweep_A4(probe, buf, sink, ctr, dbg):
    """A4: what does scope cost? cta / gpu / sys, uncontended and contended."""
    rows = []
    for n_addr in (FREE, 32):
        for op in ("red.u32.cta", "red.u32", "red.u32.sys",
                   "atom.u32.cta", "atom.u32", "atom.u32.sys"):
            rows.append(run_rate(probe, buf, sink, op, n_addr=n_addr,
                                 stride_b=128))
    return rows


def sweep_A5(probe, ctr, dbg, rounds=2000):
    """A5: arrive -> observe latency, and whether observers make it worse."""
    rows = []
    for advance_atomic in (1, 0):
        for n_pollers in (0, 6, 30, 130):
            ctr.zero_(); dbg.zero_(); torch.cuda.synchronize()

            def go():
                ctr.zero_()
                probe.pingpong(ctr.data_ptr(), rounds, advance_atomic,
                               n_pollers, dbg.data_ptr())

            us = _time_us(go, reps=10)
            torch.cuda.synchronize()
            if int(dbg.max().item()) != 0:
                raise RuntimeError(f"watchdog fired: {dbg[dbg != 0][:8].tolist()}")
            rows.append(dict(advance="red.release.add" if advance_atomic
                             else "st.release", n_pollers=n_pollers,
                             ctas=2 + n_pollers, rounds=rounds, us=us,
                             ns_per_hop=us * 1000.0 / (2 * rounds)))
    return rows


def render_A5(rows) -> str:
    hdr = (f"{'advance':>18} {'observers':>10} {'CTAs':>6} {'rounds':>7} "
           f"{'us':>9} {'ns/hop':>8}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['advance']:>18} {r['n_pollers']:>10} {r['ctas']:>6} "
                   f"{r['rounds']:>7} {r['us']:>9.1f} {r['ns_per_hop']:>8.1f}")
    return "\n".join(out)


SWEEPS = {"A1": sweep_A1, "A2": sweep_A2, "A3": sweep_A3, "A4": sweep_A4}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", default="A1,A2,A3,A4,A5")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}",
          flush=True)

    probe = Probe(a.verbose)
    buf = torch.zeros(BUF_B // 4, dtype=torch.int32, device="cuda")
    sink = torch.zeros(N_SM * N_THREADS, dtype=torch.int32, device="cuda")
    ctr = torch.zeros(1, dtype=torch.int32, device="cuda")
    dbg = torch.zeros(280, dtype=torch.int64, device="cuda")

    results = {"sweeps": {}}
    for key in [s.strip().upper() for s in a.sweeps.split(",") if s.strip()]:
        if key == "A5":
            print(f"\n[sweep A5] {sweep_A5.__doc__.splitlines()[0]}", flush=True)
            rows = sweep_A5(probe, ctr, dbg, a.rounds)
            print(render_A5(rows), flush=True)
        elif key in SWEEPS:
            print(f"\n[sweep {key}] {SWEEPS[key].__doc__.splitlines()[0]}",
                  flush=True)
            rows = SWEEPS[key](probe, buf, sink, ctr, dbg)
            print(render(rows), flush=True)
        else:
            raise SystemExit(f"unknown sweep {key!r}")
        results["sweeps"][key] = rows
    results["timer"] = TIMER
    if TIMER != "cupti":
        print(f"\n[timer] {TIMER} -- includes launch overhead; not comparable "
              f"with constants recorded under CUPTI", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
        print(f"\n[json] {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
