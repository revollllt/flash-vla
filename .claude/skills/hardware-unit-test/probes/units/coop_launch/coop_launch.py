"""coop_launch -- what a device-wide barrier costs, against the relaunch it replaces.

`launch.lat.dev.ramp` says every kernel starts ~1.24 us in debt. The decision to
reject cooperative launch for this repo's decoder rested on the ASSUMPTION that
a grid barrier costs about the same -- marked `[I, UNMEASURED]` and load-bearing:
if a `grid_sync` is much cheaper than a relaunch, a persistent kernel with N
barriers beats N launches, and the megakernel direction changes.

Three questions, each with the hypothesis it kills:

  C1  What does ONE grid_sync cost, against grid size?
      Kills "a grid barrier is free" and "a grid barrier is a launch". Measured
      as the DIFFERENCE between two modes of one kernel on one grid, so nothing
      but the barrier moved.

  C2  How many blocks can a cooperative launch actually place?
      Kills "cooperative launch covers the machine". `cudaLaunchCooperativeKernel`
      REFUSES a grid it cannot make co-resident, so the limit is enumerated from
      the API's own answer rather than inferred from the SM count.

  C3  Is grid_sync cheaper than the relaunch it replaces?
      The decisive pair, and the only one that changes a design. Same loop body,
      same grid, same work: once as N barriers inside one cooperative launch,
      once as N ordinary launches.

Names are the vendor's throughout -- `grid_group`, `grid_sync`, `num_blocks`,
`cudaLaunchCooperativeKernel`, `max_active_blocks_per_sm`. [vocabulary.md]

    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/units/coop_launch/coop_launch.py \\
        --json profiles/hardware-unit-test/coop.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hut import abi, harness  # noqa: E402

_SRC = Path(__file__).resolve().with_suffix(".cu")

N_SM = 132
# `cfg` selects what the cooperative kernel's loop contains; `mode` selects the
# LAUNCH PATH. Keeping them separate is the point: the C1 pair must differ only
# in the barrier, so BOTH its arms are cooperative launches. Collapsing these
# two axes would have made the "empty" arm an ordinary launch and turned C1 into
# a measurement of the launch path. [protocol.md rule 3]
CFG_GRID_SYNC, CFG_EMPTY = 0, 1
PATH_COOPERATIVE, PATH_RELAUNCH = 0, 1

# cudaErrorCooperativeLaunchTooLarge. Recorded, not raised: what the API refuses
# is the measurement in C2.
ERR_TOO_LARGE = 720

# Enough barriers that the per-launch ramp is a small share of the span, few
# enough that a 132-block grid still finishes inside a Slurm slot.
N_ITERS = 2000

# Median over this many launches, matching every other unit here. [rule 14]
REPS = 7


class Probe:
    def __init__(self, verbose: bool = False):
        self.unit = harness.load(_SRC, verbose=verbose)
        self.unit.lib.hut_max_blocks.restype = ctypes.c_int32
        self.unit.lib.hut_max_blocks.argtypes = [ctypes.c_int32]

    def max_blocks(self, n_threads: int) -> int:
        """Co-resident block ceiling from the runtime's own occupancy query."""
        return self.unit.lib.hut_max_blocks(n_threads)

    def launch_rc(self, *, cfg, path, n_ctas, n_threads, n_iters, bufs) -> int:
        """Launch and return the raw rc. A refusal is DATA, so it is not raised."""
        cycles, sm_id, sink = bufs
        p = abi.HutParams(cfg=cfg, mode=path, n_ctas=n_ctas,
                          n_threads=n_threads, k_tile_count=1)
        p.opt[0] = n_iters
        # Both structs are bound to locals: byref() on a temporary lets it be
        # collected before the call returns.
        b = abi.buffers(cycles_a=cycles, sm_id=sm_id, sink=sink)
        # The CURRENT stream, never NULL. A launch on the legacy default stream
        # is not recorded by a capture in progress, which produced an empty
        # graph and a replay that measured nothing.
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        return self.unit.lib.hut_launch(ctypes.byref(p), ctypes.byref(b), stream)


def _time_us(fn) -> tuple[float, float]:
    return harness.time_us(fn, reps=REPS)


# ------------------------------------------------------------------ C2 first
def sweep_placement(probe, n_threads, bufs) -> list[dict]:
    """C2: what the API actually accepts. Run FIRST -- it bounds C1 and C3.

    A grid above the co-resident limit does not deadlock here, it is refused,
    which is the whole reason this can be enumerated rather than guessed.
    """
    ceiling = probe.max_blocks(n_threads)
    # The ladder must CROSS the query's answer, or it proves nothing: a sweep
    # that stops below the limit only shows that legal grids are legal.
    grids = [32, 66, N_SM, N_SM + 1, 2 * N_SM, 4 * N_SM]
    if ceiling > 0:
        grids += [ceiling - 1, ceiling, ceiling + 1, 2 * ceiling]
    rows = []
    for n_ctas in sorted(set(grids)):
        rc = probe.launch_rc(cfg=CFG_GRID_SYNC, path=PATH_COOPERATIVE,
                             n_ctas=n_ctas, n_threads=n_threads, n_iters=8,
                             bufs=bufs)
        torch.cuda.synchronize() if rc == 0 else None
        rows.append(dict(n_ctas=n_ctas, n_threads=n_threads, rc=rc,
                         accepted=(rc == 0),
                         too_large=(rc == ERR_TOO_LARGE),
                         max_blocks_query=ceiling))
    return rows


def render_placement(rows) -> str:
    out = [f"{'blocks':>8} {'threads':>8} {'accepted':>9} {'rc':>5}  note",
           "-" * 56]
    for r in rows:
        note = ("" if r["accepted"] else
                "cudaErrorCooperativeLaunchTooLarge" if r["too_large"]
                else "rc passed through unchanged")
        out.append(f"{r['n_ctas']:>8} {r['n_threads']:>8} "
                   f"{str(r['accepted']):>9} {r['rc']:>5}  {note}")
    out.append(f"  max_active_blocks_per_sm x SMs = {rows[0]['max_blocks_query']}"
               "  [cudaOccupancyMaxActiveBlocksPerMultiprocessor]")
    return "\n".join(out)


# --------------------------------------------------------------- C1 and C3
def measure_sync(probe, n_ctas, n_threads, bufs) -> dict | None:
    """C1: one grid_sync, as the difference between two modes of one kernel."""
    cycles, sm_id, sink = bufs

    def run(cfg):
        def go():
            rc = probe.launch_rc(cfg=cfg, path=PATH_COOPERATIVE, n_ctas=n_ctas,
                                 n_threads=n_threads, n_iters=N_ITERS, bufs=bufs)
            if rc != 0:
                raise RuntimeError(f"cudaLaunchCooperativeKernel rc={rc}")
        return go

    us_sync, sp_sync = _time_us(run(CFG_GRID_SYNC))
    cyc_sync = float(cycles[:n_ctas].float().median())
    us_empty, sp_empty = _time_us(run(CFG_EMPTY))
    cyc_empty = float(cycles[:n_ctas].float().median())

    per_sync_us = (us_sync - us_empty) / N_ITERS
    per_sync_cyc = (cyc_sync - cyc_empty) / N_ITERS
    return dict(n_ctas=n_ctas, n_threads=n_threads, n_iters=N_ITERS,
                us_sync=us_sync, us_empty=us_empty,
                cyc_sync=cyc_sync, cyc_empty=cyc_empty,
                per_sync_us=per_sync_us, per_sync_ns=per_sync_us * 1000.0,
                per_sync_cyc=per_sync_cyc,
                spread=max(sp_sync, sp_empty))


def measure_relaunch(probe, n_ctas, n_threads, bufs, n_launches=64) -> dict:
    """C3: the same work as N ordinary launches -- what grid_sync replaces.

    CAPTURED IN A CUDA GRAPH. Dispatching these from Python through ctypes
    measures Python: the first attempt read 6345 ns per launch against
    `launch.lat.dev.ramp`'s measured 1.24 us for the same thing, a 5x host-side
    inflation that would have made grid_sync look far better than it is. The
    graph replays the same launches with the host out of the loop, which is the
    only form in which this comparison means anything.
    """
    cycles, sm_id, sink = bufs
    per = max(1, N_ITERS // n_launches)

    def enqueue():
        for _ in range(n_launches):
            rc = probe.launch_rc(cfg=CFG_EMPTY, path=PATH_RELAUNCH,
                                 n_ctas=n_ctas, n_threads=n_threads,
                                 n_iters=per, bufs=bufs)
            if rc != 0:
                raise RuntimeError(f"relaunch rc={rc}")

    # Warm, then capture. A capture on the default stream is illegal, so this
    # runs on a side stream the way torch requires.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        enqueue()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        enqueue()
    torch.cuda.synchronize()

    us, spread = _time_us(graph.replay)
    return dict(n_ctas=n_ctas, n_launches=n_launches, iters_per_launch=per,
                us=us, per_launch_us=us / n_launches,
                per_launch_ns=us / n_launches * 1000.0, spread=spread,
                captured="cuda_graph")


def render_sync(rows) -> str:
    out = [f"{'blocks':>7} {'thr':>5} {'sync us':>9} {'empty us':>9} "
           f"{'ns/sync':>9} {'cyc/sync':>9} {'spread':>7}", "-" * 62]
    for r in rows:
        out.append(f"{r['n_ctas']:>7} {r['n_threads']:>5} {r['us_sync']:>9.1f} "
                   f"{r['us_empty']:>9.1f} {r['per_sync_ns']:>9.1f} "
                   f"{r['per_sync_cyc']:>9.1f} {100*r['spread']:>6.1f}%")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=256,
                    help="threads per block; changes max_active_blocks_per_sm")
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}",
          flush=True)

    probe = Probe(a.verbose)
    n = 4 * N_SM
    bufs = (torch.zeros(n, dtype=torch.int64, device="cuda"),
            torch.zeros(n, dtype=torch.int32, device="cuda"),
            torch.zeros(n, dtype=torch.float32, device="cuda"))

    results = {"unit": "coop_launch", "threads": a.threads, "n_iters": N_ITERS,
               "reps": REPS}

    print("\n[C2] what cudaLaunchCooperativeKernel accepts -- gates C1 and C3",
          flush=True)
    place = sweep_placement(probe, a.threads, bufs)
    results["placement"] = place
    print(render_placement(place), flush=True)

    ok = [r["n_ctas"] for r in place if r["accepted"]]
    if not ok:
        raise SystemExit("no cooperative grid was accepted; nothing below is "
                         "measurable on this device")
    grids = [g for g in (32, 66, N_SM) if g in ok]

    print("\n[C1] one grid_sync, as the difference between two modes of one "
          "kernel", flush=True)
    rows = [measure_sync(probe, g, a.threads, bufs) for g in grids]
    results["sync"] = rows
    print(render_sync(rows), flush=True)

    print("\n[C3] grid_sync against the relaunch it replaces", flush=True)
    big = max(grids)
    rl = measure_relaunch(probe, big, a.threads, bufs)
    results["relaunch"] = rl
    sync_ns = [r["per_sync_ns"] for r in rows if r["n_ctas"] == big][0]
    ratio = rl["per_launch_ns"] / sync_ns if sync_ns > 0 else float("inf")
    results["ratio_relaunch_over_sync"] = ratio
    print(f"    grid_sync   {sync_ns:>8.1f} ns   at {big} blocks", flush=True)
    print(f"    relaunch    {rl['per_launch_ns']:>8.1f} ns   "
          f"({rl['n_launches']} launches in a CUDA graph, "
          f"{rl['iters_per_launch']} iters each)", flush=True)
    print(f"    -> a relaunch costs {ratio:.2f}x a grid_sync", flush=True)
    print(f"    cross-check: launch.lat.dev.ramp measured a launch at 1240 ns "
          f"independently; this reads {rl['per_launch_ns']:.0f}.", flush=True)

    results["timer"] = harness.TIMER_USED
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
        print(f"\n[json] {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
