"""Global-atomic throughput and gmem-counter latency.

Questions, in the order they change a design:

A1  Does the RETURN VALUE cost anything? `red` discards it, `atom` returns it.
    If the difference is large, a fire-and-forget accumulate is a different
    instruction from a fetch-and-add and should be chosen deliberately.

A2  What does CONTENTION cost, and does address PLACEMENT change it? Distinct
    addresses from 1 to uncontended, at a 4 B and a 128 B stride. The two
    strides put the same number of addresses in different numbers of cache
    lines, which is the axis a reduction layout actually controls.

A3  Does WIDTH pay? Equal op count, widths from u32 to v4.f32. If the unit is
    per-transaction the wide forms move more bytes for the same cost; if it is
    per-byte they do not.

A4  What does SCOPE cost? `.cta` / `.gpu` / `.sys`, contended and not.

A5  How long from one CTA's release-increment to another's acquire-observe, and
    does adding OBSERVERS make it worse? That decides how many consumers one
    counter can gate.

This unit declares HUT_NO_SOURCE: the atomics ARE the traffic, so there is no
walk whose coldness could be checked and no host reference to compare against.
A missed increment deadlocks A5's ping-pong rather than returning a wrong rate,
which is the correctness gate rule 11 would otherwise ask for.

Run:
    python3 gmem_atomic.py --sweeps A1,A2,A3,A4,A5 --json /tmp/atomic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hut import abi, harness  # noqa: E402

_SRC = Path(__file__).resolve().with_suffix(".cu")

# Must match the Op enum in the .cuh, in order.
OPS = ["red.u32", "atom.u32", "red.f32", "atom.f32", "red.f16x2", "red.bf16x2",
       "red.v2f32", "red.v4f32", "atom.cas", "atom.exch",
       "red.u32.cta", "red.u32.sys", "atom.u32.cta", "atom.u32.sys"]
OP_IDX = {name: i for i, name in enumerate(OPS)}

# 256 MiB: exactly 65536 addresses at a 4096 B page stride, which is more than
# the 33792 threads of the default grid, so "uncontended" is reachable at every
# placement rather than only at the narrow ones.
BUF_B = 256 * 1024 * 1024
N_SM = 132
N_THREADS = 256
FREE = 65536              # enough addresses that no two threads collide
TARGET_US = 120.0         # size k_tile_count to land here, so noise is small

SITES = {abi_site: name for abi_site, name in
         ((16, "pingpong advance (counter never reached its value)"),
          (17, "pingpong observe (counter never reached the end)"))}


def params(unit, op, *, n_addr, stride_bytes, n_ctas, n_threads, k_tile_count,
           buf) -> abi.HutParams:
    p = abi.HutParams()
    p.cfg = OP_IDX[op]
    p.mode = 0
    p.n_ctas, p.n_threads = n_ctas, n_threads
    p.k_tile_count = k_tile_count
    p.opt[0], p.opt[1] = n_addr, stride_bytes
    p.operand_a = buf.data_ptr()
    return p


def run_rate(unit, buf, sink, op, *, n_addr, stride_bytes, n_ctas=N_SM,
             n_threads=N_THREADS, k_tile_count=None) -> dict:
    """One configuration.

    `k_tile_count` is calibrated rather than fixed: contention changes the rate
    by orders of magnitude, and one constant value would make the contended
    points milliseconds long and the free ones unmeasurably short.
    """
    assert n_addr * stride_bytes <= BUF_B, f"{n_addr}x{stride_bytes} > buffer"
    threads = n_ctas * n_threads
    bufs = abi.buffers(sink=sink)

    def go(k):
        p = params(unit, op, n_addr=n_addr, stride_bytes=stride_bytes,
                   n_ctas=n_ctas, n_threads=n_threads, k_tile_count=k, buf=buf)
        return lambda: unit.launch(p, bufs)

    us8, _ = harness.time_us(go(8), reps=5)
    k = k_tile_count or max(8, min(4096, int(8 * TARGET_US / max(us8, 1e-3))))
    us, spread = harness.time_us(go(k))
    ops = threads * k
    op_bytes = unit.cfg(OP_IDX[op])["op_bytes"]
    return dict(op=op, n_addr=n_addr, stride_bytes=stride_bytes, n_ctas=n_ctas,
                n_threads=n_threads, k_tile_count=k, threads=threads,
                share=threads / n_addr, us=us, spread=spread,
                gops=ops / (us * 1e-6) / 1e9, ns_per_op=us * 1000.0 / k,
                gbs=ops * op_bytes / (us * 1e-6) / 1e9)


def sweep_A1(unit, buf, sink, ctr, dbg):
    """A1: does the return value cost anything? red vs atom, uncontended."""
    return [run_rate(unit, buf, sink, op, n_addr=FREE, stride_bytes=128)
            for op in ("red.u32", "atom.u32", "red.f32", "atom.f32",
                       "atom.exch", "atom.cas")]


def sweep_A2(unit, buf, sink, ctr, dbg):
    """A2: what does contention cost? distinct addresses, two placements."""
    return [run_rate(unit, buf, sink, op, n_addr=n, stride_bytes=sb)
            for sb in (4, 128)
            for n in (1, 4, 32, 256, 2048, 16384, FREE)
            for op in ("red.u32", "atom.u32")]


def sweep_A3(unit, buf, sink, ctr, dbg):
    """A3: does width pay? equal op count, uncontended."""
    return [run_rate(unit, buf, sink, op, n_addr=FREE, stride_bytes=128)
            for op in ("red.u32", "red.f32", "red.f16x2", "red.bf16x2",
                       "red.v2f32", "red.v4f32")]


def sweep_A4(unit, buf, sink, ctr, dbg):
    """A4: what does scope cost? cta / gpu / sys, uncontended and contended."""
    return [run_rate(unit, buf, sink, op, n_addr=n, stride_bytes=128)
            for n in (FREE, 32)
            for op in ("red.u32.cta", "red.u32", "red.u32.sys",
                       "atom.u32.cta", "atom.u32", "atom.u32.sys")]


def sweep_A5(unit, ctr, dbg, rounds=2000):
    """A5: arrive -> observe latency, and whether observers make it worse."""
    rows = []
    for advance_atomic in (1, 0):
        for n_observers in (0, 6, 30, 130):
            p = abi.HutParams()
            p.mode = 1
            p.n_ctas = n_observers
            p.opt[2], p.opt[3] = rounds, advance_atomic
            p.operand_a = ctr.data_ptr()
            bufs = abi.buffers(dbg=dbg)

            def go():
                ctr.zero_()
                unit.launch(p, bufs)

            dbg.zero_()
            us, spread = harness.time_us(go, reps=10)
            torch.cuda.synchronize()
            harness.check_watchdog(dbg, SITES)
            rows.append(dict(
                advance="red.release.add" if advance_atomic else "st.release",
                n_observers=n_observers, n_ctas=2 + n_observers, rounds=rounds,
                us=us, spread=spread, ns_per_hop=us * 1000.0 / (2 * rounds)))
    return rows


def render(rows) -> str:
    hdr = (f"{'op':>13} {'addrs':>7} {'stride':>7} {'share':>7} {'k_tiles':>8} "
           f"{'us':>9} {'Gop/s':>8} {'ns/op':>8} {'GB/s':>8} {'noise':>6}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['op']:>13} {r['n_addr']:>7} {r['stride_bytes']:>7} "
                   f"{r['share']:>7.1f} {r['k_tile_count']:>8} {r['us']:>9.1f} "
                   f"{r['gops']:>8.3f} {r['ns_per_op']:>8.2f} {r['gbs']:>8.1f} "
                   f"{100*r['spread']:>5.1f}%")
    return "\n".join(out)


def render_A5(rows) -> str:
    hdr = (f"{'advance':>18} {'observers':>10} {'CTAs':>6} {'rounds':>7} "
           f"{'us':>9} {'ns/hop':>8} {'noise':>6}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['advance']:>18} {r['n_observers']:>10} "
                   f"{r['n_ctas']:>6} {r['rounds']:>7} {r['us']:>9.1f} "
                   f"{r['ns_per_hop']:>8.1f} {100*r['spread']:>5.1f}%")
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

    unit = harness.load(_SRC, verbose=a.verbose)
    print(f"[unit] {unit.name}  flags=0x{unit.flags:x}  "
          f"{unit.n_cfg} configs  opt={unit.opt_fields}", flush=True)

    buf = torch.zeros(BUF_B // 4, dtype=torch.int32, device="cuda")
    sink = torch.zeros(N_SM * N_THREADS, dtype=torch.int32, device="cuda")
    ctr = torch.zeros(1, dtype=torch.int32, device="cuda")
    dbg = torch.zeros(280 * 2, dtype=torch.int64, device="cuda")

    results = {"unit": unit.name, "flags": unit.flags, "sweeps": {}}
    for key in [s.strip().upper() for s in a.sweeps.split(",") if s.strip()]:
        if key == "A5":
            print(f"\n[sweep A5] {sweep_A5.__doc__.splitlines()[0]}", flush=True)
            rows = sweep_A5(unit, ctr, dbg, a.rounds)
            print(render_A5(rows), flush=True)
        elif key in SWEEPS:
            print(f"\n[sweep {key}] {SWEEPS[key].__doc__.splitlines()[0]}",
                  flush=True)
            rows = SWEEPS[key](unit, buf, sink, ctr, dbg)
            print(render(rows), flush=True)
        else:
            raise SystemExit(f"unknown sweep {key!r}")
        results["sweeps"][key] = rows

    results["timer"] = harness.TIMER_USED
    if harness.TIMER_USED != "cupti":
        print(f"\n[timer] {harness.TIMER_USED} -- includes launch overhead; "
              f"not comparable with constants recorded under CUPTI", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))
        print(f"\n[json] {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
