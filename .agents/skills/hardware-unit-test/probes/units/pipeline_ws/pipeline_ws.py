"""E5b -- what does the producer->consumer DEPENDENCY cost, on top of contention?

`overlap.eff.sm` (E5) measured pure CONTENTION: TMA and wgmma in one CTA with no
barrier between them cost 1.25x on the copy side and 1.05x on the math side.
That was deliberately half the question. A real mainloop also makes the consumer
wait for a stage to land and the producer wait for a stage to be released, and
this probe adds that round trip back.

The pipeline is CUTLASS's own, not a hand-rolled one, so the number describes
the pattern production code ships:

  cutlass::PipelineTmaAsync<Stages>   include/cutlass/pipeline/sm90_pipeline.hpp
  media/docs/cpp/pipeline.md          the API contract and its worked example
  sm90_mma_tma_gmma_ss_warpspecialized.hpp   the mainloop the kernel mirrors

Three modes in one kernel, same SMs, same descriptors:

  coupled        the real warp-specialized pipeline
  producer_only  no consumer; the producer self-drains its own full barrier
  consumer_only  no producer; the consumer re-reads a resident stage, no waits

The pipelining cost is `coupled` against those two, and the part attributable to
the DEPENDENCY rather than to contention is that slowdown divided by E5's.

Run:
    python3 pipeline_ws.py --json /tmp/pipeline_ws.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hut import abi, harness, regime  # noqa: E402

_SRC = Path(__file__).resolve().with_suffix(".cu")

N_SM = 132

MODE = {"coupled": 0, "producer_only": 1, "consumer_only": 2}

# The residency thresholds come from hut/regime.py rather than a local copy --
# three units had their own and they are one machine property, not three.
L2_BYTES = regime.L2_BYTES
COLD_MIN_L2_RATIO = regime.COLD_MIN_L2_RATIO

# overlap.eff.sm's two slowdowns, which this unit divides out to separate
# barrier cost from the engine contention overlap already measured. Copied
# rather than imported: they are a RECORDED CONSTANT, so a drifting import
# would silently redefine what this unit's ratio means.
E5_TMA_SLOW, E5_MMA_SLOW = 1.25, 1.05

# Kept at the pre-migration count so the medians stay comparable. [rule 14]
REPS = 7


class Probe:
    """The unit's binding layer over the hut ABI.

    Method names are kept from the pre-migration probe so the sweeps below did
    not move with the ABI; the (N, BK, stages) table is still read FROM the
    library rather than duplicated here.
    """

    def __init__(self, verbose: bool = False):
        self.unit = harness.load(_SRC, verbose=verbose)
        self.n_cfg = self.unit.lib.hut_cfg_count()

    def cfg(self, i):
        g = self.unit.lib.hut_cfg
        p = abi.HutParams(cfg=i)
        return dict(n=g(i, 0), bk=g(i, 1), stages=g(i, 2),
                    smem=self.unit.lib.hut_smem(abi.ctypes.byref(p)))

    def launch(self, cfg, a, b, *, n_ctas, k_tiles, mode, bufs):
        cp, cc, sink, dbg = bufs
        p = abi.HutParams(cfg=cfg, mode=mode, n_ctas=n_ctas,
                          k_tile_count=k_tiles,
                          operand_a=a.data_ptr(), operand_b=b.data_ptr())
        self.unit.launch(p, abi.buffers(cycles_a=cp, cycles_b=cc,
                                        sink=sink, dbg=dbg))


def run_mode(probe, ci, a, b, *, n_ctas, k_tiles, mode, bufs) -> dict:
    cp, cc, sink, dbg = bufs
    p_s, c_s = [], []
    for _ in range(REPS):
        cp.zero_(); cc.zero_(); dbg.zero_()
        probe.launch(ci, a, b, n_ctas=n_ctas, k_tiles=k_tiles,
                     mode=MODE[mode], bufs=bufs)
        torch.cuda.synchronize()
        if int(dbg.max().item()) != 0:
            sites = {11: "producer_acquire (empty barrier)",
                     12: "producer self-drain (full barrier)",
                     13: "producer drain tail",
                     14: "producer_tail (empty barrier)",
                     21: "consumer_wait (full barrier)"}
            hit = dbg.view(-1, 2)
            first = int(hit[hit[:, 0] != 0][0, 0].item())
            raise RuntimeError(
                f"watchdog fired in {mode}: site {first} = "
                f"{sites.get(first, '?')} -- an arrival count or a phase is "
                f"wrong, not a slow measurement")
        p, c = cp[:n_ctas].float(), cc[:n_ctas].float()
        p_s.append(float(p.median()) if p.max() > 0 else 0.0)
        c_s.append(float(c.median()) if c.max() > 0 else 0.0)
    p_s.sort(); c_s.sort()
    return dict(prod=p_s[REPS // 2], cons=c_s[REPS // 2],
                spread=max((p_s[-1] - p_s[0]) / max(p_s[REPS // 2], 1.0),
                           (c_s[-1] - c_s[0]) / max(c_s[REPS // 2], 1.0)))


def sweep(probe, a_bufs, *, n_ctas, k_tiles, bufs):
    rows = []
    for ci in range(probe.n_cfg):
        c = probe.cfg(ci)
        if c["smem"] > 232448:
            rows.append({**c, "skipped": f"smem {c['smem']}"})
            continue
        a, b = a_bufs[c["n"]]
        kw = dict(n_ctas=n_ctas, k_tiles=k_tiles, bufs=bufs)
        iso_p = run_mode(probe, ci, a, b, mode="producer_only", **kw)
        iso_c = run_mode(probe, ci, a, b, mode="consumer_only", **kw)
        both = run_mode(probe, ci, a, b, mode="coupled", **kw)
        p_slow = both["prod"] / max(iso_p["prod"], 1.0)
        c_slow = both["cons"] / max(iso_c["cons"], 1.0)
        rows.append({
            **c, "k_tiles": k_tiles,
            "prod_iso": iso_p["prod"], "prod_both": both["prod"],
            "cons_iso": iso_c["cons"], "cons_both": both["cons"],
            "prod_slow": p_slow, "cons_slow": c_slow,
            # Coupled, the two sides span ONE window (measured spans agree to
            # 0.03%), so "producer slowdown" and "consumer slowdown" are the
            # same number over two different baselines. The quantity is coupled
            # over the SLOWER side; dividing that by E5's contention leaves what
            # the dependency itself costs.
            "vs_slower": both["cons"] / max(iso_p["prod"], iso_c["cons"], 1.0),
            "dep": (both["cons"] / max(iso_p["prod"], iso_c["cons"], 1.0))
                   / E5_TMA_SLOW,
            "cons_duty": iso_c["cons"] / max(both["cons"], 1.0),
            "spread": max(iso_p["spread"], iso_c["spread"], both["spread"])})
    return rows


def render(rows) -> str:
    hdr = (f"{'N':>4} {'BK':>3} {'S':>2} {'smemKB':>7} | "
           f"{'prod_iso':>9} {'cons_iso':>9} {'coupled':>9} | "
           f"{'vs_slower':>10} {'dep':>5} {'duty':>6} {'noise':>6}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        if "skipped" in r:
            out.append(f"{r['n']:>4} {r['bk']:>3} {r['stages']:>2}   "
                       f"-- {r['skipped']}")
            continue
        out.append(
            f"{r['n']:>4} {r['bk']:>3} {r['stages']:>2} {r['smem']/1024:>7.0f} | "
            f"{r['prod_iso']:>9.0f} {r['cons_iso']:>9.0f} "
            f"{r['cons_both']:>9.0f} | {r['vs_slower']:>10.2f} "
            f"{r['dep']:>5.2f} {100*r['cons_duty']:>5.0f}% "
            f"{100*r['spread']:>5.1f}%")
    out.append("")
    out.append("  vs_slower = coupled / max(prod_iso, cons_iso). Coupled, the "
               "two sides span ONE window,")
    out.append("              so this -- not two separate slowdowns -- is the "
               "quantity.")
    out.append(f"  dep       = vs_slower / {E5_TMA_SLOW} (overlap.eff.sm's "
               "contention). ~1.00 means the barrier is free.")
    out.append("  duty      = cons_iso / coupled: how much of the coupled "
               "window the tensor core is fed.")
    out.append("  A producer that is DRAM-bandwidth-bound reads a spurious "
               "speedup here -- cycles are the")
    out.append("  wrong unit for a bandwidth-bound side (protocol rule 10). "
               "The N=256 row is such a case.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctas", type=int, default=N_SM)
    ap.add_argument("--k-tiles", type=int, default=12288,
                    help="k-tiles walked AND allocated. Sets the footprint: "
                         "k_tiles x 2 x (64+N) x BK bytes. Must clear L2 by "
                         "COLD_MIN_L2_RATIO or the pipeline is measured "
                         "cache-resident.")
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}",
          flush=True)

    probe = Probe(a.verbose)
    # k_tiles TILES per operand, not one. The footprint the producer walks is
    # k_tiles x (A_tile + B_tile), and it has to clear L2 or this measures a
    # cache-resident pipeline -- which is exactly what the first version did.
    a_bufs = {}
    for ci in range(probe.n_cfg):
        c = probe.cfg(ci)
        if c["n"] in a_bufs:
            continue
        a_bufs[c["n"]] = (
            torch.randn(a.k_tiles * 64 * c["bk"], dtype=torch.bfloat16,
                        device="cuda"),
            torch.randn(a.k_tiles * c["n"] * c["bk"], dtype=torch.bfloat16,
                        device="cuda"))
        per_k = 2 * (64 + c["n"]) * c["bk"]
        fp = a.k_tiles * per_k
        print(f"  N={c['n']:>3}: footprint {fp/1e6:>6.1f} MB "
              f"= {fp/L2_BYTES:>4.1f}x L2"
              + ("" if fp >= COLD_MIN_L2_RATIO * L2_BYTES
                 else "   <- NOT COLD, raise --k-tiles"), flush=True)
    n = a.ctas
    bufs = (torch.zeros(n, dtype=torch.int64, device="cuda"),
            torch.zeros(n, dtype=torch.int64, device="cuda"),
            torch.zeros(n, dtype=torch.float32, device="cuda"),
            torch.zeros(n * 2, dtype=torch.int64, device="cuda"))

    print("\n[E5b] CUTLASS PipelineTmaAsync, warp-specialized. The producer is\n"
          "      warpgroup 0 (one lane issues TMA), the consumer is warpgroup 1.",
          flush=True)
    rows = sweep(probe, a_bufs, n_ctas=a.ctas, k_tiles=a.k_tiles, bufs=bufs)
    print(render(rows), flush=True)

    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(ctas=a.ctas, k_tiles=a.k_tiles, rows=rows), indent=2))
        print(f"\n[json] {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
