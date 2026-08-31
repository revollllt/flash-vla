"""E5 -- do the copy engine and the tensor core actually run at once?

`wgmma.bytes.wg.tma` says a CTA needs ~4 producer warps per math warpgroup. It
is arithmetic over two constants measured in SEPARATE kernels, and SKILL.md
names it the biggest untested thing here: every fused kernel in this repo rests
on the assumption that TMA and wgmma both reach their isolated rates while
running together, and until now nothing had run them together.

The probe puts a wgmma consumer warpgroup and N TMA producer warps in one CTA,
DELIBERATELY UNCOUPLED -- no barrier joins them and the consumer never reads
what the producer lands. That separates the two questions the ratio conflates:

  CONTENTION  do the two engines slow each other down when both are busy?
  PIPELINING  what does the producer->consumer dependency cost on top?

Contention is the one the ratio assumes away, so it is measured first. A
coupled variant measures pipelining and is a different experiment.

Each configuration is run three ways in the same job, on the same SMs, with the
same descriptors and the same shared-memory offsets:

  tma_only     producers run, consumer warps skip their loop
  wgmma_only   consumer runs, producer warps skip theirs
  both         concurrently

Each side reports its OWN cycle span, so a shortfall is attributable rather
than aggregate. Perfect concurrency is slowdown 1.00 on both sides.

Run:
    python3 overlap.py --json /tmp/overlap.json
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hut import abi, harness, tma  # noqa: E402

_SRC = Path(__file__).resolve().with_suffix(".cu")

N_SM = 132
MODE = {"tma_only": 1, "wgmma_only": 2, "both": 0}

# The walk's addressable range, matching the tma unit's. Local rather than
# imported: this unit no longer reaches into another unit's module for its
# descriptor machinery -- that lives in hut/tma.py, and the descriptor itself is
# encoded through THIS unit's own hut_encode_tensor_map.
BUF_BYTES = 256 * 1024 * 1024


class Probe:
    """The unit's binding layer over the hut ABI."""

    def __init__(self, verbose: bool = False):
        self.unit = harness.load(_SRC, verbose=verbose)
        self.n_cfg = self.unit.lib.hut_cfg_count()

    def cfg(self, i):
        g = self.unit.lib.hut_cfg
        return dict(n=g(i, 0), ngroup=g(i, 1), wait=g(i, 2))

    def _params(self, *, cfg, n_ctas, num_producers, stages, box_bytes,
                k_tiles_tma, k_tiles_mma, mode, walk):
        p = abi.HutParams(
            cfg=cfg, mode=mode, n_ctas=n_ctas,
            n_threads=0,                      # the kernel fixes its own width
            num_producers=num_producers, stages=stages, box_bytes=box_bytes,
            k_tile_count=k_tiles_tma,
            mask0=walk["mask0"], shift0=walk["shift0"], step0=walk["step0"],
            mask1=walk["mask1"], step1=walk["step1"])
        p.opt[0] = k_tiles_mma
        return p

    def smem(self, num_producers, stages, box_bytes, n, cfg=0):
        p = self._params(cfg=cfg, n_ctas=1, num_producers=num_producers,
                         stages=stages, box_bytes=box_bytes, k_tiles_tma=1,
                         k_tiles_mma=1, mode=0,
                         walk=dict(mask0=0, shift0=0, step0=0, mask1=0, step1=0))
        return self.unit.lib.hut_smem(abi.ctypes.byref(p))

    def encode(self, ptr, plan):
        mapbuf, rc = tma.encode(self.unit, ptr, plan)
        if rc != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled rc={rc}")
        return mapbuf

    def launch(self, cfg, mapbuf, *, n_ctas, num_producers, stages, box_bytes,
               k_tiles_tma, k_tiles_mma, mode, walk, cyc_tma, cyc_mma, sink,
               dbg):
        p = self._params(cfg=cfg, n_ctas=n_ctas, num_producers=num_producers,
                         stages=stages, box_bytes=box_bytes,
                         k_tiles_tma=k_tiles_tma, k_tiles_mma=k_tiles_mma,
                         mode=mode, walk=walk)
        p.tensor_map = mapbuf[1]
        self.unit.launch(p, abi.buffers(cycles_a=cyc_tma, cycles_b=cyc_mma,
                                        sink=sink, dbg=dbg))


# One launch per mode is not a measurement. The first run of this probe read
# tma_iso 11% apart across configs whose TMA work was IDENTICAL, and produced a
# 0.85 "speedup" from adding work -- both artefacts of a single sample against a
# 6% noise floor. Every mode is now REPS launches and the median of the medians.
REPS = 7


def run_mode(probe, cfg, mapbuf, *, n_ctas, num_producers, stages, box_bytes,
             k_tiles_tma, k_tiles_mma, mode, walk, bufs) -> dict:
    cyc_tma, cyc_mma, sink, dbg = bufs
    t_s, m_s = [], []
    for _ in range(REPS):
        cyc_tma.zero_(); cyc_mma.zero_(); dbg.zero_()
        probe.launch(cfg, mapbuf, n_ctas=n_ctas, num_producers=num_producers, stages=stages,
                     box_bytes=box_bytes, k_tiles_tma=k_tiles_tma, k_tiles_mma=k_tiles_mma,
                     mode=MODE[mode], walk=walk, cyc_tma=cyc_tma,
                     cyc_mma=cyc_mma, sink=sink, dbg=dbg)
        torch.cuda.synchronize()
        if int(dbg.max().item()) != 0:
            raise RuntimeError(f"watchdog fired in {mode}: "
                               f"{dbg[dbg != 0][:4].tolist()}")
        t, m = cyc_tma[:n_ctas].float(), cyc_mma[:n_ctas].float()
        t_s.append(float(t.median()) if t.max() > 0 else 0.0)
        m_s.append(float(m.median()) if m.max() > 0 else 0.0)
    t_s.sort(); m_s.sort()
    return dict(mode=mode, cyc_tma=t_s[REPS // 2], cyc_mma=m_s[REPS // 2],
                tma_spread=(t_s[-1] - t_s[0]) / max(t_s[REPS // 2], 1.0),
                mma_spread=(m_s[-1] - m_s[0]) / max(m_s[REPS // 2], 1.0))


def sweep(probe, buf, dbg_bufs, *, n_ctas, stages, box_dim_1, verbose=False):
    """One row per (config, num_producers): both sides, isolated and together."""
    geom = tma.geoms()["stride8k"]
    box_bytes = geom.box_bytes(box_dim_1)
    plan = geom.plan(box_dim_1, BUF_BYTES)
    mapbuf = probe.encode(buf.data_ptr(), plan)
    walk = {k: plan[k] for k in ("mask0", "shift0", "step0", "mask1", "step1")}

    rows = []
    for ci in range(probe.n_cfg):
        c = probe.cfg(ci)
        for num_producers in (1, 2, 4):
            sm = probe.smem(num_producers, stages, box_bytes, c["n"])
            if sm > 232448:
                rows.append(dict(**c, num_producers=num_producers, box_bytes=box_bytes,
                                 skipped=f"smem {sm} > 232448"))
                continue
            k_tiles_tma = 256
            kw = dict(n_ctas=n_ctas, num_producers=num_producers, stages=stages,
                      box_bytes=box_bytes, walk=walk, bufs=dbg_bufs)
            # Calibrate so both sides occupy the same window in `both`: a side
            # that finishes early would run the rest alone and dilute exactly
            # the contention this probe exists to see.
            iso_t = run_mode(probe, ci, mapbuf, k_tiles_tma=k_tiles_tma,
                             k_tiles_mma=1, mode="tma_only", **kw)
            probe_mma = run_mode(probe, ci, mapbuf, k_tiles_tma=1,
                                 k_tiles_mma=1024, mode="wgmma_only", **kw)
            per_iter = probe_mma["cyc_mma"] / 1024.0
            k_tiles_mma = max(16, int(round(iso_t["cyc_tma"] / max(per_iter, 1.0))))
            iso_m = run_mode(probe, ci, mapbuf, k_tiles_tma=1, k_tiles_mma=k_tiles_mma,
                             mode="wgmma_only", **kw)
            both = run_mode(probe, ci, mapbuf, k_tiles_tma=k_tiles_tma,
                            k_tiles_mma=k_tiles_mma, mode="both", **kw)
            rows.append(dict(
                **c, num_producers=num_producers, box_bytes=box_bytes, k_tiles_tma=k_tiles_tma,
                k_tiles_mma=k_tiles_mma,
                tma_iso=iso_t["cyc_tma"], tma_both=both["cyc_tma"],
                mma_iso=iso_m["cyc_mma"], mma_both=both["cyc_mma"],
                tma_slow=both["cyc_tma"] / max(iso_t["cyc_tma"], 1.0),
                mma_slow=both["cyc_mma"] / max(iso_m["cyc_mma"], 1.0),
                spread=max(iso_t["tma_spread"], iso_m["mma_spread"],
                           both["tma_spread"], both["mma_spread"])))
    return rows


def render(rows) -> str:
    hdr = (f"{'N':>4} {'grp':>3} {'wait':>4} {'prod':>4} {'box':>7} "
           f"{'tma_iso':>9} {'tma_both':>9} {'x':>6} | "
           f"{'mma_iso':>9} {'mma_both':>9} {'x':>6} {'noise':>6}  verdict")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        if "skipped" in r:
            out.append(f"{r['n']:>4} {r['ngroup']:>3} {r['wait']:>4} "
                       f"{r['num_producers']:>4} {r['box_bytes']:>7}   -- {r['skipped']}")
            continue
        worst = max(r["tma_slow"], r["mma_slow"])
        # A slowdown inside the run-to-run spread is not a finding (rule 7).
        thresh = max(1.06, 1.0 + r["spread"])
        verdict = ("concurrent" if worst < thresh else
                   "CONTENTION" if worst < 1.9 else "SERIALIZED")
        out.append(
            f"{r['n']:>4} {r['ngroup']:>3} {r['wait']:>4} {r['num_producers']:>4} "
            f"{r['box_bytes']:>7} {r['tma_iso']:>9.0f} {r['tma_both']:>9.0f} "
            f"{r['tma_slow']:>6.2f} | {r['mma_iso']:>9.0f} "
            f"{r['mma_both']:>9.0f} {r['mma_slow']:>6.2f} "
            f"{100*r['spread']:>5.1f}%  {verdict}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctas", default="8,132",
                    help="grids to sweep. 8 is where TMA is issue-limited, "
                         "132 where it is DRAM-limited; the ratio's arithmetic "
                         "lives at the first")
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--box-dim-1", type=int, default=64,
                    help="box rows; box_bytes = box_dim_1 x 128 B for stride8k")
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    dev = torch.cuda.current_device()
    print(f"[env] {torch.cuda.get_device_name(dev)}  torch {torch.__version__}",
          flush=True)

    probe = Probe(a.verbose)
    buf = torch.empty(BUF_BYTES // 2, dtype=torch.bfloat16, device="cuda")
    buf.normal_()
    grids = [int(x) for x in str(a.ctas).split(",") if x]
    n = max(grids)
    bufs = (torch.zeros(n, dtype=torch.int64, device="cuda"),
            torch.zeros(n, dtype=torch.int64, device="cuda"),
            torch.zeros(n, dtype=torch.float32, device="cuda"),
            torch.zeros(n * 2, dtype=torch.int64, device="cuda"))

    print("\n[E5] TMA and wgmma in one CTA, uncoupled. Slowdown 1.00 = the two "
          "engines do not\n     interfere; the ratio wgmma.bytes.wg.tma "
          "assumes exactly that and has never\n     been checked.", flush=True)
    rows = []
    for g in grids:
        print(f"\n  --- {g} CTAs "
              f"({'TMA issue-limited' if g <= 32 else 'TMA DRAM-limited'}) ---",
              flush=True)
        r = sweep(probe, buf, bufs, n_ctas=g, stages=a.stages,
                  box_dim_1=a.box_dim_1, verbose=a.verbose)
        for row in r:
            row["n_ctas"] = g
        rows += r
        print(render(r), flush=True)

    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(ctas=grids, stages=a.stages, box_dim_1=a.box_dim_1,
                 rows=rows), indent=2))
        print(f"\n[json] {a.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
