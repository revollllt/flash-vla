"""wgmma issue-rate probe -- the other half of an L3 timeline's ratio.

The TMA unit says a producer warp delivers one box every 248 ns. Nothing so far
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
    wait_group count. This is the math column's version of the TMA ring stages,
    and it sets how many accumulator registers a stage must hold -- an L2
    budget decision, not a codegen one.

M3  What does wgmma.wait_group cost at each count? Same sweep, read down the
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
    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/units/mma_rate/mma_rate.py
    sbatch sbatch/pi05_cuda.sh .claude/skills/hardware-unit-test/probes/units/mma_rate/mma_rate.py \
        --sweeps M1 --json profiles/hardware-unit-test/mma.json
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

M_TILE, K_TILE = 64, 16
N_SM = 132
TARGET_US = 100.0

# Modes, matching the enum in mma_rate.cu.
MODE_WGMMA, MODE_MMA_SYNC, MODE_MMA_LDM = 0, 1, 2
LDM_CFG_BASE = 100          # the cfg-space partition, see mma_rate.cu

# H100 SXM5 dense bf16 peak, 989.4 TFLOP/s at a 1755 MHz boost over 132 SMs.
# [I] datasheet-derived, and used ONLY as a yardstick for "are we at peak" --
# every constant this probe reports is a measured cycle count, which does not
# depend on it. Clocks are unpinnable here, so a FLOP/s figure would move with
# the clock and a cycle count does not.
PEAK_FLOP_PER_CYCLE_PER_SM = 989.4e12 / (1.755e9 * 132)

# The recorded constants were taken as a median of 20. Kept, and now RECORDED,
# because a median over a different sample count is a different statistic --
# comparing across counts is what rule 14 warns about. Override with --reps.
DEFAULT_REPS = 20


class Probe:
    """The unit's binding layer: hut ABI in, this unit's vocabulary out.

    Method names are kept from the pre-migration probe so the sweeps below did
    not have to change with the ABI. What changed is underneath: eight bespoke
    entry points became the nine uniform symbols, and the (N, n_groups, wait)
    and (a_tiles_m, b_tiles_n) tables are still read FROM the library rather
    than duplicated here.
    """

    def __init__(self, verbose: bool = False):
        self.unit = harness.load(_SRC, verbose=verbose)
        self.n_cfg = self.unit.lib.hut_cfg_count()
        # The ldm table has no count symbol: enumerate it until the library
        # rejects an index, the same way descriptor legality is enumerated
        # rather than assumed. [protocol.md rule 11]
        self.n_ldm = 0
        while self.unit.lib.hut_cfg(LDM_CFG_BASE + self.n_ldm, 3) >= 0:
            self.n_ldm += 1

    def _params(self, **kw):
        return abi.HutParams(**kw)

    def cfg(self, i):
        return tuple(self.unit.lib.hut_cfg(i, f) for f in (0, 1, 2))

    def ldm_cfg(self, i):
        return tuple(self.unit.lib.hut_cfg(LDM_CFG_BASE + i, f) for f in (3, 4))

    def _rate(self, mode, cfg, n_ctas, n_threads, a, b, k_tile_count,
              sink, cycles, opt=(0, 0, 0, 0)):
        p = abi.HutParams(cfg=cfg, mode=mode, n_ctas=n_ctas,
                          n_threads=n_threads, k_tile_count=k_tile_count,
                          operand_a=a.data_ptr(), operand_b=b.data_ptr())
        for i, v in enumerate(opt):
            p.opt[i] = v
        self.unit.launch(p, abi.buffers(sink=sink, cycles_a=cycles))

    def rate(self, cfg, n_ctas, n_threads, a, b, k_tile_count, sink, cycles):
        self._rate(MODE_WGMMA, cfg, n_ctas, n_threads, a, b, k_tile_count,
                   sink, cycles)

    def sync_rate(self, nacc, n_ctas, n_threads, a, b, k_tile_count, sink,
                  cycles):
        self._rate(MODE_MMA_SYNC, 0, n_ctas, n_threads, a, b, k_tile_count,
                   sink, cycles, opt=(nacc, 0, 0, 0))

    def ldm_rate(self, cfg, n_ctas, n_threads, a, b, k_tile_count, sink,
                 cycles):
        self._rate(MODE_MMA_LDM, LDM_CFG_BASE + cfg, n_ctas, n_threads, a, b,
                   k_tile_count, sink, cycles)

    def _check(self, mode, a, b, out, opt=(0, 0, 0, 0)):
        p = abi.HutParams(mode=mode, n_ctas=1, n_threads=128, k_tile_count=1,
                          operand_a=a.data_ptr(), operand_b=b.data_ptr())
        for i, v in enumerate(opt):
            p.opt[i] = v
        self.unit.check(p, abi.buffers(out=out))

    def check(self, n, a, b, out):
        self._check(MODE_WGMMA, a, b, out, opt=(0, n, 0, 0))

    def sync_check(self, a, b, out):
        self._check(MODE_MMA_SYNC, a, b, out)

    def ldm_check(self, a, b, out):
        self._check(MODE_MMA_LDM, a, b, out)


TIMER = ""


def _time_us(run, reps=None) -> float:
    global TIMER
    us, _ = harness.time_us(run, reps=reps)
    TIMER = harness.TIMER_USED
    return us


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


SYNC_SAT_CYC = 6.26      # [mma.issue.warp] register-resident, saturated
WGMMA_N32 = 2623.0       # [wgmma.issue.wg.ss] FLOP/cyc/SM at the shipped FFN tile
TOK_REAL, TOK_PAD_WG, TOK_PAD_SY = 50, 64, 56


def check_ldm(probe, dev) -> dict:
    """MS4. The ldmatrix addressing, verified rather than reasoned about."""
    torch.manual_seed(11)
    a = torch.randn(16, 16, dtype=torch.bfloat16, device=dev)
    b = torch.randn(8, 16, dtype=torch.bfloat16, device=dev)
    out = torch.zeros(16, 8, dtype=torch.float32, device=dev)
    probe.ldm_check(a.contiguous(), b.contiguous(), out)
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    err = (out - ref).abs().max().item()
    scale = max(ref.abs().max().item(), 1e-9)
    return dict(max_abs_err=err, rel=err / scale, ok=err / scale < 2e-2)


def run_ldm(probe, a, b, sink, cycles, cfg, n_ctas, n_threads) -> dict:
    am, bn = probe.ldm_cfg(cfg)
    warps = n_threads // 32
    per_iter = am * bn

    def go(t):
        return lambda: probe.ldm_rate(cfg, n_ctas, n_threads, a, b, t, sink,
                                      cycles)

    probe.ldm_rate(cfg, n_ctas, n_threads, a, b, 8, sink, cycles)
    torch.cuda.synchronize()
    us8 = _time_us(go(8), reps=5)
    k_tile_count = max(16, min(400000, int(8 * TARGET_US / max(us8, 1e-3))))
    us = _time_us(go(k_tile_count))
    torch.cuda.synchronize()

    cyc = cycles[:n_ctas].float().median().item()
    cyc_per_mma = cyc / (k_tile_count * per_iter)
    fpc = warps * 2.0 * 16 * 8 * 16 / cyc_per_mma
    useful = fpc * TOK_REAL / TOK_PAD_SY
    return dict(am=am, bn=bn, tile=f"{am*16}x{bn*8}", warps=warps,
                reuse=per_iter / float(am + bn), k_tile_count=k_tile_count, us=us,
                cyc_per_mma=cyc_per_mma, tax=cyc_per_mma / SYNC_SAT_CYC,
                flop_per_cyc_sm=fpc,
                pct_peak=100.0 * fpc / PEAK_FLOP_PER_CYCLE_PER_SM,
                useful=useful,
                vs_wgmma=useful / (WGMMA_N32 * TOK_REAL / TOK_PAD_WG),
                clock_ghz=cyc / (us * 1000.0))


def render_ldm(rows) -> str:
    hdr = (f"{'warp tile':>10} {'mma/ldm':>8} {'cyc/mma':>8} {'tax':>6} "
           f"{'FLOP/cyc/SM':>12} {'%peak':>6} {'useful':>8} {'vs wgmma':>9}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['tile']:>10} {r['reuse']:>8.2f} {r['cyc_per_mma']:>8.2f} "
                   f"{r['tax']:>5.2f}x {r['flop_per_cyc_sm']:>12.0f} "
                   f"{r['pct_peak']:>5.0f}% {r['useful']:>8.0f} "
                   f"{r['vs_wgmma']:>8.2f}x")
    return "\n".join(out)


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
    k_tile_count = max(16, min(400000, int(8 * TARGET_US / max(us8, 1e-3))))
    us = _time_us(go(k_tile_count))
    torch.cuda.synchronize()

    cyc_med = cycles[:n_ctas].float().median().item()
    inst_per_warp = k_tile_count * nacc
    cyc_per_inst = cyc_med / inst_per_warp
    flop_per_cyc_sm = warps * flop_per_inst / cyc_per_inst
    flops = n_ctas * warps * inst_per_warp * flop_per_inst
    return dict(kind="mma.sync", nacc=nacc, warps=warps, n_ctas=n_ctas,
                k_tile_count=k_tile_count, us=us, cycles=cyc_med, cyc_per_inst=cyc_per_inst,
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
    k_tile_count = max(16, min(200000, int(8 * TARGET_US / max(us8, 1e-3))))
    us = _time_us(go(k_tile_count))
    torch.cuda.synchronize()

    cyc = cycles[:n_ctas].float()
    cyc_med = cyc.median().item()
    inst_per_wg = k_tile_count * ngroup
    cyc_per_inst = cyc_med / inst_per_wg
    flops = (n_ctas * wg * inst_per_wg) * 2.0 * M_TILE * n * K_TILE
    tflops = flops / (us * 1e-6) / 1e12
    ideal = 2.0 * M_TILE * n * K_TILE / PEAK_FLOP_PER_CYCLE_PER_SM / wg
    return dict(cfg=cfg, n=n, ngroup=ngroup, wait=wait, n_ctas=n_ctas,
                warpgroups=wg, inflight=ngroup * (wait + 1), k_tile_count=k_tile_count, us=us,
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
    """M2/M3: how many in flight, and what the wait_group setting costs."""
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


def sweep_MS5(probe, a, b, sink, cycles):
    """MS5: the ldmatrix tax vs reuse -- what a real mma.sync mainloop pays."""
    return [run_ldm(probe, a, b, sink, cycles, c, N_SM, 128)
            for c in range(probe.n_ldm)]


def sweep_MS2(probe, a, b, sink, cycles):
    """MS2: mma.sync -- warps per SM, at 4 independent accumulators."""
    return [run_sync(probe, a, b, sink, cycles, 4, N_SM, nt)
            for nt in (32, 64, 128, 256)]


SWEEPS = {"M1": sweep_M1, "M2": sweep_M2, "M4": sweep_M4,
          "MS1": sweep_MS1, "MS2": sweep_MS2, "MS5": sweep_MS5}
SYNC_SWEEPS = {"MS1", "MS2"}
LDM_SWEEPS = {"MS5"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", default="M1,M2,M4,MS1,MS2,MS5")
    ap.add_argument("--json", default="")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS,
                    help="timed samples per point; the median is taken over "
                         "these and the count is recorded in the JSON. The "
                         "shipped constants were taken at 20.")
    ap.add_argument("--verbose", action="store_true")
    a_ = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    dev = "cuda"
    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}",
          flush=True)

    harness.REPS = a_.reps
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

    print("\n[MS4] correctness -- ldmatrix-fed mma.sync against torch",
          flush=True)
    cl = check_ldm(probe, dev)
    print(f"  max|err| {cl['max_abs_err']:9.4f}  rel {cl['rel']:.2e}  "
          f"{'OK' if cl['ok'] else 'FAIL'}", flush=True)
    if not cl["ok"]:
        raise SystemExit("ldmatrix addressing FAILED against torch -- the "
                         "per-lane row pointers are wrong; no tax below would "
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
        if key in LDM_SWEEPS:
            print(render_ldm(rows), flush=True)
            print(f"  tax  = cyc/mma against the {SYNC_SAT_CYC} of "
                  f"[mma.issue.warp], whose operands were already in registers",
                  flush=True)
            print(f"  useful = FLOP/cyc/SM x {TOK_REAL}/{TOK_PAD_SY}; the "
                  f"shipped wgmma m64n32k16 useful = "
                  f"{WGMMA_N32 * TOK_REAL / TOK_PAD_WG:.0f}", flush=True)
        elif key in SYNC_SWEEPS:
            print(render_sync(rows), flush=True)
        else:
            print(render(rows), flush=True)

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
    results["reps"] = harness.REPS
    if TIMER != "cupti":
        print(f"\n[timer] {TIMER} -- includes launch overhead", flush=True)
    if a_.json:
        Path(a_.json).write_text(json.dumps(results, indent=2))
        print(f"\n[json] {a_.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
