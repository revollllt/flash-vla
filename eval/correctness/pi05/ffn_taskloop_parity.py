"""Parity, replay gate, acceptance bench, and counter probe for ffn_taskloop.

Phase 2 gate of specs/tile/ffn_taskloop.md. Three separately runnable pieces,
each printed behind a [phase] marker BEFORE its device sync, because a
persistent-kernel bug hangs rather than fails and the log must show which
phase died:

- parity (default): the fused kernel against a torch recomputation mirroring
  the kernels' precision exactly (bf16 (x*F)*S in the mainloop, f32
  accumulator/bias/gelu/gate/residual, bf16 stores) -- the same recipe as
  kernel_parity's check_gated_ffn, with F an input rather than computed.
  `--modes gu,dr,full` is the bisection: gu runs only the GU task list and
  checks hidden; dr pre-fills hidden + counters and checks out; full runs the
  counter protocol end to end.
- --replay-check: execute the full fixed-132-CTA table repeatedly with fresh
  residual/counter state, proving that counter reset is replay-safe.
- --bench: the acceptance measurement named in the spec -- one CUDA graph
  cycling 3 distinct weight sets (75 MB > L2, so每 launch cold), fused single
  launch vs the tl_ada_scaled_gate + tl_matmul_gated_res composition, same
  process, median over replays.
- --probe: Phase 0 measurement for the [I] counter figures -- release->acquire
  RTT over 40 concurrent CTA pairs via %%globaltimer.

Correctness and performance are separate gates (repo rule): --bench refuses to
run unless parity passed in the same invocation.
"""
from __future__ import annotations

import argparse
import json
import statistics

import torch

from eval.correctness.pi05.prefix_parity import error_metrics
from flash_vla.hardware.nvidia.h100.pi05.backends.cuda.taskloop import (
    COUNTER_ARRIVE,
    FFNTaskloop,
    N_COUNTERS,
    WATCHDOG_SITES,
    build_table,
)
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels.base import (
    GELU_C0,
    GELU_C1,
)

CHUNK, M_PAD, D, FF = 50, 64, 1024, 4096
TOLERANCE = 0.999


def _rand(gen, device, *shape):
    return (torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
            * 0.05).bfloat16()


def make_inputs(gen, device, ablate=frozenset()):
    """`ablate` bisects the GU-only paths without a kernel rebuild: f1/s1 make
    the in-loop scale an exact identity (a mis-ADDRESSED scale becomes
    harmless, so a pass isolates the smem addressing); b0 zeroes the biases
    (isolates the epilogue column mapping); w2w1 aliases W2 := W1 (isolates the
    second gemm batch of the pair)."""
    x50 = _rand(gen, device, CHUNK, D)
    f50 = torch.rsqrt(
        x50.float().square().mean(dim=1) + 1e-6).bfloat16()
    s = (1.0 + torch.randn((D,), generator=gen, device=device) * 0.1).bfloat16()
    w1, w2 = _rand(gen, device, D, FF), _rand(gen, device, D, FF)
    b1, b2 = _rand(gen, device, FF), _rand(gen, device, FF)
    if "f1" in ablate:
        f50 = torch.ones_like(f50)
    if "s1" in ablate:
        s = torch.ones_like(s)
    if "b0" in ablate:
        b1, b2 = torch.zeros_like(b1), torch.zeros_like(b2)
    if "w2w1" in ablate:
        w2 = w1

    def _block(w):
        # (K, N) -> (N/32, K, 32) flattened: one task tile's weights contiguous.
        k, n = w.shape
        return w.reshape(k, n // 32, 32).permute(1, 0, 2).contiguous().view(-1, 32)
    w1b_raw, w2b = _block(w1), _block(w2)
    # Offline gate/up interleave: one row is [W1_tile(32), W2_tile(32)],
    # matching the kernel's single 128B TMA box and SW128 shared layout.
    w1b = torch.cat((w1b_raw, w2b), dim=1).contiguous()
    wd = _rand(gen, device, FF, D)
    g = _rand(gen, device, D)
    xres = _rand(gen, device, CHUNK, D)

    x_pad = torch.zeros((M_PAD, D), dtype=torch.bfloat16, device=device)
    x_pad[:CHUNK] = x50
    f_pad = torch.zeros((M_PAD,), dtype=torch.bfloat16, device=device)
    f_pad[:CHUNK] = f50
    hidden = torch.zeros((M_PAD, FF), dtype=torch.bfloat16, device=device)
    out = torch.zeros((M_PAD, D), dtype=torch.bfloat16, device=device)
    out[:CHUNK] = xres
    counters = torch.zeros((N_COUNTERS,), dtype=torch.int32, device=device)
    return dict(x50=x50, f50=f50, s=s, w1=w1, w2=w2, b1=b1, b2=b2, wd=wd, g=g,
                w1b=w1b, w2b=w2b, wdb=_block(wd),
                xres=xres, x_pad=x_pad, f_pad=f_pad,
                hidden=hidden, out=out,
                counters=counters)


def torch_ref(t):
    """Mirror of the kernels' rounding (spec: rounding_contract)."""
    a = (t["x50"] * t["f50"][:, None]) * t["s"][None, :]        # bf16 chain
    c1 = a.float() @ t["w1"].float() + t["b1"].float()[None, :]
    c2 = a.float() @ t["w2"].float() + t["b2"].float()[None, :]
    gelu = c1 * torch.sigmoid(GELU_C0 * c1 * (1.0 + GELU_C1 * c1 * c1))
    hidden = (gelu * c2).bfloat16()
    out = (t["xres"].float()
           + (hidden.float() @ t["wd"].float()) * t["g"].float()[None, :]).bfloat16()
    return hidden, out


def run_parity(kt, mode, gen, device, ablate=frozenset()):
    t = make_inputs(gen, device, ablate)
    hidden_ref, out_ref = torch_ref(t)
    table = build_table(mode).to(device)
    zero = True
    if mode == "dr":
        t["hidden"][:CHUNK] = hidden_ref
        t["counters"].fill_(COUNTER_ARRIVE)
        zero = False
    dbg = torch.zeros((table.shape[0], 4), dtype=torch.int64, pin_memory=True)
    tag = mode + (f"[{'+'.join(sorted(ablate))}]" if ablate else "")
    print(f"[phase] launch mode={tag} grid={table.shape[0]}", flush=True)
    kt.launch(table, t["x_pad"], t["f_pad"], t["s"], t["w1b"], t["w2b"],
              t["b1"], t["b2"], t["wdb"], t["g"], t["hidden"], t["out"],
              t["counters"], dbg=dbg, zero_counters=zero)
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        stuck = dbg[dbg[:, 3] == 1]
        for row in stuck.tolist():
            print(f"[watchdog] cta stuck at "
                  f"{WATCHDOG_SITES.get(row[0], row[0])} g={row[1]} "
                  f"tid={row[2]}", flush=True)
        raise
    print(f"[phase] mode={mode} synced", flush=True)

    report = {}
    if mode in ("gu", "full"):
        report["hidden"] = error_metrics(hidden_ref, t["hidden"][:CHUNK])
        # error-pattern localization: column halves split task#1 vs task#2 of
        # each GU CTA (ring-continuity bugs light up the second half); row
        # groups split the wgmma fragment's r / r+8 halves (layout bugs)
        h = t["hidden"][:CHUNK]
        report["hidden_task1"] = error_metrics(hidden_ref[:, :FF // 2], h[:, :FF // 2])
        report["hidden_task2"] = error_metrics(hidden_ref[:, FF // 2:], h[:, FF // 2:])
        report["hidden_rows_lo"] = error_metrics(hidden_ref[:8], h[:8])
        report["hidden_rows_hi"] = error_metrics(hidden_ref[8:16], h[8:16])
    if mode in ("dr", "full"):
        report["out"] = error_metrics(out_ref, t["out"][:CHUNK])
    for name, m in report.items():
        print(f"[gate] {tag}/{name} cosine={m['cosine_similarity']:.7f} "
              f"max_abs={m['max_abs']:.3e}", flush=True)
    return report


def run_replay_check(kt, gen, device, reps):
    """Replay the full 132-CTA schedule with fresh residual/counter state."""
    t = make_inputs(gen, device)
    hidden_ref, out_ref = torch_ref(t)
    table = build_table("full").to(device)
    out_seed = t["out"].clone()
    dbg = torch.zeros((table.shape[0], 4), dtype=torch.int64, pin_memory=True)
    worst = 1.0
    print(f"[phase] replay-check grid={table.shape[0]} reps={reps}", flush=True)
    for rep in range(reps):
        t["hidden"].zero_()
        t["out"].copy_(out_seed)
        t["counters"].zero_()
        kt.launch(table, t["x_pad"], t["f_pad"], t["s"], t["w1b"], t["w2b"],
                  t["b1"], t["b2"], t["wdb"], t["g"], t["hidden"], t["out"],
                  t["counters"], dbg=dbg)
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            stuck = dbg[dbg[:, 3] == 1]
            for row in stuck.tolist():
                print(f"[watchdog] replay={rep} cta stuck at "
                      f"{WATCHDOG_SITES.get(row[0], row[0])} g={row[1]} "
                      f"tid={row[2]}", flush=True)
            raise
        hm = error_metrics(hidden_ref, t["hidden"][:CHUNK])
        om = error_metrics(out_ref, t["out"][:CHUNK])
        worst = min(worst, hm["cosine_similarity"], om["cosine_similarity"])
        print(f"[gate] replay={rep} hidden_cos={hm['cosine_similarity']:.7f} "
              f"out_cos={om['cosine_similarity']:.7f}", flush=True)
    return {"reps": reps, "worst_cosine": worst}


def run_bench(kt, gen, device, reps):
    """Acceptance: fused single launch vs the 2-kernel TileLang composition."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels import (
        adarms,
    )

    sets = [make_inputs(gen, device) for _ in range(3)]  # 3 x 25 MB > 50 MB L2
    shared = sets[0]
    table = build_table("full").to(device)

    print("[phase] bench warmup fused", flush=True)
    for t in sets:
        kt.launch(table, t["x_pad"], t["f_pad"], t["s"], t["w1b"], t["w2b"],
                  t["b1"], t["b2"], t["wdb"], t["g"], shared["hidden"],
                  shared["out"], shared["counters"])
    torch.cuda.synchronize()

    g_fused = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_fused):
        for t in sets:
            kt.launch(table, t["x_pad"], t["f_pad"], t["s"], t["w1b"], t["w2b"],
                      t["b1"], t["b2"], t["wdb"], t["g"], shared["hidden"],
                      shared["out"], shared["counters"])

    print("[phase] bench warmup composition (TileLang compile)", flush=True)
    gate = wrappers._compiled(adarms.tl_ada_scaled_gate, M=CHUNK, N=FF, K=D,
                              **wrappers._DEC_GATE)
    down = wrappers._compiled(adarms.tl_matmul_gated_res, M=CHUNK, N=D, K=FF,
                              **wrappers._DEC_RESIDUAL)
    hid50 = shared["hidden"][:CHUNK]
    out50 = shared["out"][:CHUNK]
    for t in sets:
        gate(t["x50"], t["f50"], t["s"], t["w1"], t["w2"], t["b1"], t["b2"], hid50)
        down(hid50, t["wd"], t["g"], out50, out50)
    torch.cuda.synchronize()

    g_base = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_base):
        for t in sets:
            gate(t["x50"], t["f50"], t["s"], t["w1"], t["w2"], t["b1"], t["b2"], hid50)
            down(hid50, t["wd"], t["g"], out50, out50)

    # split timings localize where the wall time sits: gu-only and dr-only
    # (dr with counters pre-filled -- perf only, hidden contents irrelevant)
    table_gu = build_table("gu").to(device)
    table_dr = build_table("dr").to(device)
    for t in sets:
        shared["counters"].fill_(COUNTER_ARRIVE)
        kt.launch(table_dr, t["x_pad"], t["f_pad"], t["s"], t["w1b"], t["w2b"],
                  t["b1"], t["b2"], t["wdb"], t["g"], shared["hidden"],
                  shared["out"], shared["counters"], zero_counters=False)
    torch.cuda.synchronize()

    def capture(table, prefill):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for t in sets:
                if prefill:
                    shared["counters"].fill_(COUNTER_ARRIVE)
                kt.launch(table, t["x_pad"], t["f_pad"], t["s"], t["w1b"],
                          t["w2b"], t["b1"], t["b2"], t["wdb"], t["g"],
                          shared["hidden"], shared["out"], shared["counters"],
                          zero_counters=not prefill)
        return g

    g_gu = capture(table_gu, prefill=False)
    g_dr = capture(table_dr, prefill=True)

    def time_graph(g):
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); g.replay(); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000.0 / 3.0)  # us per launch-set
        return times

    print("[phase] bench timing", flush=True)
    fused = time_graph(g_fused)
    base = time_graph(g_base)
    gu_only = time_graph(g_gu)
    dr_only = time_graph(g_dr)
    r = {
        "fused_us_median": statistics.median(fused),
        "fused_us_min": min(fused),
        "fused_gu_only_us_median": statistics.median(gu_only),
        "fused_dr_only_us_median": statistics.median(dr_only),
        "composition_us_median": statistics.median(base),
        "composition_us_min": min(base),
        "reps": reps,
        "method": "CUDA-graph x3 weight sets (75 MB, cold), events, per-set us",
    }
    print(f"[bench] fused    {r['fused_us_median']:8.2f} us median "
          f"({r['fused_us_min']:.2f} min)", flush=True)
    print(f"[bench]   gu-only {r['fused_gu_only_us_median']:7.2f} us, "
          f"dr-only {r['fused_dr_only_us_median']:.2f} us", flush=True)
    print(f"[bench] compose  {r['composition_us_median']:8.2f} us median "
          f"({r['composition_us_min']:.2f} min)", flush=True)
    print(f"[bench] speedup  {r['composition_us_median']/r['fused_us_median']:.2f}x "
          f"(spec floor 11.3 us, target 15 us)", flush=True)
    return r


def run_gu_bench(kt, gen, device, reps):
    """Cold-weight CUDA-graph timing for only the GatedUp task list."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels import (
        adarms,
    )

    sets = [make_inputs(gen, device) for _ in range(4)]
    shared = sets[0]
    table = build_table("gu").to(device)
    for t in sets:
        kt.launch(table, t["x_pad"], t["f_pad"], t["s"], t["w1b"], t["w2b"],
                  t["b1"], t["b2"], t["wdb"], t["g"], shared["hidden"],
                  shared["out"], shared["counters"])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for t in sets:
            kt.launch(table, t["x_pad"], t["f_pad"], t["s"], t["w1b"], t["w2b"],
                      t["b1"], t["b2"], t["wdb"], t["g"], shared["hidden"],
                      shared["out"], shared["counters"])

    gate = wrappers._compiled(adarms.tl_ada_scaled_gate, M=CHUNK, N=FF, K=D,
                              **wrappers._DEC_GATE)
    hidden50 = shared["hidden"][:CHUNK]
    for t in sets:
        gate(t["x50"], t["f50"], t["s"], t["w1"], t["w2"], t["b1"], t["b2"],
             hidden50)
    torch.cuda.synchronize()
    tilelang_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(tilelang_graph):
        for t in sets:
            gate(t["x50"], t["f50"], t["s"], t["w1"], t["w2"], t["b1"], t["b2"],
                 hidden50)
    def time_graph(target):
        samples = []
        for _ in range(5):
            target.replay()
        torch.cuda.synchronize()
        for _ in range(reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); target.replay(); end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / len(sets))
        return samples

    samples = time_graph(graph)
    tilelang_samples = time_graph(tilelang_graph)
    result = {
        "gated_up_us_median": statistics.median(samples),
        "gated_up_us_min": min(samples),
        "tilelang_gate_us_median": statistics.median(tilelang_samples),
        "tilelang_gate_us_min": min(tilelang_samples),
        "reps": reps,
        "method": "CUDA-graph x4 cold-weight sets, events, per launch",
    }
    print(f"[bench-gu] {result['gated_up_us_median']:.2f} us median "
          f"({result['gated_up_us_min']:.2f} min)", flush=True)
    print(f"[bench-gu] TileLang {result['tilelang_gate_us_median']:.2f} us median "
          f"({result['tilelang_gate_us_min']:.2f} min)", flush=True)
    return result


def run_probe(kt, device, iters):
    pairs = 40
    counters = torch.zeros((pairs,), dtype=torch.int32, device=device)
    t0s = torch.zeros((pairs,), dtype=torch.int64, device=device)
    out_ns = torch.zeros((pairs,), dtype=torch.int64, device=device)
    samples = []
    print(f"[phase] probe {iters} iters x {pairs} pairs", flush=True)
    for _ in range(iters):
        counters.zero_()
        kt.probe(counters, t0s, out_ns, pairs)
        torch.cuda.synchronize()
        samples.extend(out_ns.tolist())
    samples.sort()
    r = {"pairs": pairs, "iters": iters,
         "rtt_ns_median": samples[len(samples) // 2],
         "rtt_ns_p95": samples[int(len(samples) * 0.95)],
         "rtt_ns_max": samples[-1]}
    print(f"[probe] counter RTT median {r['rtt_ns_median']} ns, "
          f"p95 {r['rtt_ns_p95']} ns, max {r['rtt_ns_max']} ns "
          f"(40 concurrent pairs)", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="gu,dr,full")
    ap.add_argument("--gu-ablations", default="",
                    help="comma list of +-joined ablation sets run as extra gu "
                         "modes, e.g. 'f1+s1,b0,w2w1' -- diagnostics only, "
                         "excluded from the gate")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-gu-only", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--probe-iters", type=int, default=200)
    ap.add_argument("--replay-check", type=int, default=3,
                    help="full-schedule counter/graph replay repetitions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print("[phase] building kernel (nvcc)", flush=True)
    kt = FFNTaskloop(verbose=True)

    report = {}
    worst = 1.0
    for mode in [m for m in args.modes.split(",") if m]:
        r = run_parity(kt, mode, gen, device)
        report[mode] = r
        # only the full-tensor comparisons gate; *_task/_rows are diagnostics
        worst = min([worst] + [m["cosine_similarity"]
                               for name, m in r.items() if "_" not in name])
    for ab in [a for a in args.gu_ablations.split(",") if a]:
        ablate = frozenset(ab.split("+"))
        report[f"gu[{ab}]"] = run_parity(kt, "gu", gen, device, ablate)
    report["worst_cosine"] = worst
    if args.replay_check > 0:
        report["replay"] = run_replay_check(kt, gen, device, args.replay_check)
        worst = min(worst, report["replay"]["worst_cosine"])
        report["worst_cosine"] = worst
    report["passed"] = bool(worst > args.tolerance)
    print(f"[gate] worst cosine {worst:.7f} -> "
          f"{'PASS' if report['passed'] else 'FAIL'}", flush=True)

    if args.probe:
        report["probe"] = run_probe(kt, device, args.probe_iters)
    if args.bench:
        if not report["passed"]:
            print("[bench] SKIPPED: parity failed, perf numbers would be noise",
                  flush=True)
        else:
            report["bench"] = run_bench(kt, gen, device, args.reps)
    if args.bench_gu_only:
        if not report["passed"]:
            print("[bench-gu] SKIPPED: parity failed", flush=True)
        else:
            report["bench_gu"] = run_gu_bench(kt, gen, device, args.reps)

    print(json.dumps(report, indent=2, default=float))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
