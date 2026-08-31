"""Parity, stage bisection, replay gate and acceptance bench for the Pi0.5
attention block (specs/tile/attention_block_contract.md, sections 4-6).

Pieces, each printed behind a [phase] marker BEFORE its device sync, because a
persistent-kernel bug hangs rather than fails and the log must show which
phase died:

- parity (default, ``--modes``): the fused task loop on a truncated or full
  task table against the torch mirrors of the same ABI. ``qkv`` / ``attn`` /
  ``oproj`` run one task kind with the missing producers' outputs and counters
  pre-filled from ``attn_reference``; ``qkv_attn`` and ``full`` run the counter
  protocol end to end and are gated by ``attn_block_reference`` on the real
  rows. ``--impl tilelang`` runs the control adapter through the same gate.
- ``--replay-check N``: the full table from identical inputs, N times, must be
  bit-identical (contract 4).
- ``--bench``: contract 6 -- CUPTI over a CUDA graph, cold L2, span of every
  kernel the block issues, for the fused launch and the TileLang control in
  one process, plus a per-kernel record dump (torch.profiler, also CUPTI) and
  the event-timed rotating-set cross-check. Refuses to run unless parity
  passed in the same invocation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from eval.correctness.pi05.prefix_parity import error_metrics
from flash_vla.hardware.nvidia.h100.pi05.backends.cuda import attn_block_reference as blockref
from flash_vla.hardware.nvidia.h100.pi05.backends.cuda import attn_reference as taskref
from flash_vla.hardware.nvidia.h100.pi05.backends.cuda.attn_taskloop import (
    CM, KEYS, KEYS_PAD, M, MODES, N_CTAS, PREFIX_LEN, WATCHDOG_SITES,
    STANDALONE_DEFAULT_OPS, STANDALONE_OP_GROUPS, AttnTaskloop, Workspace, build_table, launch_standalone,
    prefill_values,
)

TOLERANCE = 0.999
H, DH, D = taskref.H, taskref.DH, taskref.D
TENSOR_ARGS = ("x", "rms_factor", "ada_scale", "w_qkv", "qkv_bias", "rope", "key_mask",
               "w_o", "ada_gate", "k_cache", "v_cache", "out", "q_buf", "o_buf")


# ---------------------------------------------------------------------------
# inputs and references
# ---------------------------------------------------------------------------
def make_inputs(seed: int, device: str, alias_out: bool = True) -> dict:
    t = blockref.make_inputs(seed=seed, device=device, alias_out=alias_out)
    # cache pad rows are never written and must be finite (contract 3.1)
    t["k_cache"][KEYS:] = 0
    t["v_cache"][KEYS:] = 0
    # experiment: a build with ATTN_QKV_WT consumes W_qkv pre-transposed
    t["w_qkv_t"] = t["w_qkv"].t().contiguous()
    return t


def _kernel_args(t: dict) -> dict:
    from flash_vla.hardware.nvidia.h100.pi05.backends.cuda.attn_taskloop import QKV_WEIGHT_TRANSPOSED
    args = {k: t[k] for k in TENSOR_ARGS}
    if QKV_WEIGHT_TRANSPOSED:
        args["w_qkv"] = t["w_qkv_t"]
    return args


def clone_inputs(t: dict) -> dict:
    """Independent copies, preserving the out-aliases-x relation."""
    c = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in t.items()
         if k != "out"}
    c["out"] = c["x"] if t["out"] is t["x"] else t["out"].clone()
    return c


def task_reference(t: dict) -> dict:
    """attn_reference.run on copies: q_buf, o_buf, caches and out, task by task."""
    c = clone_inputs(t)
    taskref.run(**{k: c[k] for k in TENSOR_ARGS})
    return c


def block_reference(t: dict) -> dict:
    c = clone_inputs(t)
    blockref.AttnBlockReference().forward(**{k: c[k] for k in TENSOR_ARGS})
    return c


def _gate(name: str, ref: torch.Tensor, got: torch.Tensor, report: dict) -> None:
    m = error_metrics(ref, got)
    if m["cosine_similarity"] != m["cosine_similarity"]:   # NaN: a failure, not a pass
        m["cosine_similarity"] = 0.0
    report[name] = m
    print(f"[gate] {name} cosine={m['cosine_similarity']:.7f} max_abs={m['max_abs']:.3e}",
          flush=True)


# ---------------------------------------------------------------------------
# implementations behind one call signature
# ---------------------------------------------------------------------------
class FusedBlock:
    """The task loop. `mode` picks the table; `q_buf`/`o_buf` are observable."""
    fills_scratch = True

    def __init__(self, kt: AttnTaskloop, ws: Workspace, mode: str, dbg=None, timeline=None):
        self.kt, self.ws, self.mode, self.dbg = kt, ws, mode, dbg
        self.timeline = timeline
        self.table = None

    def to(self, device):
        self.table = build_table(self.mode).to(device)
        self.prefill = prefill_values(self.mode).to(device)
        return self

    def __call__(self, t: dict, prefill: bool = False) -> None:
        if prefill:
            self.ws.counters.copy_(self.prefill)      # D2D: graph-capturable
        self.kt.launch(self.table, self.ws, dbg=self.dbg, timeline=self.timeline,
                       zero_counters=not prefill, **_kernel_args(t))


class StandaloneBlock:
    """The same task bodies as ordinary grid kernels: six launches in order
    (qkv split + reduce, attention split + combine, o_proj split + reduce).
    Fills `q_buf` / `o_buf` like the fused launch, so the stage diffs apply."""
    fills_scratch = True

    def __init__(self, kt: AttnTaskloop, ws: Workspace, ops=None, timelines=None):
        self.kt, self.ws = kt, ws
        self.ops = STANDALONE_DEFAULT_OPS if ops is None else tuple(ops)
        self.timelines = timelines or {}          # op -> (N_CTAS, TASK_SLOTS, 5) int64 buffer

    def to(self, device):
        return self

    def __call__(self, t: dict, prefill: bool = False) -> None:
        for op in self.ops:
            launch_standalone(self.kt._lib, op, self.ws, timeline=self.timelines.get(op),
                              **_kernel_args(t))


class TileLangControl:
    """The four-kernel TileLang composition through the contract's tensors.

    `rms_factor[:M]` is passed straight into `tl_ada_qkv_gemm_rope` (no
    `tl_rms_factor` node, contract 1). K/V are written into the cache suffix
    rows in place; Q lives in a private token-major scratch; `q_buf`/`o_buf`
    are left untouched.
    """

    def __init__(self, device):
        from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers
        from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels import adarms
        self.w = wrappers
        self.qkv = wrappers._compiled(adarms.tl_ada_qkv_gemm_rope, M=M, N=taskref.QKV_W, K=D,
                                      HEAD_DIM=DH, NUM_HEADS=H, **wrappers._DEC_QKV)
        self.q_flat = torch.empty((M * H, DH), dtype=torch.bfloat16, device=device)
        self.attn_out = torch.empty((M * H, DH), dtype=torch.bfloat16, device=device)

    def __call__(self, t: dict, prefill: bool = False) -> None:
        self.qkv(t["x"][:M], t["rms_factor"][:M], t["ada_scale"], t["w_qkv"], t["qkv_bias"],
                 t["rope"][:M], self.q_flat.view(M, H * DH),
                 t["k_cache"][PREFIX_LEN:KEYS], t["v_cache"][PREFIX_LEN:KEYS])
        self.w.decoder_attention(self.q_flat, t["k_cache"][:KEYS], t["v_cache"][:KEYS],
                                 t["key_mask"][:KEYS], self.attn_out)
        self.w.decoder_out_proj_residual(self.attn_out.view(M, H * DH), t["w_o"],
                                         t["ada_gate"], t["out"][:M])


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _sync_or_dump(dbg, tag):
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        if dbg is not None:
            for row in dbg[dbg[:, 3] == 1].tolist():
                print(f"[watchdog] {tag}: cta stuck at "
                      f"{WATCHDOG_SITES.get(row[0], row[0])} g={row[1]} tid={row[2]}", flush=True)
        raise


def run_parity(impl, mode: str, seed: int, device: str, alias_out: bool, dbg=None) -> dict:
    t = make_inputs(seed, device, alias_out)
    tref = task_reference(t)
    bref = block_reference(t)
    tag = f"{mode}[{'alias' if alias_out else 'noalias'}]"
    if mode in ("attn", "oproj"):
        # the missing producers' outputs come from the task reference
        t["q_buf"].copy_(tref["q_buf"])
        t["k_cache"].copy_(tref["k_cache"])
        t["v_cache"].copy_(tref["v_cache"])
    if mode == "oproj":
        t["o_buf"].copy_(tref["o_buf"])
    prefix_k, prefix_v = t["k_cache"][:PREFIX_LEN].clone(), t["v_cache"][:PREFIX_LEN].clone()
    pad_k, pad_v = t["k_cache"][KEYS:].clone(), t["v_cache"][KEYS:].clone()

    print(f"[phase] launch impl={type(impl).__name__} mode={tag}", flush=True)
    impl(t, prefill=mode in ("attn", "oproj"))
    _sync_or_dump(dbg, tag)
    print(f"[phase] mode={tag} synced", flush=True)

    report: dict = {}
    if mode in ("qkv", "qkv_attn", "full"):
        _gate(f"{tag}/k_suffix", tref["k_cache"][PREFIX_LEN:KEYS], t["k_cache"][PREFIX_LEN:KEYS], report)
        _gate(f"{tag}/v_suffix", tref["v_cache"][PREFIX_LEN:KEYS], t["v_cache"][PREFIX_LEN:KEYS], report)
        if getattr(impl, "fills_scratch", False):
            _gate(f"{tag}/q_buf", tref["q_buf"][:, :M], t["q_buf"][:, :M], report)
    if mode in ("attn", "qkv_attn", "full") and getattr(impl, "fills_scratch", False):
        _gate(f"{tag}/o_buf", tref["o_buf"][:, :M], t["o_buf"][:, :M], report)
    if mode in ("oproj", "full"):
        ref_out = tref["out"] if mode == "oproj" else bref["out"]
        _gate(f"{tag}/out", ref_out[:M], t["out"][:M], report)
    # invariants: prefix and pad cache rows untouched
    for name, before, after in (("k_prefix", prefix_k, t["k_cache"][:PREFIX_LEN]),
                                ("v_prefix", prefix_v, t["v_cache"][:PREFIX_LEN]),
                                ("k_pad", pad_k, t["k_cache"][KEYS:]),
                                ("v_pad", pad_v, t["v_cache"][KEYS:])):
        if not torch.equal(before, after):
            report[f"{tag}/{name}_untouched"] = {"cosine_similarity": 0.0, "max_abs": float("inf")}
            print(f"[gate] {tag}/{name} MODIFIED -- contract violation", flush=True)
    return report


def run_replay_check(impl, seed: int, device: str, reps: int, dbg=None) -> dict:
    t0 = make_inputs(seed, device, alias_out=True)
    outs = []
    print(f"[phase] replay-check reps={reps}", flush=True)
    for rep in range(reps):
        t = clone_inputs(t0)
        impl(t)
        _sync_or_dump(dbg, f"replay {rep}")
        outs.append((t["out"].clone(), t["k_cache"].clone(), t["v_cache"].clone()))
    identical = all(torch.equal(a, b) for o in outs[1:] for a, b in zip(outs[0], o))
    print(f"[gate] replay bit-identical={identical}", flush=True)
    return {"reps": reps, "bit_identical": identical}


# ---------------------------------------------------------------------------
# bench (contract 6)
# ---------------------------------------------------------------------------
def _block_flops_bytes() -> tuple[int, int]:
    from flash_vla.bench import attention_flops
    flops = (2 * M * D * taskref.QKV_W
             + attention_flops(batch_size=1, qo_seqlen=M, kv_seqlen=KEYS, head_dim_qk=DH,
                               head_dim_vo=DH, num_qo_heads=H, causal=False)
             + 2 * M * H * DH * D)
    nbytes = 2 * (D * taskref.QKV_W + H * DH * D + 2 * KEYS_PAD * DH + 4 * M * D)
    return flops, nbytes


def _kernel_records(fn, t: dict) -> dict:
    """One graph replay under torch.profiler: per-kernel durations and the gap term."""
    from torch.profiler import ProfilerActivity, profile
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(t)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn(t)
    g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type.name == "CUDA"]
    if not evs:
        return {"kernels": [], "span_us": None, "gap_us": None}
    start = min(e.time_range.start for e in evs)
    end = max(e.time_range.end for e in evs)
    kernels = [{"name": e.name[:80], "us": e.time_range.elapsed_us()} for e in evs]
    total = sum(k["us"] for k in kernels)
    return {"kernels": kernels, "span_us": end - start, "gap_us": (end - start) - total}


def _event_graph_cross_check(fn, sets: list[dict], reps: int) -> dict:
    """FFN-harness method: one graph over rotating cold sets, CUDA events."""
    for t in sets:
        fn(t)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for t in sets:
            fn(t)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0 / len(sets))
    return {"median_us": statistics.median(times), "min_us": min(times), "sets": len(sets)}


def run_timeline(kt: AttnTaskloop, ws: Workspace, seed: int, device: str, reps: int = 5) -> dict:
    """Critical path of the full table from per-task globaltimer stamps.

    For each task kind: median/max of (first frame - slot start) = dependency
    wait plus first TMA latency, (retired - first frame) = mainloop, (end -
    retired) = epilogue and split join; plus the kind's earliest start and
    latest end relative to the kernel's earliest stamp. Warm, in a loop, so the
    numbers describe the schedule rather than the first cold launch.
    """
    from flash_vla.hardware.nvidia.h100.pi05.backends.cuda.attn_taskloop import TASK_SLOTS
    t = make_inputs(seed, device, alias_out=True)
    tl_buf = torch.zeros((N_CTAS, TASK_SLOTS, 5), dtype=torch.int64, device=device)
    fused = FusedBlock(kt, ws, "full", timeline=tl_buf).to(device)
    table = fused.table.cpu()
    print(f"[phase] timeline reps={reps}", flush=True)
    rows = []
    for _ in range(reps):
        tl_buf.zero_()
        fused(clone_inputs(t))
        torch.cuda.synchronize()
        rows.append(tl_buf.cpu().clone())
    names = {0: "qkv", 1: "attn", 2: "oproj", 3: "combine"}
    out = {}
    for kind, name in names.items():
        sel = [(c, s) for c in range(N_CTAS) for s in range(TASK_SLOTS) if int(table[c, s, 0]) == kind]
        dep, main, join, epi, start, end = [], [], [], [], [], []
        for tlb in rows:
            t0 = int(tlb[..., 0][tlb[..., 0] > 0].min())
            for c, s in sel:
                st = tlb[c, s].tolist()
                dep.append((st[1] - st[0]) / 1e3); main.append((st[2] - st[1]) / 1e3)
                if int(table[c, s, 3]) == 0:               # split 0: it waited for its siblings
                    join.append((st[3] - st[2]) / 1e3)
                epi.append((st[4] - st[3]) / 1e3)
                start.append((st[0] - t0) / 1e3); end.append((st[4] - t0) / 1e3)
        med = lambda v: statistics.median(v)  # noqa: E731
        out[name] = {"n": len(sel), "dep_wait_us": (med(dep), max(dep)), "mainloop_us": (med(main), max(main)),
                     "join_wait_us": (med(join), max(join)), "epilogue_us": (med(epi), max(epi)),
                     "start_us": (min(start), med(start), max(start)), "end_us": (min(end), med(end), max(end))}
        # the tail defines the critical path: name the slowest tasks of the last rep
        tlb = rows[-1]
        t0 = int(tlb[..., 0][tlb[..., 0] > 0].min())
        per_task = sorted(((int(tlb[c, s, 4]) - t0) / 1e3, c, int(table[c, s, 1]), int(table[c, s, 3]),
                           [(int(tlb[c, s, k + 1]) - int(tlb[c, s, k])) / 1e3 for k in range(4)])
                          for c, s in sel)
        for end, c, col, split, parts in per_task[-5:]:
            print(f"[timeline]   slow {name} cta={c:3d} column={col:2d} split={split}  end {end:6.2f}"
                  f"  dep+first {parts[0]:5.2f} main {parts[1]:5.2f} join {parts[2]:5.2f} epi {parts[3]:5.2f}", flush=True)
        r = out[name]
        print(f"[timeline] {name:6s} n={r['n']:3d}  dep+first {r['dep_wait_us'][0]:5.2f}/{r['dep_wait_us'][1]:5.2f}"
              f"  mainloop {r['mainloop_us'][0]:5.2f}/{r['mainloop_us'][1]:5.2f}"
              f"  join-wait {r['join_wait_us'][0]:5.2f}/{r['join_wait_us'][1]:5.2f}"
              f"  epilogue {r['epilogue_us'][0]:5.2f}/{r['epilogue_us'][1]:5.2f}"
              f"  start {r['start_us'][0]:5.2f}..{r['start_us'][2]:5.2f}"
              f"  end {r['end_us'][0]:5.2f}..{r['end_us'][2]:5.2f} us (median/max; min..max)", flush=True)
    return out


def run_timeline_standalone(kt: AttnTaskloop, ws: Workspace, seed: int, device: str, reps: int = 5) -> dict:
    """Per-task stamps of the standalone split kernels (qkv op 0, attention
    op 2, o_proj op 4): dependency-free, so `dep+first` is pure first-frame
    latency and `mainloop` the stage cadence with nothing else on the machine."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.cuda.attn_taskloop import TASK_SLOTS
    t = make_inputs(seed, device, alias_out=True)
    kinds = {0: ("qkv", 0), 2: ("attn", 1), 4: ("oproj", 2)}
    bufs = {op: torch.zeros((N_CTAS, TASK_SLOTS, 5), dtype=torch.int64, device=device) for op in kinds}
    sa = StandaloneBlock(kt, ws, timelines=bufs)
    print(f"[phase] standalone timeline reps={reps}", flush=True)
    rows = []
    for _ in range(reps):
        for b in bufs.values():
            b.zero_()
        sa(clone_inputs(t))
        torch.cuda.synchronize()
        rows.append({op: b.cpu().clone() for op, b in bufs.items()})
    out = {}
    med = lambda v: statistics.median(v)  # noqa: E731
    for op, (name, _) in kinds.items():
        dep, main, join, epi, end = [], [], [], [], []
        for rep in rows:
            tlb = rep[op][:, 0]
            active = [c for c in range(N_CTAS) if int(tlb[c, 0]) > 0]
            t0 = min(int(tlb[c, 0]) for c in active)
            for c in active:
                st = tlb[c].tolist()
                dep.append((st[1] - st[0]) / 1e3); main.append((st[2] - st[1]) / 1e3)
                join.append((st[3] - st[2]) / 1e3); epi.append((st[4] - st[3]) / 1e3)
                end.append((st[4] - t0) / 1e3)
        out[name] = {"n": len(active), "dep_first_us": (med(dep), max(dep)), "mainloop_us": (med(main), max(main)),
                     "join_us": (med(join), max(join)), "epilogue_us": (med(epi), max(epi)),
                     "end_us": (min(end), med(end), max(end))}
        r = out[name]
        print(f"[sa-timeline] {name:6s} n={r['n']:3d}  first {r['dep_first_us'][0]:5.2f}/{r['dep_first_us'][1]:5.2f}"
              f"  mainloop {r['mainloop_us'][0]:5.2f}/{r['mainloop_us'][1]:5.2f}"
              f"  join {r['join_us'][0]:5.2f}/{r['join_us'][1]:5.2f}"
              f"  epilogue {r['epilogue_us'][0]:5.2f}/{r['epilogue_us'][1]:5.2f}"
              f"  end {r['end_us'][0]:5.2f}..{r['end_us'][2]:5.2f} us (median/max; min..max)", flush=True)
    return out


def run_stage_bench(kt: AttnTaskloop, ws: Workspace, seed: int, device: str, reps: int) -> dict:
    """Stage-only spans: one task kind on its truncated table, inputs and
    counters pre-filled (the prefill is captured in the graph and is inside
    the span; it is a handful of 64-element writes). Compared with the
    per-stage floors in the proposal, not with the composition."""
    from flash_vla.bench import KernelResult, bench_gpu_time
    t = make_inputs(seed, device, alias_out=True)
    tref = task_reference(t)
    t["q_buf"].copy_(tref["q_buf"]); t["k_cache"].copy_(tref["k_cache"])
    t["v_cache"].copy_(tref["v_cache"]); t["o_buf"].copy_(tref["o_buf"])
    floors = {"qkv": 1.98, "attn": 1.62, "oproj": 1.51}
    out = {}
    for mode in ("qkv", "attn", "oproj"):
        fn = FusedBlock(kt, ws, mode).to(device)
        stage = lambda tt, _fn=fn, _m=mode: _fn(tt, prefill=_m in ("attn", "oproj"))  # noqa: E731
        print(f"[phase] stage bench {mode}", flush=True)
        samples = bench_gpu_time(stage, input_args=(t,), enable_cupti=True, use_cuda_graph=True,
                                 cold_l2_cache=True, dry_run_iters=10, repeat_iters=reps)
        r = KernelResult(f"stage:{mode}", samples)
        out[mode] = {"median_us": r.median_ms * 1e3, "min_us": r.min_ms * 1e3,
                     "p99_us": r.p99_ms * 1e3, "n": len(samples), "floor_us": floors[mode]}
        print(f"[stage] {mode:6s} median {r.median_ms * 1e3:7.2f} us  min {r.min_ms * 1e3:7.2f}"
              f"  floor {floors[mode]:.2f} us  ({r.median_ms * 1e3 / floors[mode]:.1f}x)", flush=True)
    return out


def run_op_bench(kt: AttnTaskloop, ws: Workspace, ctl, seed: int, device: str, reps: int, rounds: int) -> dict:
    """Per-op comparison, standalone kernels vs the TileLang kernel(s) for the
    same op, each a CUPTI-over-graph span (contract 6.2) of exactly the
    launches that op needs: qkv = split + reduce vs tl_ada_qkv_gemm_rope;
    attention = split + combine vs fd_split + fd_combine; o_proj = split +
    reduce vs tl_matmul_gated_res.  Inputs are produced once by the
    standalone pipeline so every op reads a realistic operand."""
    from flash_vla.bench import KernelResult, bench_gpu_time
    t = make_inputs(seed, device, alias_out=True)
    StandaloneBlock(kt, ws)(t)                     # populates q_buf/o_buf/caches
    ctl(t)                                          # populates the control's scratch
    torch.cuda.synchronize()
    M_, H_, DH_ = M, H, DH
    ops = {
        "qkv": (lambda tt: StandaloneBlock(kt, ws, STANDALONE_OP_GROUPS["qkv"])(tt),
                lambda tt: ctl.qkv(tt["x"][:M_], tt["rms_factor"][:M_], tt["ada_scale"], tt["w_qkv"],
                                   tt["qkv_bias"], tt["rope"][:M_], ctl.q_flat.view(M_, H_ * DH_),
                                   tt["k_cache"][PREFIX_LEN:KEYS], tt["v_cache"][PREFIX_LEN:KEYS])),
        "attention": (lambda tt: StandaloneBlock(kt, ws, (2, 3))(tt),
                      lambda tt: ctl.w.decoder_attention(ctl.q_flat, tt["k_cache"][:KEYS],
                                                         tt["v_cache"][:KEYS], tt["key_mask"][:KEYS],
                                                         ctl.attn_out)),
        "oproj": (lambda tt: StandaloneBlock(kt, ws, STANDALONE_OP_GROUPS["oproj"])(tt),
                  lambda tt: ctl.w.decoder_out_proj_residual(ctl.attn_out.view(M_, H_ * DH_), tt["w_o"],
                                                             tt["ada_gate"], tt["out"][:M_])),
    }
    floors = {"qkv": 1.98, "attention": 1.62, "oproj": 1.51}
    out = {}
    for name, (fn_sa, fn_tl) in ops.items():
        samples = {"standalone": [], "tilelang": []}
        for r in range(rounds):
            for label, fn in (("standalone", fn_sa), ("tilelang", fn_tl)):
                print(f"[phase] op-bench {name} round {r} {label}", flush=True)
                samples[label].extend(bench_gpu_time(
                    fn, input_args=(t,), enable_cupti=True, use_cuda_graph=True,
                    cold_l2_cache=True, dry_run_iters=10, repeat_iters=reps))
        res = {k: KernelResult(f"{name}:{k}", v) for k, v in samples.items()}
        out[name] = {k: {"median_us": r.median_ms * 1e3, "min_us": r.min_ms * 1e3, "p99_us": r.p99_ms * 1e3,
                         "n": len(r.samples)} for k, r in res.items()}
        out[name]["floor_us"] = floors[name]
        out[name]["ratio"] = res["tilelang"].median_ms / res["standalone"].median_ms
        print(f"[op] {name:10s} standalone {out[name]['standalone']['median_us']:6.2f} us"
              f" (min {out[name]['standalone']['min_us']:6.2f})  tilelang {out[name]['tilelang']['median_us']:6.2f}"
              f" (min {out[name]['tilelang']['min_us']:6.2f})  floor {floors[name]:.2f}  ratio {out[name]['ratio']:.2f}x",
              flush=True)
    return out


def run_bench(impls: dict, seed: int, device: str, reps: int, rounds: int) -> dict:
    from flash_vla.bench import KernelResult, bench_gpu_time
    from flash_vla.bench.timer import _cupti_available

    if not _cupti_available():
        raise RuntimeError("contract 6.2: CUPTI timer required (pip install -U cupti-python)")
    flops, nbytes = _block_flops_bytes()
    t = make_inputs(seed, device, alias_out=True)
    # per-kernel records first: torch.profiler and cupti-python cannot share CUPTI
    records = {}
    for name, fn in impls.items():
        print(f"[phase] kernel records {name}", flush=True)
        records[name] = _kernel_records(fn, t)
        for k in records[name]["kernels"]:
            print(f"[records] {name:8s} {k['us']:8.2f} us  {k['name']}", flush=True)
        print(f"[records] {name:8s} span {records[name]['span_us']:.2f} us, "
              f"gap {records[name]['gap_us']:.2f} us", flush=True)

    # CUPTI over a CUDA graph, cold L2, A/B interleaved across rounds
    samples: dict[str, list[float]] = {name: [] for name in impls}
    for r in range(rounds):
        for name, fn in impls.items():
            print(f"[phase] cupti round {r} {name}", flush=True)
            samples[name].extend(bench_gpu_time(
                fn, input_args=(t,), enable_cupti=True, use_cuda_graph=True,
                cold_l2_cache=True, dry_run_iters=10, repeat_iters=reps))
    results = {name: KernelResult(name, s, flops=flops, bytes=nbytes) for name, s in samples.items()}
    for name, r in results.items():
        print(f"[bench] {r.perf_line()}", flush=True)

    # cross-check: event-timed graph over enough rotating sets to exceed L2
    l2 = torch.cuda.get_device_properties(device).L2_cache_size
    n_sets = max(2, -(-2 * l2 // nbytes))
    sets = [make_inputs(seed + i, device, alias_out=True) for i in range(n_sets)]
    cross = {}
    for name, fn in impls.items():
        print(f"[phase] event cross-check {name} sets={n_sets}", flush=True)
        cross[name] = _event_graph_cross_check(fn, sets, reps)
        print(f"[bench] {name:8s} events/graph median {cross[name]['median_us']:.2f} us "
              f"(min {cross[name]['min_us']:.2f})", flush=True)

    out = {
        "timer": "cupti+cudagraph, cold L2 (2xL2 flush), per-call span",
        "reps_per_round": reps, "rounds": rounds,
        "flops": flops, "bytes": nbytes,
        "results": {name: {"median_us": r.median_ms * 1e3, "min_us": r.min_ms * 1e3,
                           "p99_us": r.p99_ms * 1e3, "std_us": r.std_ms * 1e3,
                           "n": len(r.samples), "tb_per_sec": r.tb_per_sec}
                    for name, r in results.items()},
        "records": records, "event_cross_check": cross,
    }
    if "fused" in results and "tilelang" in results:
        out["speedup"] = results["tilelang"].median_ms / results["fused"].median_ms
        print(f"[bench] speedup {out['speedup']:.2f}x (target 10 us, floor 7.7 us)", flush=True)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="fused", choices=["fused", "tilelang", "standalone", "both", "all"])
    ap.add_argument("--op-bench", action="store_true",
                    help="standalone vs TileLang per op (needs a build and the TileLang control)")
    ap.add_argument("--modes", default="qkv,attn,oproj,qkv_attn,full")
    ap.add_argument("--alias", default="both", choices=["alias", "noalias", "both"])
    ap.add_argument("--replay-check", type=int, default=3)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--force-bench", action="store_true",
                    help="run the benches even when parity failed (ablation builds)")
    ap.add_argument("--force-timeline", action="store_true",
                    help="run the timeline even when parity failed (ablation builds)")
    ap.add_argument("--timeline", action="store_true",
                    help="fused only: per-task globaltimer stamps, critical-path summary")
    ap.add_argument("--stage-bench", action="store_true",
                    help="fused only: per-task-kind spans on truncated tables vs their floors")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    report: dict = {"env": {"torch": torch.__version__, "device": torch.cuda.get_device_name(0)}}
    worst = 1.0
    alias_modes = {"alias": [True], "noalias": [False], "both": [True, False]}[args.alias]

    impls: dict = {}
    kt = ws = None
    if args.impl in ("fused", "standalone", "both", "all") or args.op_bench:
        print("[phase] building kernel (nvcc)", flush=True)
        t0 = time.time()
        kt = AttnTaskloop(verbose=True)
        print(f"[phase] built in {time.time() - t0:.1f}s", flush=True)
        ws = Workspace(device)
    if args.impl in ("fused", "both", "all"):
        dbg = torch.zeros((N_CTAS, 4), dtype=torch.int64, pin_memory=True)
        for mode in [m for m in args.modes.split(",") if m]:
            if mode not in MODES:
                raise SystemExit(f"unknown mode {mode}")
            fused = FusedBlock(kt, ws, mode, dbg).to(device)
            for alias in alias_modes:
                r = run_parity(fused, mode, args.seed, device, alias, dbg)
                report[f"fused/{mode}/{alias}"] = r
                worst = min([worst] + [m["cosine_similarity"] for m in r.values()])
        impls["fused"] = FusedBlock(kt, ws, "full").to(device)
        if args.replay_check > 0:
            rr = run_replay_check(impls["fused"], args.seed, device, args.replay_check, dbg)
            report["fused/replay"] = rr
            if not rr["bit_identical"]:
                worst = 0.0
    if args.impl in ("standalone", "all"):
        sa = StandaloneBlock(kt, ws)
        for alias in alias_modes:
            r = run_parity(sa, "full", args.seed, device, alias)
            report[f"standalone/full/{alias}"] = r
            worst = min([worst] + [m["cosine_similarity"] for m in r.values()])
        if args.replay_check > 0:
            rr = run_replay_check(sa, args.seed, device, args.replay_check)
            report["standalone/replay"] = rr
            if not rr["bit_identical"]:
                worst = 0.0
        impls["standalone"] = sa
    ctl = None
    if args.impl in ("tilelang", "both", "all") or args.op_bench:
        print("[phase] TileLang control (compile)", flush=True)
        ctl = TileLangControl(device)
        if args.impl in ("tilelang", "both", "all"):
            for alias in alias_modes:
                r = run_parity(ctl, "full", args.seed, device, alias)
                report[f"tilelang/full/{alias}"] = r
                worst = min([worst] + [m["cosine_similarity"] for m in r.values()])
            impls["tilelang"] = ctl

    report["worst_cosine"] = worst
    report["passed"] = bool(worst > args.tolerance)
    print(f"[gate] worst cosine {worst:.7f} -> {'PASS' if report['passed'] else 'FAIL'}", flush=True)

    if args.timeline and "fused" in impls and (report["passed"] or args.force_timeline):
        report["timeline"] = run_timeline(kt, ws, args.seed, device)
    if args.timeline and "standalone" in impls and (report["passed"] or args.force_timeline):
        report["timeline_standalone"] = run_timeline_standalone(kt, ws, args.seed, device)
    if args.bench or args.stage_bench or args.op_bench:
        if not report["passed"] and not args.force_bench:
            print("[bench] SKIPPED: parity failed, perf numbers would be noise", flush=True)
        else:
            if args.stage_bench and "fused" in impls:
                report["stage_bench"] = run_stage_bench(kt, ws, args.seed, device, args.reps)
            if args.bench:
                report["bench"] = run_bench(impls, args.seed, device, args.reps, args.rounds)
            if args.op_bench:
                report["op_bench"] = run_op_bench(kt, ws, ctl, args.seed, device, args.reps, args.rounds)

    print(json.dumps(report, indent=2, default=float))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
