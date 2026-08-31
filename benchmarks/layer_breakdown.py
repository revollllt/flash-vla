"""Per-call-site latency for one transformer layer, from a graph-replay trace.

`profile_pi05` aggregates by KERNEL NAME, and both stages reuse one kernel at
two call sites, so their largest rows arrive undifferentiated:

    prefix   `_matmul_res_kernel`        34 calls = 17 o_proj + 17 ffn_down
    decoder  `_matmul_gated_res_kernel`  360 calls = 180 o_proj + 180 ffn_down

Between them that is 27.6% of the prefix stage and 29.4% of the decoder, with
no attribution. This recovers it from a Chrome trace by walking the kernels in
launch order and matching them against the sequence the pipeline emits.

**How the mapping is established, and what it is not.** Under CUDA-graph replay
the CPU side is one `cudaGraphLaunch`, so the trace carries no CPU/GPU
correlation and no Python stack: a source location cannot be read out of it, and
this script never guesses one from a kernel name. What it uses instead is that
`pipeline.prefix` and `pipeline.decoder` emit a FIXED sequence, which the graph
replays in that order. The expected kernel name is asserted at every position,
so a pipeline edit that moves a call site fails the run rather than silently
relabelling a row. For the two ambiguous kernels the result is cross-checked a
second, independent way: the two call sites differ 8x in K, so their durations
must separate into two clusters that agree with the positional assignment.

Per the profiler skill's reporting contract these are DIAGNOSTIC timings from an
instrumented replay, not a latency baseline; `e2e-pi05` owns that.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

#: The kernel a call site emits, in the order `pipeline.prefix` calls them.
#: The final layer runs only its QKV projection: nothing downstream reads that
#: layer's output, only its K and V.
PREFIX_PROLOGUE = [("encoder_projector", "tl_layer_norm_kernel"),
                   ("encoder_projector", "_matmul_bias_kernel"),
                   ("encoder_embed_prompt", "vectorized_gather_kernel"),
                   ("encoder_embed_prompt", "elementwise_kernel")]
PREFIX_LAYER = [("qkv:rms_norm", "tl_rms_norm_kernel"),
                ("qkv:gemm", "_matmul_kernel"),
                ("qkv:rope_scatter", "tl_rope_scatter_bf16_kernel"),
                ("attn:qk", "nvjet"),
                ("attn:softmax", "triton_per_fused"),
                ("attn:pv", "nvjet"),
                ("o_proj", "_matmul_res_kernel"),
                ("ffn:rms_norm", "tl_rms_norm_kernel"),
                ("ffn:gate_up", "tl_matmul_gate_kernel"),
                ("ffn:down", "_matmul_res_kernel")]
PREFIX_TAIL = PREFIX_LAYER[:3]          # the final layer's QKV-only pass

#: `pipeline.decoder`, one flow step: an in-projection, 18 layers, an out-projection.
#: The per-layer body depends on the call-site plan the engine was built with
#: (`--plan`); each variant below is the exact launch sequence its wrappers
#: emit. The cuda attention pair keeps the TileLang rms_factor node; the cuda
#: FFN pair replaces {rms_factor, gate_up, down} with {counter reset, K-major
#: XFS producer, one persistent kernel} (`backends/cuda/wrappers.py`).
DECODER_PROLOGUE = [("action_in_proj", "_matmul_bias_kernel")]
_QKV_TILELANG = [("qkv:rms_factor", "tl_rms_factor_kernel"),
                 ("qkv:gemm_rope", "tl_ada_qkv_gemm_rope_kernel")]
_QKV_CUDA = [("qkv:rms_factor", "tl_rms_factor_kernel"),
             ("qkv:gemm_rope", "attn_standalone_kernel")]
_ATTN_TILELANG = [("attn:split", "tl_fd_flat_split_mask_kernel"),
                  ("attn:combine", "tl_fd_flat_combine_kernel")]
_ATTN_CUDA = [("attn:split", "attn_standalone_kernel"),
              ("attn:combine", "combine_rows_kernel")]
_OPROJ = [("o_proj", "_matmul_gated_res_kernel")]
_FFN_TILELANG = [("ffn:rms_factor", "tl_rms_factor_kernel"),
                 ("ffn:gate_up", "tl_ada_scaled_gate_kernel"),
                 ("ffn:down", "_matmul_gated_res_kernel")]
_FFN_CUDA = [("ffn:reset", "reset_ffn_counters_kernel"),
             ("ffn:xfs", "tl_rms_xfs_kmajor"),
             ("ffn:persistent", "ffn_taskloop_kernel")]
DECODER_LAYER_BY_PLAN = {
    "tilelang": _QKV_TILELANG + _ATTN_TILELANG + _OPROJ + _FFN_TILELANG,
    "attn-cuda": _QKV_CUDA + _ATTN_CUDA + _OPROJ + _FFN_TILELANG,
    "ffn-cuda": _QKV_TILELANG + _ATTN_TILELANG + _OPROJ + _FFN_CUDA,
    "attn-ffn-cuda": _QKV_CUDA + _ATTN_CUDA + _OPROJ + _FFN_CUDA,
}
DECODER_EPILOGUE = [("action_out_proj", "tl_fused_rms_matmul_bias_res_kernel")]

#: (prologue, per-layer body, tail, layers, repeats). The prefix runs all
#: TileLang under every plan; only the decoder body is plan-selected.
PLANS = {
    "prefix": (PREFIX_PROLOGUE, PREFIX_LAYER, PREFIX_TAIL, 17, 1),
    "decoder": (DECODER_PROLOGUE, DECODER_LAYER_BY_PLAN, DECODER_EPILOGUE, 18, 10),
}


def _load(path: Path) -> list[dict]:
    """Kernel events from a Chrome trace, in launch order."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    kernels = [e for e in events
               if e.get("ph") == "X"
               and e.get("cat", "").lower() in ("kernel", "gpu_memcpy", "gpu_memset")]
    return sorted(kernels, key=lambda e: e["ts"])


def _expand(stage_spec, plan_name: str) -> list[tuple[str, str]]:
    prologue, layer, tail, layers, repeats = stage_spec
    if isinstance(layer, dict):
        layer = layer[plan_name]
    return (list(prologue) + list(layer) * layers + list(tail)) * repeats


def walk(events: list[dict], stage: str, plan_name: str) -> dict[str, list[float]]:
    """Assign every kernel to a call site, asserting the expected name at each step."""
    expected = _expand(PLANS[stage], plan_name)
    if len(events) != len(expected):
        raise SystemExit(
            f"{stage}: trace has {len(events)} kernels, the pipeline emits "
            f"{len(expected)}. The sequence in this file is stale -- re-derive it "
            "from pipeline.py rather than adjusting the counts.")
    per_site: dict[str, list[float]] = defaultdict(list)
    for index, (event, (site, name)) in enumerate(zip(events, expected)):
        if name not in event["name"]:
            raise SystemExit(
                f"{stage}: position {index} expected {name!r} for call site {site!r}, "
                f"trace has {event['name']!r}. Do not relabel -- fix the sequence.")
        per_site[site].append(event["dur"])
    return per_site


def cross_check(per_site: dict[str, list[float]]) -> str:
    """The two call sites sharing a kernel name must separate on duration too."""
    o, f = per_site.get("o_proj"), per_site.get("ffn:down")
    if not o or not f:
        return "unavailable"
    ratio = statistics.median(f) / statistics.median(o)
    if max(o) < min(f):
        return f"disjoint, ffn:down/o_proj = {ratio:.1f}x"
    return (f"OVERLAPPING (o_proj max {max(o):.1f} >= ffn:down min {min(f):.1f}) -- "
            "positional assignment unconfirmed")


#: Roofline peaks, same constants as `e2e_pi05.FLOOR_MS` and
#: `hardware/nvidia/h100/spec.py`: H100 SXM5, 989 TFLOP/s dense bf16,
#: 3.35 TB/s HBM3. Ridge = 295 FLOP/byte.
PEAK_FLOPS = 989e12
PEAK_BW = 3.35e12
RIDGE = PEAK_FLOPS / PEAK_BW


def models(plan_name: str) -> dict[str, tuple[float, float]]:
    """Per-call (FLOP, HBM bytes) for every decoder call site, logical M=50.

    The byte model is minimal traffic -- each input read once, each output
    written once, residual read+written -- the same convention as
    `roofline_pi05`, so a low MBU means bandwidth left on the table, not model
    slack. FlashDecoding adds the per-split Q re-read and partial O/LSE
    round-trip explicitly; the split count is 6 for the TileLang kernel and 8
    for the cuda one. The persistent FFN row carries the whole GatedUp +
    DownResidual chain: packed gate/up (2*D*FF) + Wd (FF*D) weights, the K-major
    XFS read, the 64-row hidden write, and the residual read+write.
    """
    M, D, FF, H, DH, KEYS, QKV_W, BF = 50, 1024, 4096, 8, 256, 1018, 2560, 2
    MF = M * H
    split = 8 if "attn" in plan_name else 6

    def attn_pair(s):
        return {
            "attn:split": (4 * MF * KEYS * DH,
                           (s * MF * DH + 2 * KEYS * DH + s * MF * DH) * BF + s * MF * 4),
            "attn:combine": (3 * s * MF * DH,
                             (s * MF * DH + MF * DH) * BF + s * MF * 4),
        }

    return {
        "qkv:rms_factor": (0, M * D * BF),
        "ffn:rms_factor": (0, M * D * BF),
        "qkv:gemm_rope": (2 * M * QKV_W * D, (M * D + D * QKV_W + M * QKV_W) * BF),
        **attn_pair(split),
        "o_proj": (2 * M * (H * DH) * D, (M * H * DH + H * DH * D + 2 * M * D) * BF),
        "ffn:gate_up": (4 * M * FF * D, (M * D + 2 * D * FF + M * FF) * BF),
        "ffn:down": (2 * M * FF * D, (M * FF + FF * D + 2 * M * D) * BF),
        "ffn:reset": (0, 64 * 4),
        "ffn:xfs": (0, (M * D + D + D * 64) * BF),
        "ffn:persistent": (6 * M * FF * D,
                           (D * 64 + 2 * D * FF + 2 * FF + FF * D + D
                            + 64 * FF + 2 * 64 * D) * BF),
        "action_in_proj": (2 * M * D * 32, (M * 32 + 32 * D + M * D) * BF),
        "action_out_proj": (2 * M * 32 * D, (M * D + D * 32 + M * 32) * BF),
    }


def report(stage: str, per_site: dict[str, list[float]], plan_name: str) -> dict:
    total = sum(sum(v) for v in per_site.values())
    site_models = models(plan_name)
    rows = []
    for site, v in per_site.items():
        row = {"call_site": site, "calls": len(v), "total_us": round(sum(v), 1),
               "per_call_us": round(statistics.median(v), 2),
               "share": round(sum(v) / total, 4)}
        model = site_models.get(site)
        if model:
            flop, bytes_ = model
            seconds = row["per_call_us"] * 1e-6
            row.update({
                "gflop": round(flop / 1e9, 3), "mb": round(bytes_ / 1e6, 2),
                "flop_per_byte": round(flop / bytes_, 1),
                "tflops": round(flop / seconds / 1e12, 1) if seconds else 0.0,
                "tb_s": round(bytes_ / seconds / 1e12, 2) if seconds else 0.0,
                "mfu": round(flop / seconds / PEAK_FLOPS, 4) if seconds else 0.0,
                "mbu": round(bytes_ / seconds / PEAK_BW, 4) if seconds else 0.0,
                "bound": "compute" if flop / bytes_ > RIDGE else "memory",
            })
        rows.append(row)
    rows.sort(key=lambda r: -r["total_us"])

    launches = sum(r["calls"] for r in rows)
    print(f"\n=== {stage} ({plan_name})  {total / 1000:.3f} ms "
          f"across {launches} launches ===")
    print(f"  {'call site':<16}{'calls':>6}{'us':>8}{'GFLOP':>8}{'MB':>8}"
          f"{'FLOP/B':>8}{'TFLOP/s':>9}{'TB/s':>7}{'MFU':>7}{'MBU':>7}  bound")
    modeled_flop = modeled_bytes = modeled_us = 0.0
    for r in rows:
        if "gflop" in r:
            modeled_flop += r["calls"] * r["gflop"] * 1e9
            modeled_bytes += r["calls"] * r["mb"] * 1e6
            modeled_us += r["total_us"]
            print(f"  {r['call_site']:<16}{r['calls']:>6}{r['per_call_us']:>8.2f}"
                  f"{r['gflop']:>8.3f}{r['mb']:>8.2f}{r['flop_per_byte']:>8.1f}"
                  f"{r['tflops']:>9.1f}{r['tb_s']:>7.2f}{r['mfu'] * 100:>6.1f}%"
                  f"{r['mbu'] * 100:>6.1f}%  {r['bound']}")
        else:
            print(f"  {r['call_site']:<16}{r['calls']:>6}{r['per_call_us']:>8.2f}"
                  f"{'':>8}{'':>8}{'':>8}{'':>9}{'':>7}{'':>7}{'':>7}  "
                  f"({r['share'] * 100:.1f}% of stage, no model)")
    if modeled_us:
        seconds = modeled_us * 1e-6
        intensity = modeled_flop / modeled_bytes
        print(f"  {'stage':<16}{launches:>6}{modeled_us / 1000:>7.3f}m"
              f"{modeled_flop / 1e12:>7.2f}T{modeled_bytes / 1e9:>7.2f}G"
              f"{intensity:>8.1f}{modeled_flop / seconds / 1e12:>9.1f}"
              f"{modeled_bytes / seconds / 1e12:>7.2f}"
              f"{modeled_flop / seconds / PEAK_FLOPS * 100:>6.1f}%"
              f"{modeled_bytes / seconds / PEAK_BW * 100:>6.1f}%  "
              f"{'compute' if intensity > RIDGE else 'memory'}")
    checked = cross_check(per_site)
    print(f"  shared-name cross-check: {checked}")
    return {"stage": stage, "plan": plan_name, "total_us": round(total, 1),
            "call_sites": rows, "cross_check": checked}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path,
                        help="directory holding the per-stage traces written by "
                             "profile-pi05 --trace-dir")
    parser.add_argument("--stages", default="prefix,decoder")
    parser.add_argument("--plan", default="tilelang", choices=sorted(DECODER_LAYER_BY_PLAN),
                        help="the call-site plan the traced engine was built with")
    parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    args = parser.parse_args(argv)

    out = []
    for stage in args.stages.split(","):
        candidates = sorted(args.trace_dir.glob(f"{stage}*.json*"))
        if not candidates:
            raise SystemExit(f"no trace for {stage!r} in {args.trace_dir}")
        out.append(report(stage, walk(_load(candidates[0]), stage, args.plan), args.plan))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
