"""Per-kernel breakdown and wave-quantization analysis for the Pi0.5 target.

Two views of the same pass.

**Measured.** Each of the three graphs is replayed once under the CUDA
profiler and self-device-time is summed per kernel name. Profiling outside a
graph measures launch overhead rather than kernels -- at these sizes a launch
costs more than the kernel -- so everything here is timed inside a replay, the
regime production runs in.

**Analytic.** The grid every decoder call site launches, derived from the tuned
configs in `wrappers.py` rather than restated, against the 132 SMs of an H100.
At M=50 the decoder cannot fill the machine by tiling M: one m-tile of 64 covers
the whole chunk, so the CTA count is set almost entirely by `N / BLOCK_N`. That
makes wave quantization a first-order effect here rather than a rounding
detail, and it is invisible in a kernel-time table -- a kernel using a third of
the SMs looks merely slow.

The column that matters is `sm_used`: CTAs divided by 132, capped at 1. A kernel
below 1.0 leaves that fraction of the machine idle for its whole duration, and
for a weight-bandwidth-bound kernel that is exactly the fraction of HBM
bandwidth it cannot reach.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile as torch_profile

from flash_vla.models.pi05.spec import (
    ACTION_DIM,
    DECODER_DIM,
    DECODER_FFN,
    DECODER_HEADS,
    HEAD_DIM,
    QKV_WIDTH,
)
from flash_vla.models.pi05.tokenize import Pi05Tokenizer
from flash_vla.models.pi05.weights import fold, random_checkpoint

from .metrics import require_cuda

SM_COUNT = 132          # H100 SXM5
DEFAULT_PROMPT = "pick up the plate and put it in the sink"


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def wave_report(chunk: int, keys: int) -> list[dict]:
    """Grid and SM coverage per decoder call site, derived from the live configs."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers as w

    flat = chunk * DECODER_HEADS
    num_split, _ = w._num_splits(keys, w._FD_SPLIT["BLOCK_N"], w._FD_SPLIT["NUM_SPLIT"])

    def gemm(cfg, m, n, per_layer):
        return _ceil(n, cfg["BLOCK_N"]) * _ceil(m, cfg["BLOCK_M"]), per_layer

    sites = [
        ("tl_rms_factor", *gemm({"BLOCK_N": 1, "BLOCK_M": w._DEC_RMS["BLOCK_M"]}, chunk, 1, 2)),
        ("tl_matmul_bias (action_in)", *gemm(w._DEC_ACTION_IN, chunk, DECODER_DIM, 0)),
        ("tl_ada_qkv_gemm_rope", *gemm(w._DEC_QKV, chunk, QKV_WIDTH, 1)),
        ("tl_fd_flat_split_mask", _ceil(flat, w._FD_SPLIT["BLOCK_M"]) * num_split, 1),
        ("tl_fd_flat_combine", _ceil(flat, w._FD_COMBINE_BLOCK_M), 1),
        ("tl_matmul_gated_res (o_proj)", *gemm(w._DEC_RESIDUAL, chunk, DECODER_DIM, 1)),
        ("tl_ada_scaled_gate", *gemm(w._DEC_GATE, chunk, DECODER_FFN, 1)),
        ("tl_matmul_gated_res (ffn_down)", *gemm(w._DEC_RESIDUAL, chunk, DECODER_DIM, 1)),
        ("tl_fused_rms_matmul_bias_res (action_out)",
         *gemm(w._DEC_OUT_PROJ, chunk, ACTION_DIM, 0)),
    ]

    rows = []
    for name, ctas, per_layer in sites:
        rows.append({
            "kernel": name,
            "ctas": ctas,
            "per_layer": bool(per_layer),
            "launches_per_step": per_layer * 18 if per_layer else 1,
            "sm_used": round(min(ctas / SM_COUNT, 1.0), 3),
            "waves": round(ctas / SM_COUNT, 2),
            "idle_sms": max(SM_COUNT - ctas, 0),
        })
    return rows


def _profile_graph(graph, label: str, top: int, trace_dir: str | Path | None = None) -> list[dict]:
    """Replay once under the profiler; return per-kernel self-device-time in us."""
    if trace_dir:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{re.sub(r'[^a-zA-Z0-9_.-]+', '_', label)}.json"
    else:
        trace_path = None
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    activities = [ProfilerActivity.CUDA]
    if trace_path:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with torch_profile(activities=activities) as prof:
        graph.replay()
        torch.cuda.synchronize()
    if trace_path:
        prof.export_chrome_trace(str(trace_path))

    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    events.sort(key=lambda e: -e.self_device_time_total)
    total = sum(e.self_device_time_total for e in events)
    rows = []
    for event in events[:top]:
        rows.append({
            "kernel": event.key[:60],
            "calls": event.count,
            "self_us": round(event.self_device_time_total, 1),
            "per_call_us": round(event.self_device_time_total / max(event.count, 1), 2),
            "share": round(event.self_device_time_total / total, 4),
        })
    return rows, round(total, 1)


def run(num_views: int = 3, chunk_size: int = 50, steps: int = 10, layers: int = 18,
        top: int = 14, seed: int = 0, prompt: str = DEFAULT_PROMPT,
        tokenizer_path: str | None = None, trace_dir: str | Path | None = None,
        plan: dict[str, str] | None = None) -> dict:
    """Profile each stage and print both views."""
    trace_dir = trace_dir or os.environ.get("GPU_PROFILE_OUTPUT_DIR")
    require_cuda()
    torch.cuda.init()
    device = "cuda"

    tokenizer = Pi05Tokenizer(tokenizer_path)
    checkpoint = fold(random_checkpoint(seed=seed, device=device), steps=steps)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    images = torch.randn(num_views, 224, 224, 3, generator=generator, device=device,
                         dtype=torch.bfloat16)
    state = torch.randn(32, generator=generator, device=device, dtype=torch.float32)
    noise = torch.randn(chunk_size, 32, generator=generator, device=device,
                        dtype=torch.bfloat16)

    from flash_vla.hardware.nvidia.h100.pi05 import Pi05Inference

    engine = Pi05Inference(checkpoint, tokenizer, num_views=num_views, chunk_size=chunk_size,
                           steps=steps, layers=layers, device=device, plan=plan)
    del checkpoint
    torch.cuda.empty_cache()
    engine.set_task(prompt)
    engine.forward(images, state, noise)
    torch.cuda.synchronize()

    keys = engine.encoder_seq_len + chunk_size
    report = {"device": torch.cuda.get_device_name(0), "sm_count": SM_COUNT,
              "config": {"chunk": chunk_size, "steps": steps, "layers": layers,
                         "decoder_keys": keys, "plan": engine.plan},
              "stages": {}, "waves": wave_report(chunk_size, keys),
              "trace_dir": str(trace_dir) if trace_dir else None}
    for name, graph in (("vision", engine.vision_graph), ("prefix", engine.prefix_graph),
                        ("decoder", engine.decoder_graph)):
        rows, total = _profile_graph(graph, name, top, trace_dir=trace_dir)
        report["stages"][name] = {"total_us": total, "kernels": rows}

    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--top", type=int, default=14, help="kernels listed per stage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--trace-dir", default=None,
                        help="export stage Chrome traces here; defaults to GPU_PROFILE_OUTPUT_DIR")
    parser.add_argument("--plan", default=None,
                        help="call-site plan, a name from e2e_pi05.PLANS or a JSON object")
    args = parser.parse_args(argv)
    from .e2e_pi05 import parse_plan
    run(args.num_views, args.chunk_size, args.steps, args.layers, args.top, args.seed,
        args.prompt, args.tokenizer, trace_dir=args.trace_dir,
        plan=parse_plan(args.plan) if args.plan else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
