"""Per-kernel GPU-time breakdown, measured inside the CUDA graph replay.

Profiling outside the graph measures launch overhead, not kernels: at these
sizes the ~15 us launch cost swamps a 4 us kernel. This captures the pass into a
graph, profiles one replay, and sums self-device-time per kernel name, so every
row is real GPU work in the regime production runs in.

Run it on the decoder (`--stage decoder`) to see where the diffusion loop spends
its time, or on the whole pipeline (`--stage full`) for the stage split. With
`--compare` both the fused and unfused configurations are profiled and the
per-kernel difference is printed, which is how each fusion's contribution was
attributed.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile as torch_profile

from flash_vla import Pi0Inference, random_checkpoint
from flash_vla.hardware.nvidia.h100.pi0 import pipeline
from flash_vla.hardware.nvidia.h100.pi0.backends.tilelang import wrappers
from flash_vla.hardware.nvidia.h100.pi0.ops import op_table
from flash_vla.runtime.cuda import ScratchPool
from .metrics import capture, require_cuda
from .synthetic import decoder_buffers, decoder_weights, encoder_seq_len


def is_copy(name: str) -> bool:
    lowered = name.lower()
    return ("memcpy" in lowered or "memset" in lowered or "copy_" in lowered
            or ("elementwise" in lowered and "copy" in lowered))


def _profile_replay(graph, top: int, label: str, trace_dir: str | Path | None = None):
    """Replay once under the profiler; print and return per-kernel self-device-time in us."""
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
    total = sum(e.self_device_time_total for e in events)
    copies = sum(e.self_device_time_total for e in events if is_copy(e.key))

    print(f"\n===== {label} =====")
    print(f"  total GPU {total / 1000:7.3f} ms   "
          f"(compute {(total - copies) / 1000:.3f} + copy {copies / 1000:.3f})")
    for e in sorted(events, key=lambda e: -e.self_device_time_total)[:top]:
        tag = "COPY" if is_copy(e.key) else "    "
        print(f"   {tag} {e.self_device_time_total / 1000:7.3f} ms  x{e.count:4d}  {e.key[:56]}")
    return {e.key: e.self_device_time_total for e in events}, total


def _decoder_graph(fused: bool, weights, buffers, enc_len, steps, layers):
    ops = op_table(fused)
    pool = ScratchPool()

    def run():
        with wrappers.use_pool(pool):
            pipeline.transformer_decoder(ops, weights, buffers, enc_len,
                                         steps=steps, layers=layers)

    graph = capture(run)
    pool.freeze()
    return graph


def _full_graph(fused: bool, num_views, chunk_size, prompt_len, steps, layers):
    checkpoint = random_checkpoint(num_views=num_views, chunk_size=chunk_size,
                                   prompt_len=prompt_len)
    engine = Pi0Inference(checkpoint, num_views=num_views, chunk_size=chunk_size, steps=steps,
                          layers=layers, fused=fused)
    return engine.graph, engine


def run(stage: str = "decoder", fused: bool = True, compare: bool = False, num_views: int = 3,
        prompt_len: int = 0, chunk_size: int = 50, steps: int = 10, layers: int = 18,
        top: int = 16, trace_dir: str | Path | None = None) -> dict:
    """Profile one or both configurations of `stage` and print the breakdown."""
    trace_dir = trace_dir or os.environ.get("GPU_PROFILE_OUTPUT_DIR")
    require_cuda()
    torch.cuda.init()
    configs = [False, True] if compare else [fused]

    per_config = {}
    keep_alive = []
    for use_fused in configs:
        label = f"{stage} / {'fused' if use_fused else 'unfused'}, {steps}x{layers}"
        if stage == "decoder":
            graph = _decoder_graph(use_fused, decoder_weights(),
                                   decoder_buffers(num_views=num_views, prompt_len=prompt_len,
                                                   chunk_size=chunk_size),
                                   encoder_seq_len(num_views, prompt_len), steps, layers)
        else:
            graph, engine = _full_graph(use_fused, num_views, chunk_size, prompt_len, steps, layers)
            keep_alive.append(engine)
        per_config[use_fused] = _profile_replay(graph, top, label, trace_dir=trace_dir)

    if compare:
        unfused, fused_times = per_config[False][0], per_config[True][0]
        print("\n===== per-kernel difference (fused vs unfused) =====")
        print(f"  {'kernel':46} {'unfused ms':>11} {'fused ms':>10} {'delta ms':>9}")
        for name in sorted(set(unfused) | set(fused_times),
                           key=lambda n: -abs(fused_times.get(n, 0) - unfused.get(n, 0))):
            a, b = unfused.get(name, 0.0) / 1000, fused_times.get(name, 0.0) / 1000
            if abs(b - a) < 0.005:
                continue
            print(f"  {name[:46]:46} {a:11.3f} {b:10.3f} {b - a:+9.3f}")
        print(f"\n  TOTAL unfused {per_config[False][1] / 1000:.3f} ms -> "
              f"fused {per_config[True][1] / 1000:.3f} ms "
              f"({per_config[False][1] / per_config[True][1]:.3f}x)")
    return {("fused" if k else "unfused"): v[1] / 1000 for k, v in per_config.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", default="decoder", choices=["decoder", "full"])
    p.add_argument("--compare", action="store_true", help="profile fused and unfused, and diff")
    p.add_argument("--unfused", action="store_true", help="profile the unfused config only")
    p.add_argument("--num-views", type=int, default=3)
    p.add_argument("--prompt-len", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--layers", type=int, default=18)
    p.add_argument("--top", type=int, default=16)
    p.add_argument("--trace-dir", default=None,
                   help="export a Chrome trace here; defaults to GPU_PROFILE_OUTPUT_DIR")
    a = p.parse_args(argv)
    run(stage=a.stage, fused=not a.unfused, compare=a.compare, num_views=a.num_views,
        prompt_len=a.prompt_len, chunk_size=a.chunk_size, steps=a.steps, layers=a.layers, top=a.top,
        trace_dir=a.trace_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
