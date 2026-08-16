"""End-to-end wall clock of the full Pi0 pipeline: vision + encoder + decoder.

Builds one engine per configuration on a shared synthetic checkpoint, captures
each into its own CUDA graph, and times `forward()` with CUDA events. This is
the headline number -- it includes the input copies and the graph launch, not
just kernel time, so it is roughly 1% above the sum of kernel self-times.

`--prompt-len` changes the workload, not just the measurement: prompt=0 gives
encoder M=768 and decoder keys=819, the shape every kernel was tuned at;
prompt=256 gives M=1024 and keys=1075, which is about a third more encoder work.
Numbers from the two are not comparable.

Timing runs at `--steps 10`. Numerical parity is read separately and at steps=1
(`eval.correctness.pi0.fused_vs_unfused`).
"""
from __future__ import annotations

import argparse
import statistics

import torch

from tilelang_infer import Pi0Inference, random_checkpoint
from .metrics import diff_stats, require_cuda

CONFIGS = {"fused": True, "unfused": False}


def time_forward(engine, images, state, noise, reps: int, warmup: int = 3) -> tuple[float, float]:
    """Median and min ms per `forward()` (one graph replay), CUDA-event timed."""
    for _ in range(warmup):
        engine.forward(images, state, noise)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        engine.forward(images, state, noise)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def run(num_views: int = 3, chunk_size: int = 50, prompt_len: int = 0, steps: int = 10,
        layers: int = 18, reps: int = 30, configs: tuple[str, ...] = tuple(CONFIGS),
        seed: int = 0) -> dict:
    """Time each configuration on identical inputs; returns {name: {median_ms, min_ms, out}}."""
    require_cuda()
    torch.cuda.init()
    device = "cuda"

    checkpoint = random_checkpoint(num_views=num_views, chunk_size=chunk_size,
                                   prompt_len=prompt_len, seed=seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    images = torch.randn(num_views, 224, 224, 3, generator=generator, device=device,
                         dtype=torch.bfloat16)
    state = torch.randn(32, generator=generator, device=device, dtype=torch.bfloat16)
    noise = torch.randn(chunk_size, 32, generator=generator, device=device, dtype=torch.bfloat16)

    enc_len = num_views * 256 + prompt_len
    print(f"# Pi0 full-graph wall clock on {torch.cuda.get_device_name(0)}")
    print(f"# num_views={num_views} chunk={chunk_size} steps={steps} layers={layers} reps={reps}")
    print(f"# prompt_len={prompt_len} -> encoder M={enc_len}, decoder keys={enc_len + chunk_size + 1}")

    results = {}
    for name in configs:
        engine = Pi0Inference(checkpoint, num_views=num_views, chunk_size=chunk_size,
                              steps=steps, layers=layers, fused=CONFIGS[name])
        out = engine.forward(images, state, noise).clone()
        median, fastest = time_forward(engine, images, state, noise, reps)
        results[name] = {"median_ms": median, "min_ms": fastest, "out": out}
        del engine

    print(f"\n{'config':10} {'median ms':>10} {'min ms':>10} {'vs unfused':>11}")
    base = results.get("unfused", {}).get("median_ms")
    for name in configs:
        r = results[name]
        speedup = f"{base / r['median_ms']:.3f}x" if base else "-"
        print(f"{name:10} {r['median_ms']:10.3f} {r['min_ms']:10.3f} {speedup:>11}")

    if len(results) > 1 and "unfused" in results and "fused" in results:
        print(f"\nfused vs unfused output: "
              f"{diff_stats(results['unfused']['out'], results['fused']['out'])}")
        if steps > 1:
            print("NOTE: output deviation at steps>1 is diffusion-loop chaos on random weights, "
                  "not a correctness signal -- run the Pi0 correctness gate at steps=1.")
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--num-views", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--prompt-len", type=int, default=0,
                   help="0 = the tuned shape (encoder M=768, decoder keys=819)")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--layers", type=int, default=18)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    a = p.parse_args(argv)
    run(num_views=a.num_views, chunk_size=a.chunk_size, prompt_len=a.prompt_len, steps=a.steps,
        layers=a.layers, reps=a.reps, configs=tuple(a.configs), seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
