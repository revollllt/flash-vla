"""End-to-end wall clock of the full Pi0.5 pipeline: vision + prefix + decoder.

Two numbers matter here and they are not the same, which is why both are
reported.

`forward()` is the deliverable: images and state in, a denoised action chunk
out. It includes the input copies, the host tokenization, and three graph
launches, so it is what a caller actually waits for.

The per-stage split is the diagnostic. The three graphs are replayed
individually, back to back, and timed separately -- which is only possible
because the engine keeps them apart -- so each stage can be read against the
analytic floor in `models/pi05/PLAN.md` §1.2.

`forward()` interleaves a host section -- tokenize plus copies -- between the
vision and prefix replays, so a CUDA event spanning the whole call measures host
stalls, not GPU time: on a contended node its median balloons while the GPU work
is unchanged (watch `forward_device.stdev`). So the headline is the wall clock,
which is stable and is what a caller waits for; `forward_device.min` is the
uncontended device floor; and the per-stage graph replays, which contain no host
gap, are the clean per-stage signal. The `forward_device` median is kept only to
make its own unreliability visible.

Unlike Pi0's `e2e`, there is no fused/unfused axis: Pi0.5 v1 has exactly one
implementation per call site. Numerical parity is read separately
(`eval.correctness.pi05.kernel_parity`, then suffix parity against OpenPI).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from flash_vla.models.pi05.tokenize import Pi05Tokenizer
from flash_vla.models.pi05.weights import fold, random_checkpoint

from .metrics import require_cuda

DEFAULT_PROMPT = "pick up the plate and put it in the sink"

#: Analytic per-stage floors from PLAN.md §1.2, at 3 views / prompt 200 /
#: chunk 50 / 10 steps on an H100 SXM5 at 989 TFLOP/s bf16 and 3.35 TB/s.
FLOOR_MS = {"vision": 0.66, "prefix": 3.88, "decoder": 1.88}


def _time_event(call, reps: int, warmup: int = 3) -> dict[str, float]:
    """Median/min ms of device time for `call`, CUDA-event timed."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": round(statistics.median(samples), 3),
            "min_ms": round(min(samples), 3),
            "stdev_ms": round(statistics.stdev(samples), 4) if len(samples) > 1 else 0.0}


def _time_wall(call, reps: int, warmup: int = 3) -> dict[str, float]:
    """Median/min ms of wall clock around a synchronized call, host time included."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return {"median_ms": round(statistics.median(samples), 3),
            "min_ms": round(min(samples), 3)}


def run(num_views: int = 3, chunk_size: int = 50, steps: int = 10, layers: int = 18,
        reps: int = 30, seed: int = 0, prompt: str = DEFAULT_PROMPT,
        tokenizer_path: str | None = None) -> dict:
    """Build one engine, time the whole pass and each stage, and print a report."""
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
                           steps=steps, layers=layers, device=device)
    del checkpoint
    torch.cuda.empty_cache()
    engine.set_task(prompt)
    engine.forward(images, state, noise)          # settles n_valid before timing
    torch.cuda.synchronize()

    prompt_tokens = engine.inputs.n_valid - num_views * 256
    report = {
        "device": torch.cuda.get_device_name(0),
        "config": {"num_views": num_views, "chunk": chunk_size, "steps": steps,
                   "layers": layers, "reps": reps, "prompt_len": engine.prompt_len,
                   # Which implementation each call site resolved to. A timing
                   # report that does not say this cannot be compared to another.
                   "plan": engine.plan},
        "shapes": {"encoder_seq_len": engine.encoder_seq_len,
                   "prompt_tokens_valid": prompt_tokens,
                   "decoder_keys": engine.encoder_seq_len + chunk_size,
                   "decoder_M": chunk_size},
        "forward_device": _time_event(lambda: engine.forward(images, state, noise), reps),
        "forward_wall": _time_wall(lambda: engine.forward(images, state, noise), reps),
        "host_tokenize": _time_wall(lambda: engine.inputs.build(state), reps, warmup=50),
        "stages": {},
    }
    for name, graph in (("vision", engine.vision_graph),
                        ("prefix", engine.prefix_graph),
                        ("decoder", engine.decoder_graph)):
        stage = _time_event(graph.replay, reps)
        stage["floor_ms"] = FLOOR_MS.get(name)
        if stage["floor_ms"]:
            stage["above_floor"] = round(stage["median_ms"] / stage["floor_ms"], 2)
        report["stages"][name] = stage

    total = sum(report["stages"][s]["median_ms"] for s in report["stages"])
    report["stage_sum_ms"] = round(total, 3)
    # forward() interleaves a host section (tokenize + copies) between two graph
    # replays, so a CUDA event spanning it measures host stalls, not GPU time --
    # its median balloons on a contended node (watch forward_device.stdev). The
    # uncontended floor is forward_device.min, and the wall clock is what a
    # caller actually waits for. Derive the derived numbers from those two.
    report["launch_overhead_ms"] = round(report["forward_device"]["min_ms"] - total, 3)
    report["forward_device_median_note"] = (
        "unreliable: spans the host tokenize gap; use min_ms or forward_wall")
    floor = sum(FLOOR_MS.values())
    report["roofline_ms"] = round(floor, 2)
    report["above_roofline"] = round(report["forward_wall"]["median_ms"] / floor, 2)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokenizer", default=None,
                        help="paligemma_tokenizer.model (default: $PALIGEMMA_TOKENIZER)")
    args = parser.parse_args(argv)
    run(args.num_views, args.chunk_size, args.steps, args.layers, args.reps, args.seed,
        args.prompt, args.tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
