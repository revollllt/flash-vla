"""Screen mxfp8_backbone.cu's tile configurations on Pi0.5's backbone FFN GEMMs.

Each GEMM runs once per layer on that layer's own weights, 18 layers in one
CUDA graph, as the forward runs them: weights cold (18 x 64 MiB gate|up and
18 x 32 MiB down against a 96 MiB L2), the activation warm, one output buffer
for every layer. Reports the median time per GEMM over graph replays and each
configuration's output against configuration 0 (all compute the same product).

    python lab/pi05/mxfp8_gemm_screen.py --out results/pi05-rtx5090/<run>/gemm-screen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from flash_vla.hardware.nvidia.quant_ops import ops
from flash_vla.hardware.nvidia.rtx5090.pi05.backends.mxfp8_backbone import GemmPlan, library
from flash_vla.runtime.cuda.graph import StreamGraph
from flash_vla.runtime.runner import Scratch

ROWS, LAYERS = 968, 18
# GEMM -> (N, K, beta): gate|up writes bf16 [M, 2 x 16384]; down adds [M, 2048] to the residual.
SHAPES = {"gate_up": (32768, 2048, 0.0), "down": (2048, 16384, 1.0)}
CONFIGS = {0: "128x128x128 persistent", 1: "128x64x128 persistent", 2: "128x32x128 persistent",
           3: "128x128x128 stream-K", 4: "128x64x128 stream-K"}   # mxfp8_backbone.cu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    if library().mxfp8_gemm_configs() != len(CONFIGS):
        raise RuntimeError("CONFIGS no longer describes mxfp8_backbone.cu")
    scratch = Scratch(device)
    generator = torch.Generator(device=device).manual_seed(0)
    rows = []
    for gemm, (cols, depth, beta) in SHAPES.items():
        activation = ops.quantize(
            torch.randn(ROWS, depth, generator=generator, device=device).bfloat16(),
            ops.empty(ROWS, depth, "mxfp8", device))
        weights = [ops.quantize(
            (torch.randn(cols, depth, generator=generator, device=device) * 0.02).bfloat16(),
            ops.empty(cols, depth, "mxfp8", device)) for _ in range(LAYERS)]
        residual = torch.randn(ROWS, cols, generator=generator, device=device).bfloat16()
        output = torch.empty(ROWS, cols, dtype=torch.bfloat16, device=device)   # bf16 [M, N]
        first_layer: torch.Tensor | None = None                                # config 0's
        for config, tiles in CONFIGS.items():
            plans = [GemmPlan(config, activation, weight, output, beta, scratch, rows=ROWS,
                              mask=None) for weight in weights]
            output.copy_(residual)
            plans[0].run()
            layer_output = output.float()
            first_layer = layer_output if first_layer is None else first_layer
            graph = StreamGraph()
            graph.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(graph.stream):
                for plan in plans:
                    plan.run()
            graph.stream.synchronize()
            with graph.capture():
                for plan in plans:
                    plan.run()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            samples_us = []
            with torch.cuda.stream(graph.stream):
                for _ in range(args.reps):
                    output.copy_(residual)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples_us.append(start.elapsed_time(end) * 1e3 / LAYERS)
            median_us = statistics.median(samples_us)
            row = dict(gemm=gemm, m=ROWS, n=cols, k=depth, beta=beta, config=config, tiles=tiles,
                       median_us=median_us, min_us=min(samples_us),
                       tflops=2 * ROWS * cols * depth / median_us / 1e6,
                       rel_rms_against_config0=float(
                           (layer_output - first_layer).norm() / first_layer.norm()),
                       samples_us=samples_us)
            rows.append(row)
            print(f"{gemm:8s} config {config} {tiles:24s} {median_us:8.1f} us "
                  f"{row['tflops']:6.0f} TFLOP/s  rel_rms vs 0: "
                  f"{row['rel_rms_against_config0']:.2e}", flush=True)
            del graph, plans
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(device), layers=LAYERS, reps=args.reps, rows=rows),
        indent=1) + "\n")


if __name__ == "__main__":
    main()
