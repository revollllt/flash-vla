"""Focused expert QKV parity and cold-weight graph timing at the Pi0.5 shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_qkv, torch_ops
from flash_vla.runtime.runner import Scratch
from measurement.kernel_bench import bench_gpu_time, calculate_rotation_count


def inputs(seed, magnitude=1.0):
    torch.manual_seed(seed)
    device, dtype = "cuda", torch.bfloat16
    x = (torch.randn(50, 1024, device=device) * magnitude).to(dtype)
    scale = (1 + torch.randn(1024, device=device) * 0.1).to(dtype)
    weight = (torch.randn(1024, 2560, device=device) / 32).to(dtype)
    bias = (torch.randn(2560, device=device) * 0.1).to(dtype)
    phase = torch.randn(50, 128, device=device)
    rope = torch.stack((phase.cos(), phase.sin()), dim=-1).flatten(1).to(dtype)
    return x, scale, weight, bias, rope


def outputs():
    q = torch.empty(400, 256, device="cuda", dtype=torch.bfloat16)
    # Deployed K/V point at action rows after the 968-token prefix.
    k = torch.full((1018, 256), 3, device="cuda", dtype=torch.bfloat16)
    v = torch.full_like(k, 5)
    factor = torch.empty(50, device="cuda", dtype=torch.bfloat16)
    return (q, k[968:], v[968:], factor), (k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scratch = Scratch(torch.device("cuda"))
    candidate = fused_qkv.make_wrappers(scratch)["action_expert_norm_qkv_rope"]
    reference = torch_ops.action_expert_norm_qkv_rope
    checks = []
    limits = tolerances()["shallow"]
    for seed, magnitude in ((42, 1.0), (7, 1e-3), (19, 1e3)):
        values = inputs(seed, magnitude)
        ref_out, _ = outputs()
        test_out, cache = outputs()
        reference(*values, *ref_out)
        candidate(*values, *test_out)
        torch.cuda.synchronize()
        metrics = {name: error_metrics(ref, test)
                   for name, ref, test in zip(("q", "k", "v", "factor"), ref_out, test_out)}
        for result in metrics.values():
            assert result["rel_rms"] <= limits["rel_rms_max"], result
            assert result["cosine_similarity"] >= limits["cosine_min"], result
        assert torch.all(cache[0][:968] == 3)
        assert torch.all(cache[1][:968] == 5)
        checks.append({"seed": seed, "magnitude": magnitude, "metrics": metrics,
                       "bitwise_equal": {name: torch.equal(ref, test)
                           for name, ref, test in zip(("q", "k", "v", "factor"),
                                                     ref_out, test_out)}})
    values = inputs(42)
    buffers, _ = outputs()
    tensors = (*values, *buffers)
    candidate(*tensors)
    reference(*tensors)
    torch.cuda.synchronize()
    calls = calculate_rotation_count(tensors, "cuda")
    timings = []
    for name, fn in (("torch", reference), ("fused", candidate),
                     ("torch", reference), ("fused", candidate)):
        samples = bench_gpu_time(
            fn, input_args=tensors, enable_cupti=False, use_cuda_graph=True,
            cold_l2_cache=True, num_iters_within_graph=calls,
            dry_run_iters=5, repeat_iters=30)
        timings.append({"backend": name, "median_ms": median(samples), "samples_ms": samples})
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "shape": {"x": [50, 1024], "weight": [1024, 2560], "q": [400, 256]},
              "dtype": "bfloat16", "inputs": "synthetic seed 42 at deployed shapes",
              "timer": "CUDA graph on capture stream", "cold_l2_cache": True,
              "calls_per_graph": calls, "correctness": checks, "timings": timings}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"correctness": checks, "calls_per_graph": calls,
                      "timings": [{k: v for k, v in t.items() if k != "samples_ms"}
                                  for t in timings]}, indent=2))


if __name__ == "__main__":
    main()
