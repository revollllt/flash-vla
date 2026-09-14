"""Actual checkpoint vision call snapshots: LayerNorm/QKV and LayerNorm/FFN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from benchmarks.kernels import _graph_samples
from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_vision, torch_ops
from flash_vla.inference import build, resolve
from flash_vla.runtime.runner import Scratch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scratch = Scratch(torch.device("cuda"))
    candidates = fused_vision.make_wrappers(scratch)
    engine = build(
        resolve("rtx5090/pi05"), "reference", seed=42,
        converted_checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
        checkpoint_digest=args.checkpoint_id)
    inputs = engine.sample_inputs(42)
    calls = {name: [] for name in candidates}
    checks = []
    limits = tolerances()["shallow"]

    def inspect(name, fn):
        if name not in candidates:
            return fn

        def check(x, norm_w, norm_b, weight, bias, out):
            layer = len(calls[name])
            # Each call keeps its actual input before the vision residual changes.
            snapshot = x.clone()
            call = (snapshot, norm_w, norm_b, weight, bias, out)
            calls[name].append(call)
            fn(x, norm_w, norm_b, weight, bias, out)
            if layer in (0, 13, 26):
                expected = out.clone()
                candidates[name](*call)
                expected_norm = torch_ops._ln(snapshot.view(-1, 1152), norm_w, norm_b)
                actual_norm = scratch(
                    "pi05_vision_normalized", expected_norm.shape, x.dtype, x.device)
                metrics = {"output": error_metrics(expected, out),
                           "norm": error_metrics(expected_norm, actual_norm)}
                for result in metrics.values():
                    assert result["rel_rms"] <= limits["rel_rms_max"], result
                    assert result["cosine_similarity"] >= limits["cosine_min"], result
                assert torch.isfinite(out).all().item()
                checks.append({"site": name, "layer": layer, **metrics})
                print(json.dumps(checks[-1]), flush=True)
                # Later snapshots remain on the original reference trajectory.
                out.copy_(expected)
            return out

        return check

    engine.stage(**inputs)
    with engine.instrument(inspect):
        engine.run_eager("vision_encoder")
    torch.cuda.synchronize()
    scratch.freeze()
    timings = []
    for name, site_calls in calls.items():
        reference = getattr(torch_ops, name)
        for label, fn in (("torch", reference), ("fused", candidates[name]),
                          ("fused", candidates[name]), ("torch", reference)):
            def invoke(index):
                fn(*site_calls[index % len(site_calls)])

            samples = _graph_samples(
                invoke, n_inner=len(site_calls), reps=30, warmup=len(site_calls))
            row = {"site": name, "backend": label, "samples_ms_per_call": samples,
                   "median_ms_per_call": median(samples),
                   "median_ms_all_layers": median(samples) * len(site_calls)}
            timings.append(row)
            print(json.dumps({key: value for key, value in row.items()
                              if key != "samples_ms_per_call"}), flush=True)
    report = {
        "target": "rtx5090/pi05", "checkpoint": args.checkpoint_id, "seed": 42,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "dtype": "bfloat16", "sites": {
            name: {"calls": len(items), "x_shape": list(items[0][0].shape),
                   "weight_shape": list(items[0][3].shape)} for name, items in calls.items()},
        "timer": "CUDA graph; 27 actual layer weights and input snapshots, capture stream",
        "parity": checks, "timings": timings}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
