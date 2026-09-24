"""Compare current fused-vision QKV with its existing cfg10 bias-GEMM route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from flash_vla.runtime.cuda.timing import graph_samples
from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_vision, fused_vision
from flash_vla.inference import build, resolve
from flash_vla.runtime.runner import Scratch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    site = "vision_encoder_norm_qkv"
    candidate_scratch = Scratch(torch.device("cuda"))
    control_scratch = Scratch(torch.device("cuda"))
    candidate = cutlass_vision.make_wrappers(candidate_scratch, {site})[site]
    control = fused_vision.make_wrappers(control_scratch, {site})[site]
    engine = build(
        resolve("rtx5090/pi05"), "shipped", seed=42,
        converted_checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
        checkpoint_digest=args.checkpoint_id)
    inputs = engine.sample_inputs(42)
    calls, checks = [], []
    limits = tolerances()["shallow"]

    def inspect(name, function):
        if name != site:
            return function

        def check(x, norm_w, norm_b, weight, bias, out):
            call = (x.clone(), norm_w, norm_b, weight, bias, out)
            calls.append(call)
            function(x, norm_w, norm_b, weight, bias, out)
            expected = out.clone()
            candidate(*call)
            metrics = error_metrics(expected, out)
            assert metrics["rel_rms"] <= limits["rel_rms_max"], metrics
            assert metrics["cosine_similarity"] >= limits["cosine_min"], metrics
            checks.append({"layer": len(checks), "output": metrics,
                           "bitwise_equal": torch.equal(expected, out),
                           "bias_shape": list(bias.shape), "bias_dtype": str(bias.dtype)})
            print(json.dumps(checks[-1]), flush=True)
            # Keep every later snapshot on the current shipped trajectory.
            out.copy_(expected)
            return out

        return check

    engine.stage(**inputs)
    with engine.instrument(inspect):
        engine.run_eager("vision_encoder")
    torch.cuda.synchronize()
    # Inputs are immutable snapshots and each call fully overwrites its output;
    # both paths therefore use the same graph/reset policy without reset copies.
    for call in calls:
        control(*call)
    torch.cuda.synchronize()
    candidate_scratch.freeze()
    control_scratch.freeze()
    timings = []
    for label, function in (("fused-vision", control), ("cfg10", candidate),
                            ("cfg10", candidate), ("fused-vision", control)):
        def invoke(index):
            function(*calls[index % len(calls)])

        samples = graph_samples(invoke, n_inner=len(calls), reps=30, warmup=len(calls))
        row = {"backend": label, "samples_ms_per_call": samples,
               "median_ms_per_call": median(samples),
               "median_ms_all_layers": median(samples) * len(calls)}
        timings.append(row)
        print(json.dumps({key: value for key, value in row.items()
                          if key != "samples_ms_per_call"}), flush=True)
    report = {
        "target": "rtx5090/pi05", "checkpoint": args.checkpoint_id, "seed": 42,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "site": site, "calls": len(calls), "x_shape": list(calls[0][0].shape),
        "weight_shape": list(calls[0][3].shape), "weight_stride": list(calls[0][3].stride()),
        "dtype": str(calls[0][0].dtype), "config": 10,
        "timer": "CUDA graph on capture stream, cycles actual 27-layer input/weight snapshots",
        "reset": "None on both paths: immutable inputs and completely overwritten output",
        "cache_policy": "Actual 27-layer rotation, no artificial cache flush",
        "weight_unique_data_ptrs": len({call[3].data_ptr() for call in calls}),
        "input_and_weight_bytes": sum(t.numel() * t.element_size()
                                     for call in calls for t in call[:5]),
        "parity": checks, "timings": timings}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
