"""Actual ten-step action output snapshots and matched-reset graph timings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from benchmarks.kernels import _graph_samples
from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_qkv, torch_ops
from flash_vla.inference import build, resolve
from flash_vla.runtime.runner import Scratch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    site = "action_expert_action_out_proj"
    scratch = Scratch(torch.device("cuda"))
    candidate = fused_qkv.make_wrappers(scratch, {site})[site]
    reference = torch_ops.action_expert_action_out_proj
    engine = build(
        resolve("rtx5090/pi05"), "reference", seed=42,
        converted_checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
        checkpoint_digest=args.checkpoint_id)
    inputs = engine.sample_inputs(42)
    calls, initial_outputs, checks = [], [], []
    limits = tolerances()["shallow"]

    def inspect(name, function):
        if name != site:
            return function

        def check(x, weight, bias, out, norm_factor):
            initial = out.clone()
            original_factor = norm_factor.clone()
            call = (x.clone(), weight, bias, initial.clone(), original_factor.clone())
            calls.append(call)
            initial_outputs.append(initial)
            function(x, weight, bias, out, norm_factor)
            candidate(*call)
            result = error_metrics(out, call[3])
            assert result["rel_rms"] <= limits["rel_rms_max"], result
            assert result["cosine_similarity"] >= limits["cosine_min"], result
            assert torch.equal(call[4], original_factor)
            assert torch.equal(norm_factor, original_factor)
            checks.append({"step": len(checks), "output": result,
                           "bitwise_equal": torch.equal(out, call[3]),
                           "norm_factor_unchanged": True})
            print(json.dumps(checks[-1]), flush=True)
            return out

        return check

    engine.stage(**inputs)
    with engine.instrument(inspect):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "action_expert":
                engine.run_eager(step.name)
            else:
                engine.replay(step.name)
    torch.cuda.synchronize()
    scratch.freeze()
    timings = []
    for label, function in (("torch", reference), ("fused", candidate),
                            ("fused", candidate), ("torch", reference)):
        def invoke(index):
            call = calls[index % len(calls)]
            # Both timed paths include the same reset before the Euler update.
            call[3].copy_(initial_outputs[index % len(calls)])
            function(*call)

        samples = _graph_samples(invoke, n_inner=len(calls), reps=50, warmup=len(calls))
        row = {"backend": label, "samples_ms_per_call": samples,
               "median_ms_per_call": median(samples),
               "median_ms_all_steps": median(samples) * len(calls)}
        timings.append(row)
        print(json.dumps({key: value for key, value in row.items()
                          if key != "samples_ms_per_call"}), flush=True)
    report = {
        "target": "rtx5090/pi05", "checkpoint": args.checkpoint_id, "seed": 42,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "site": site, "calls": len(calls), "x_shape": list(calls[0][0].shape),
        "weight_shape": list(calls[0][1].shape), "dtype": str(calls[0][0].dtype),
        "timer": "CUDA graph on capture stream, all actual ten-step snapshots",
        "cache_policy": "No flush; ten-step working set reused in L2, not cold-weight timing",
        "weight_unique_data_ptrs": len({call[1].data_ptr() for call in calls}),
        "weight_unique_storage_bases": len({call[1].untyped_storage().data_ptr() for call in calls}),
        "weight_storage_offsets": [call[1].storage_offset() for call in calls],
        "snapshot_tensor_bytes": sum(t.numel() * t.element_size()
                                     for call in calls for t in call)
                                 + sum(t.numel() * t.element_size() for t in initial_outputs),
        "input_reset": "same initial out.copy_ before each timed call on both sides",
        "parity": checks, "timings": timings}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
