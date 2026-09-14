"""Synthetic expert attention screen with deployed shapes, aliases, and masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.bench import bench_gpu_time, calculate_rotation_count
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_attention, torch_ops
from flash_vla.runtime.runner import Scratch


def inputs(magnitude=1.0, valid_prompt=32):
    torch.manual_seed(42)
    q = (torch.randn(400, 256, device="cuda") * magnitude).bfloat16()
    k = (torch.randn(1018, 256, device="cuda") * magnitude).bfloat16()
    v = torch.randn_like(k)
    mask = torch.zeros(1018, device="cuda", dtype=torch.bfloat16)
    mask[768 + valid_prompt:968] = -2.3819763e38
    return q, k, v, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scratch = Scratch(torch.device("cuda"))
    candidate = fused_attention.make_wrappers(scratch)["action_expert_attention"]
    reference = torch_ops.action_expert_attention
    checks = []
    limits = tolerances()["shallow"]
    for magnitude, valid_prompt in ((1.0, 32), (1.0, 200), (4.0, 32)):
        q, k, v, mask = inputs(magnitude, valid_prompt)
        ref_scores = q.float() @ k.float().T
        scores = torch.empty_like(ref_scores)
        torch.mm(q, k.T, out_dtype=torch.float32, out=scores)
        ref_out = q.clone()
        test_out = q.clone()
        reference(ref_out, k, v, mask, ref_out)
        candidate(test_out, k, v, mask, test_out)
        result = error_metrics(ref_out, test_out)
        assert result["rel_rms"] <= limits["rel_rms_max"], result
        assert result["cosine_similarity"] >= limits["cosine_min"], result
        # Runtime masked keys must have zero influence even with large values.
        changed_v = v.clone()
        changed_v[768 + valid_prompt:968] = 10000
        changed_out = q.clone()
        candidate(changed_out, k, changed_v, mask, changed_out)
        assert torch.equal(changed_out, test_out)
        checks.append({"magnitude": magnitude, "valid_prompt": valid_prompt,
                       "scores": error_metrics(ref_scores, scores),
                       "output": result, "bitwise_equal": torch.equal(ref_out, test_out),
                       "masked_values_invariant": True})
    # Both timed paths reset Q so graph replay sees the same distribution.
    # The common copy is included in both times, and out=Q remains explicit.
    def torch_chain(source_q, q, k, v, mask):
        q.copy_(source_q)
        reference(q, k, v, mask, q)

    def fused_chain(source_q, q, k, v, mask):
        q.copy_(source_q)
        candidate(q, k, v, mask, q)

    source_q, k, v, mask = inputs()
    values = (source_q, source_q.clone(), k, v, mask)
    calls = calculate_rotation_count(values, "cuda")
    timings = []
    for name, fn in (("torch", torch_chain), ("fused", fused_chain),
                     ("torch", torch_chain), ("fused", fused_chain)):
        source_q, k, v, mask = inputs()
        values = (source_q, source_q.clone(), k, v, mask)
        fn(*tuple(value.clone() for value in values))
        torch.cuda.synchronize()
        samples = bench_gpu_time(
            fn, input_args=values, enable_cupti=False, use_cuda_graph=True,
            cold_l2_cache=True, num_iters_within_graph=calls,
            dry_run_iters=5, repeat_iters=30)
        timings.append({"backend": name, "median_ms": median(samples), "samples_ms": samples})
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "shape": {"q": [400, 256], "k": [1018, 256], "v": [1018, 256]},
              "dtype": "bfloat16 with float32 logits", "output_alias": "Q",
              "input_reset": "Q.copy_(source_Q) included in both timed paths",
              "inputs": "synthetic seed 42 at deployed shapes",
              "timer": "CUDA graph on capture stream", "cold_l2_cache": True,
              "calls_per_graph": calls, "correctness": checks, "timings": timings}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"correctness": checks, "calls_per_graph": calls,
                      "timings": [{k: v for k, v in t.items() if k != "samples_ms"}
                                  for t in timings]}, indent=2))


if __name__ == "__main__":
    main()
