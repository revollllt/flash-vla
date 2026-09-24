"""Three fixed-tile PV screens on the existing real step-0/layer-0 P/V snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from measurement.kernel_bench import bench_gpu_time


@triton.jit
def _pv(P, V, Out, BN: tl.constexpr, BK: tl.constexpr):
    # Every tested tile divides M=400 and N=256; only K=1018 has a tail.
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(0, 1018, BK):
        k = start + kk
        p = tl.load(P + rows[:, None] * 1018 + k[None, :],
                    mask=k[None, :] < 1018, other=0.0)
        v = tl.load(V + k[:, None] * 256 + cols[None, :],
                    mask=k[:, None] < 1018, other=0.0)
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
    result = acc.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Out + rows[:, None] * 256 + cols[None, :], result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with safe_open(str(args.snapshot), framework="pt", device="cpu") as stored:
        probabilities = stored.get_tensor("probabilities").cuda()
        values = stored.get_tensor("values").cuda()
        metadata = stored.metadata()
    reference_out = torch.empty((400, 256), dtype=torch.bfloat16, device="cuda")
    actual_out = torch.empty_like(reference_out)
    torch.mm(probabilities, values, out=reference_out)
    limits = tolerances()["shallow"]

    def reference(p, v, out):
        torch.mm(p, v, out=out)

    results = []
    # Baseline: 200 CTAs. Change only BK (loop depth), then BN (400 CTAs).
    for bn, bk in ((32, 64), (32, 128), (16, 64)):
        def candidate(p, v, out):
            _pv[(25, 256 // bn)](
                p, v, out, bn, bk, num_warps=4, num_stages=3,
                enable_fp_fusion=False, enable_reflect_ftz=False)

        candidate(probabilities, values, actual_out)
        torch.cuda.synchronize()
        metrics = error_metrics(reference_out, actual_out)
        assert metrics["rel_rms"] <= limits["rel_rms_max"], metrics
        assert metrics["cosine_similarity"] >= limits["cosine_min"], metrics
        bitwise_equal = torch.equal(reference_out, actual_out)

        rows = []
        for name, function, out in (
            ("torch", reference, reference_out),
            ("triton", candidate, actual_out),
            ("triton", candidate, actual_out),
            ("torch", reference, reference_out),
        ):
            samples = bench_gpu_time(
                function, input_args=(probabilities, values, out),
                enable_cupti=False, use_cuda_graph=True, cold_l2_cache=False,
                num_iters_within_graph=64, dry_run_iters=5, repeat_iters=30)
            rows.append({"case": name, "median_ms": median(samples), "samples_ms": samples})
        result = {"tile": [16, bn, bk], "grid": [25, 256 // bn],
                  "num_warps": 4, "num_stages": 3,
                  "pv_output": metrics, "bitwise_equal": bitwise_equal, "timings": rows}
        results.append(result)
        print(json.dumps({**result, "timings": [
            {k: v for k, v in row.items() if k != "samples_ms"} for row in rows]},
                         indent=2), flush=True)

    report = {
        "snapshot_metadata": metadata,
        "cache_policy": "one actual P/V pair repeatedly reused; warm-cache screen",
        "timing_boundary": "PV only; no padding, copies, JIT or allocation inside timing",
        "timer": "fresh CUDA graph on capture stream; ABBA per tile; 64 calls per graph",
        "candidates": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
