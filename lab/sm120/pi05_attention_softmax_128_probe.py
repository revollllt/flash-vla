"""Compare fixed native softmax 128/256 at nine actual attention boundaries."""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from statistics import median
import subprocess

import torch
from safetensors import safe_open

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.bench import bench_gpu_time
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_attention, triton_qk_attention
from pi05_attention_qk_triton_confirm import prepare


def measure(args):
    with safe_open(str(args.snapshot), framework="pt", device="cpu") as stored:
        metadata = stored.metadata()
        labels = [name[:-2] for name in sorted(stored.keys()) if name.endswith("_q")]
        pairs = [(label, *(stored.get_tensor(label + "_" + role).cuda()
                           for role in ("q", "k", "v", "mask"))) for label in labels]
    assert len(pairs) == 9, labels
    logits = torch.empty((400, 1018), dtype=torch.float32, device="cuda")
    probabilities = torch.empty((400, 1018), dtype=torch.bfloat16, device="cuda")
    working_q = torch.empty((400, 256), dtype=torch.bfloat16, device="cuda")
    library = fused_attention.library()
    control = library.pi05_attention_softmax_launch
    candidate = library.pi05_attention_softmax_128_launch
    candidate.argtypes = control.argtypes
    candidate.restype = ctypes.c_int32

    def attention(q, k, v, mask, softmax):
        triton_qk_attention._qk[(13, 32)](
            q, k, logits, num_warps=4, num_stages=3,
            enable_fp_fusion=False, enable_reflect_ftz=False)
        rc = softmax(
            logits.data_ptr(), mask.data_ptr(), probabilities.data_ptr(),
            400, 1018, 0.0625, torch.cuda.current_stream(q.device).cuda_stream)
        if rc:
            raise RuntimeError(f"native attention softmax CUDA error {rc}")
        torch.mm(probabilities, v, out=q)

    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "snapshot_metadata": metadata,
        "threads": {"control": 256, "candidate": 128},
        "numerical_note": "FP32 sum association changes; exact P is checked, not assumed",
        "correctness": [], "timings": [],
        "cache_policy": "nine actual pairs reused, warm working set, no L2 flush",
        "timing_boundary": "same-address Q reset + current Triton QK + native softmax + torch PV",
        "timer": "one ABBA; 16 nine-call sequences per graph; 5 warm and 15 measured replays",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    limits = tolerances()["shallow"]
    for label, q, k, v, mask in pairs:
        working_q.copy_(q)
        attention(working_q, k, v, mask, control)
        expected_p, expected_out = probabilities.clone(), working_q.clone()
        working_q.copy_(q)
        attention(working_q, k, v, mask, candidate)
        row = {"pair": label}
        passed = True
        for name, expected, actual in (
            ("probabilities", expected_p, probabilities),
            ("output", expected_out, working_q),
        ):
            metrics = error_metrics(expected, actual)
            accepted = (metrics["rel_rms"] <= limits["rel_rms_max"]
                        and metrics["cosine_similarity"] >= limits["cosine_min"])
            row[name] = {**metrics, "bitwise_equal": torch.equal(expected, actual),
                         "changed_elements": torch.count_nonzero(expected != actual).item(),
                         "passed": accepted}
            passed = passed and accepted
        report["correctness"].append(row)
        report["status"] = "checking_numerics" if passed else "numerical_failed"
        save()
        print(json.dumps(row), flush=True)
        assert passed, row
    report["status"] = "numerics_passed"
    save()

    def sequence(softmax):
        for _, saved_q, k, v, mask in pairs:
            working_q.copy_(saved_q)
            attention(working_q, k, v, mask, softmax)

    for label, softmax in (
        ("A1", control), ("B1", candidate), ("B2", candidate), ("A2", control),
    ):
        samples = bench_gpu_time(
            sequence, input_args=(softmax,), enable_cupti=False,
            use_cuda_graph=True, cold_l2_cache=False, num_iters_within_graph=16,
            dry_run_iters=5, repeat_iters=15)
        row = {"leg": label, "median_ms_per_9_calls": median(samples),
               "samples_ms_per_9_calls": samples}
        report["timings"].append(row)
        save()
        print(json.dumps(row), flush=True)
    a1, b1, b2, a2 = [r["median_ms_per_9_calls"] for r in report["timings"]]
    report["summary"] = {
        "mean_gain_ms_per_9_calls": (a1 + a2 - b1 - b2) / 2,
        "conservative_gap_ms_per_9_calls": min(a1, a2) - max(b1, b2),
        "control_drift_ms_per_9_calls": abs(a2 - a1),
        "candidate_drift_ms_per_9_calls": abs(b2 - b1),
    }
    report["status"] = "complete"
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "time"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-id")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.checkpoint is None or args.checkpoint_id is None:
            parser.error("prepare requires --checkpoint and --checkpoint-id")
        prepare(args)
    else:
        if args.out is None:
            parser.error("time requires --out")
        measure(args)


if __name__ == "__main__":
    main()
