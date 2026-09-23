"""Fixed 16x32x64 softmax+PV screen on the existing nine actual QK snapshots."""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from statistics import mean, median
import subprocess

import torch
from safetensors import safe_open

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.bench import bench_gpu_time
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import (
    fused_attention, triton_qk_attention)


def checked(status, operation):
    if status:
        raise RuntimeError(f"{operation}: CUDA error {status}")


def load_cases(snapshot):
    cases = []
    with safe_open(str(snapshot), framework="pt", device="cpu") as stored:
        metadata = stored.metadata()
        for label in sorted(name[:-2] for name in stored.keys() if name.endswith("_q")):
            q, k, v, mask = (stored.get_tensor(label + "_" + role).cuda()
                             for role in ("q", "k", "v", "mask"))
            logits = torch.empty((400, 1018), dtype=torch.float32, device="cuda")
            triton_qk_attention._qk[(13, 32)](
                q, k, logits, num_warps=4, num_stages=3,
                enable_fp_fusion=False, enable_reflect_ftz=False)
            cases.append((label, q, logits, v, mask, q.clone()))
    torch.cuda.synchronize()
    return cases, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "time"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "tile": [16, 32, 64], "warps": 4, "grid": [25, 8, 1], "v_stages": 3,
        "phase": "load", "correctness": [], "legs": [],
        "library": str(args.library),
        "timer": "one ABBA; 16 repetitions of nine calls/graph; 5 warm, 30 samples",
        "boundary": "identical saved-Q reset + full softmax/PV; QK/preparation excluded",
        "cache": "nine actual cases reused; warm working set, no L2 flush or clock lock",
        "rounding": "full-row FP32 normalization -> BF16 P -> FP32 PV -> BF16 output",
        "reduction_change": "32-thread row XOR tree replaces native 256-thread tree",
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        library = ctypes.CDLL(str(args.library.resolve()))
        library.softmax_pv_launch.argtypes = [ctypes.c_void_p] * 6
        library.softmax_pv_launch.restype = ctypes.c_int32
        native = fused_attention.library()
        cases, metadata = load_cases(args.snapshot)
        report["snapshot_metadata"] = metadata
        report["case_count"] = len(cases)
        p = torch.empty((400, 1018), dtype=torch.bfloat16, device="cuda")
        p_diagnostic = torch.empty_like(p)

        def reference(case):
            _, saved_q, logits, v, mask, out = case
            out.copy_(saved_q)
            checked(native.pi05_attention_softmax_launch(
                logits.data_ptr(), mask.data_ptr(), p.data_ptr(), 400, 1018, 0.0625,
                torch.cuda.current_stream().cuda_stream), "native softmax")
            torch.mm(p, v, out=out)

        def candidate(case, diagnostic=None):
            _, saved_q, logits, v, mask, out = case
            out.copy_(saved_q)
            checked(library.softmax_pv_launch(
                logits.data_ptr(), mask.data_ptr(), v.data_ptr(), out.data_ptr(),
                diagnostic.data_ptr() if diagnostic is not None else None,
                torch.cuda.current_stream().cuda_stream), "fused softmax/PV")

        if args.phase == "check":
            report["phase"] = "correctness"
            limits = tolerances()["shallow"]
            for case in cases:
                label, _, _, _, mask, out = case
                reference(case)
                expected_out = out.clone()
                expected_p = p.clone()
                candidate(case, p_diagnostic)
                # Validate P from the diagnostic branch and output from the timed branch.
                candidate(case)
                row = {"case": label, "mask_values": torch.unique(mask).tolist()}
                for name, expected, actual in (
                    ("probabilities", expected_p, p_diagnostic),
                    ("output", expected_out, out),
                ):
                    metrics = error_metrics(expected, actual)
                    row[name] = {**metrics, "exact": torch.equal(expected, actual)}
                row["correct"] = all(
                    row[name]["rel_rms"] <= limits["rel_rms_max"]
                    and row[name]["cosine_similarity"] >= limits["cosine_min"]
                    for name in ("probabilities", "output"))
                report["correctness"].append(row)
                save()
                print(json.dumps(row), flush=True)
                if not row["correct"]:
                    raise RuntimeError(f"softmax/PV numerical check failed: {label}")
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                candidate(cases[0])
                torch.cuda.synchronize()
            trace = args.output.with_suffix(".trace.json")
            profile.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())["traceEvents"]
            report["kernel_resources"] = [
                {"name": e["name"], "args": e["args"]} for e in events
                if e.get("cat") == "kernel" and "softmax_pv(" in e.get("name", "")]
            report["diagnostic_trace"] = str(trace)
        else:
            # Both routes share all addresses, including each case's mutable output.
            def sequence(function):
                for case in cases:
                    function(case)

            report["phase"] = "abba"
            for label, function in (
                    ("A", reference), ("B", candidate), ("B", candidate), ("A", reference)):
                samples = bench_gpu_time(
                    sequence, input_args=(function,), enable_cupti=False,
                    use_cuda_graph=True, cold_l2_cache=False,
                    num_iters_within_graph=16, dry_run_iters=5, repeat_iters=30)
                leg = {"route": label, "median_us_per_9_calls": median(samples) * 1000,
                       "samples_ms_per_9_calls": samples}
                report["legs"].append(leg)
                save()
                print(json.dumps(leg), flush=True)
            a = [r["median_us_per_9_calls"] for r in report["legs"] if r["route"] == "A"]
            b = [r["median_us_per_9_calls"] for r in report["legs"] if r["route"] == "B"]
            gain = mean(a) - mean(b)
            drift_a, drift_b = abs(a[1] - a[0]), abs(b[1] - b[0])
            report.update(mean_gain_us_per_9_calls=gain, control_drift_us=drift_a,
                          candidate_drift_us=drift_b, conservative_gain_us=min(a)-max(b),
                          gain_exceeds_observed_drift=gain > max(drift_a, drift_b))
        report["phase"] = "complete"
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
