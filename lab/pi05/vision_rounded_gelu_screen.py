"""One cfg10 rounded-GELU experiment, sharing the control's native library."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import weakref

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.inference import build, parse_options, resolve
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_backbone, cutlass_vision, fused_vision
from flash_vla.runtime.runner import Scratch
from lab.pi05.cutlass_gemm_screen import samples_ms
from lab.pi05.cutlass_vision_chain import record_calls

SITE = "vision_encoder_norm_ffn_up"


class _Plan:
    def __init__(self, library, scratch, a, weight, bias, output, stream):
        m, k = a.shape
        n = weight.shape[1]
        size = library.vision_gelu_gemm_workspace(m, k, n)
        self.workspace = scratch("lab_vision_gelu_workspace", (max(size, 1),),
                                 torch.uint8, a.device)
        self.tensors = (a, weight, bias, output)
        self.handle = ctypes.c_void_p()
        cutlass_backbone._check(library.vision_gelu_gemm_plan(
            m, k, n, a.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
            self.workspace.data_ptr(), stream, ctypes.byref(self.handle)),
            f"vision_gelu_gemm_plan M={m} K={k} N={n}")
        self.destroy = weakref.finalize(self, library.vision_gelu_gemm_destroy, self.handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True).strip()
    report = {"source_revision": revision, "seed": args.seed, "options": args.option,
              "site": SITE, "tile": [64, 128, 32], "warp_tile": [32, 64, 32],
              "stages": 5, "epilogue_k_is_heavy": True, "phase": "native_build",
              "scope": "one complete LayerNorm/bias-GEMM/GELU chain across 27 actual layers",
              "native_source": str(cutlass_backbone._SOURCE),
              "cutlass_dir": os.environ.get("CUTLASS_DIR", str(root / "third_party/cutlass")),
              "is_deployment_sequence": False, "reps_per_leg": 15, "legs": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(json.dumps(report), flush=True)
    try:
        # Load once from this checkout: the old and new cfg10 types share this .so.
        library = cutlass_vision._library()
        report["native_library"] = library._name
        library.vision_gelu_gemm_workspace.argtypes = [ctypes.c_int32] * 3
        library.vision_gelu_gemm_workspace.restype = ctypes.c_int64
        library.vision_gelu_gemm_plan.argtypes = (
            [ctypes.c_int32] * 3 + [ctypes.c_void_p] * 6
            + [ctypes.POINTER(ctypes.c_void_p)])
        library.vision_gelu_gemm_plan.restype = ctypes.c_int32
        library.vision_gelu_gemm_run.argtypes = [ctypes.c_void_p] * 2
        library.vision_gelu_gemm_run.restype = ctypes.c_int32
        library.vision_gelu_gemm_destroy.argtypes = [ctypes.c_void_p]
        library.vision_gelu_gemm_destroy.restype = None
        library.vision_gelu_gemm_shared_bytes.argtypes = []
        library.vision_gelu_gemm_shared_bytes.restype = ctypes.c_int64
        report["candidate_dynamic_shared_bytes"] = library.vision_gelu_gemm_shared_bytes()
        resources = subprocess.check_output(
            [str(Path(os.environ["CUDA_HOME"]) / "bin/cuobjdump"),
             "--dump-resource-usage", library._name], text=True)
        resource_path = args.output.with_name(args.output.stem + "-resources.txt")
        resource_path.write_text(resources)
        report["resources"] = str(resource_path)
        # nvcc's existing -Xptxas=-v output also remains in the complete run log.
        report["phase"] = "record_calls"
        save()
        engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                       **parse_options(args.option))
        report["deployment_plan"] = dict(engine.identity.plan)
        recorded = record_calls(engine, engine.sample_inputs(args.seed), (SITE,))[SITE]
        if not recorded:
            raise RuntimeError("no actual vision FFN-up calls were recorded")
        calls = [(values[:-1], expected) for values, _, expected in recorded]
        del recorded
        output = torch.empty_like(calls[0][1])
        scratch = Scratch(output.device)
        control = cutlass_vision.make_wrappers(scratch, {SITE})[SITE]
        plans = {}

        def candidate(x, norm_w, norm_b, weight, bias, out):
            normalized = fused_vision._norm(x, norm_w, norm_b, scratch)
            projected = out.view(-1, 4304)
            stream = torch.cuda.current_stream().cuda_stream
            key = (normalized.data_ptr(), weight.data_ptr(), bias.data_ptr(), projected.data_ptr())
            if key not in plans:
                plans[key] = _Plan(library, scratch, normalized, weight, bias, projected, stream)
            cutlass_backbone._check(library.vision_gelu_gemm_run(plans[key].handle, stream),
                                   "vision_gelu_gemm_run")
            return out

        report.update(actual_layers=len(calls), calls_per_graph=len(calls),
                      shape_mkn=[calls[0][0][0].numel() // 1152, 1152, 4304],
                      weight_rotation_bytes=sum(values[3].numel() * values[3].element_size()
                                                for values, _ in calls),
                      dtype=str(output.dtype))
        report["phase"] = "control_correctness"
        control_equal = []
        for values, expected in calls:
            control(*values, output)
            control_equal.append(torch.equal(expected, output))
        report["control_elementwise_equal_per_layer"] = control_equal
        save()
        if not all(control_equal):
            raise RuntimeError("same-library vision control differs from captured deployment output")

        report["phase"] = "candidate_correctness"
        save()
        metrics = []
        for values, expected in calls:
            candidate(*values, output)
            metrics.append(error_metrics(expected, output))
        limit = tolerances()["shallow"]
        valid = all(row["rel_rms"] <= limit["rel_rms_max"]
                    and row["cosine_similarity"] >= limit["cosine_min"] for row in metrics)
        report.update(tolerance=limit, candidate_correct=valid, per_layer_metrics=metrics)
        print(json.dumps({"candidate_correct": valid,
                          "worst_rel_rms": max(row["rel_rms"] for row in metrics),
                          "lowest_cosine": min(row["cosine_similarity"] for row in metrics)}), flush=True)
        save()
        if not valid:
            raise RuntimeError("rounded vision GELU numerical mismatch")

        # Both routes use the same normalized/input/output addresses; plans are warm.
        scratch.freeze()
        controls = [lambda values=values: control(*values, output) for values, _ in calls]
        candidates = [lambda values=values: candidate(*values, output) for values, _ in calls]
        report["phase"] = "full_chain_abba"
        for route, functions in (("A", controls), ("B", candidates),
                                 ("B", candidates), ("A", controls)):
            samples = samples_ms(functions, lambda: None, reps=15)
            leg = {"route": route, "samples_ms_per_call": samples,
                   "median_us_per_call": statistics.median(samples) * 1000,
                   "median_ms_all_layers": statistics.median(samples) * len(calls)}
            report["legs"].append(leg)
            print(json.dumps(leg), flush=True)
            save()
        a = [leg["median_us_per_call"] for leg in report["legs"] if leg["route"] == "A"]
        b = [leg["median_us_per_call"] for leg in report["legs"] if leg["route"] == "B"]
        report.update(mean_leg_delta_us=statistics.mean(a) - statistics.mean(b),
                      conservative_leg_delta_us=min(a) - max(b),
                      control_drift_us=abs(a[1] - a[0]), candidate_drift_us=abs(b[1] - b[0]))
        report["phase"] = "complete" if report["mean_leg_delta_us"] > 0 else "complete_no_local_gain"
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
