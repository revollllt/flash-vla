"""Fixed split-factor8 CUTLASS FFN-down screen.

Apply cutlass_split8_down.patch first; control and candidate must share the one
updated native library. See cutlass_split8_down.md for the bounded reproduction.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import subprocess
import weakref

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import (
    cutlass_backbone, cutlass_expert_residual, fused_ffn)
from flash_vla.inference import build, resolve
from flash_vla.runtime.runner import Scratch
from measurement.cli import parse_options
from lab.pi05.cutlass_gemm_screen import samples_ms

SITE = "action_expert_ffn_down_residual"


class _Plan:
    def __init__(self, library, scratch, role, x, weight, gate, out):
        size = library.expert_split8_down_workspace()
        self.workspace = scratch(role, (size,), torch.uint8, x.device)
        self.tensors = (x, weight, gate, out)
        self.handle = ctypes.c_void_p()
        cutlass_backbone.check(library.expert_split8_down_plan(
            x.data_ptr(), weight.data_ptr(), gate.data_ptr(), out.data_ptr(),
            self.workspace.data_ptr(), torch.cuda.current_stream().cuda_stream,
            ctypes.byref(self.handle)), "expert_split8_down_plan")
        self.destroy = weakref.finalize(self, library.expert_down_destroy, self.handle)


def make_candidate(library, scratch):
    library.expert_split8_down_workspace.argtypes = []
    library.expert_split8_down_workspace.restype = ctypes.c_int64
    library.expert_split8_down_plan.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.POINTER(ctypes.c_void_p)])
    library.expert_split8_down_plan.restype = ctypes.c_int32
    library.expert_down_run.argtypes = [ctypes.c_void_p] * 2
    library.expert_down_run.restype = ctypes.c_int32
    library.expert_down_destroy.argtypes = [ctypes.c_void_p]
    library.expert_down_destroy.restype = None
    plans = {}
    role = f"lab_split8_down_streamk_{id(plans)}"

    def invoke(x, weight, gate, out):
        key = (x.data_ptr(), weight.data_ptr(), gate.data_ptr(), out.data_ptr())
        plan = plans.get(key)
        if plan is None:
            plan = _Plan(library, scratch, role, x, weight, gate, out)
            plans[key] = plan
        cutlass_backbone.check(library.expert_down_run(
            plan.handle, torch.cuda.current_stream().cuda_stream), "expert_down_run")
        return out
    return invoke


def record_calls(engine, inputs):
    calls = []

    def instrument(name, function):
        if name != SITE:
            return function

        def invoke(x, weight, gate, out):
            call = {"x": x.clone(), "weight": weight, "gate": gate.clone(),
                    "residual": out.clone()}
            result = function(x, weight, gate, out)
            call["expected"] = out.clone()
            calls.append(call)
            return result
        return invoke

    engine.stage(**inputs)
    with engine.instrument(instrument):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "action_expert":
                engine.run_eager(step.name)
            else:
                engine.replay(step.name)
    torch.cuda.synchronize()
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("coverage", "actual"), required=True)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": args.seed, "options": args.option, "site": SITE,
        "shape_mkn": [50, 4096, 1024], "tile": [32, 64, 32],
        "warp": [32, 32, 32], "stages": 8, "split_factor": 8,
        "phase": "native_load", "coverage": [], "correctness": [],
        "decomposition": [], "legs": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        library = cutlass_backbone.library()
        scratch = Scratch(torch.device("cuda"))
        candidate = make_candidate(library, scratch)
        report.update(native_library=library._name,
                      control_workspace_bytes=library.expert_down_workspace(4096),
                      candidate_workspace_bytes=library.expert_split8_down_workspace())
        if args.phase == "coverage":
            report["phase"] = "coverage"
            x = torch.empty((50, 4096), device="cuda", dtype=torch.bfloat16)
            weight = torch.empty((4096, 1024), device=x.device, dtype=x.dtype)
            gate = torch.empty((1024,), device=x.device, dtype=x.dtype)
            out = torch.empty((50, 1024), device=x.device, dtype=x.dtype)
            for name, input_value, gate_value, residual in (
                ("residual_only", 0, 0, 3), ("ones_product", 1, 1, 0),
                ("column_gate", 1, None, 0),
            ):
                x.fill_(input_value)
                weight.fill_(input_value)
                if gate_value is None:
                    gate.copy_(torch.arange(1024, device=x.device, dtype=x.dtype) / 1024)
                else:
                    gate.fill_(gate_value)
                out.fill_(residual)
                # The constant GEMM is exactly 0 or 4096; no reference GEMM required.
                expected = ((4096 * input_value) * gate.float() + residual).to(x.dtype)
                expected = expected.expand_as(out)
                candidate(x, weight, gate, out)
                row = {"case": name, "exact": torch.equal(expected, out),
                       "mismatches_by_row": (expected != out).sum(1).tolist(),
                       "output": error_metrics(expected, out)}
                report["coverage"].append(row)
                save()
                print(json.dumps(row), flush=True)
                if not row["exact"]:
                    raise RuntimeError(f"split8 down constant coverage failed: {name}")

            # One warm launch per type, then one profiler launch per type solely
            # to obtain real grid/register/shared metadata. Durations are not timed evidence.
            control = cutlass_expert_residual.make_wrappers(scratch, [SITE])[SITE]
            for function in (control, candidate):
                out.zero_()
                function(x, weight, gate, out)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                for function in (control, candidate):
                    out.zero_()
                    function(x, weight, gate, out)
                torch.cuda.synchronize()
            trace = args.output.with_suffix(".trace.json")
            profile.export_chrome_trace(str(trace))
            kernels = [event for event in json.loads(trace.read_text())["traceEvents"]
                       if event.get("cat") == "kernel" and "DownEpilogue" in event.get("name", "")]
            report["kernel_resources"] = [
                {"name": event["name"], "args": event["args"]} for event in kernels]
            report["resource_trace"] = str(trace)
            report["phase"] = "coverage_complete"
            save()
            return

        report["phase"] = "record_calls"
        save()
        engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                       **parse_options(args.option))
        report["deployment_plan"] = dict(engine.identity.plan)
        calls = record_calls(engine, engine.sample_inputs(args.seed))
        weights = {call["weight"].data_ptr(): call["weight"] for call in calls}
        report.update(calls=len(calls), distinct_layer_weights=len(weights),
                      distinct_weight_bytes=sum(w.numel() * w.element_size()
                                                for w in weights.values()),
                      x_stride=list(calls[0]["x"].stride()),
                      weight_stride=list(calls[0]["weight"].stride()),
                      gate_shape=list(calls[0]["gate"].shape),
                      residual_stride=list(calls[0]["residual"].stride()),
                      timer="reset-inclusive CUDA graph, one ABBA with 15 samples per leg",
                      is_deployment_sequence=False)
        control = engine.ops[SITE]
        out = torch.empty_like(calls[0]["residual"])

        def run(call, function):
            out.copy_(call["residual"])
            function(call["x"], call["weight"], call["gate"], out)
            return out

        limit = tolerances()["shallow"]
        report["phase"] = "actual_correctness"
        for index, call in enumerate(calls):
            metrics = {}
            for route, function in (("A", control), ("B", candidate)):
                run(call, function)
                metrics[route] = error_metrics(call["expected"], out)
                metrics[route]["exact"] = torch.equal(call["expected"], out)
            valid = all(m["rel_rms"] <= limit["rel_rms_max"]
                        and m["cosine_similarity"] >= limit["cosine_min"]
                        for m in metrics.values())
            report["correctness"].append({"call": index, "output": metrics, "correct": valid})
            save()
            if not valid:
                raise RuntimeError(f"split8 down shallow check failed at call {index}")

        native = fused_ffn.library()
        ones = torch.ones_like(calls[0]["gate"])
        projection = torch.empty_like(out)
        report["phase"] = "decomposition"
        for index in (0, len(calls) // 2, len(calls) - 1):
            call = calls[index]
            projection.zero_()
            candidate(call["x"], call["weight"], ones, projection)
            out.copy_(call["residual"])
            fused_ffn.check(native.gated_residual_launch(
                projection.data_ptr(), call["gate"].data_ptr(), out.data_ptr(), 50,
                torch.cuda.current_stream().cuda_stream), "gated_residual", 50)
            expected = out.clone()
            run(call, candidate)
            row = {"call": index, "exact": torch.equal(expected, out),
                   "output": error_metrics(expected, out)}
            report["decomposition"].append(row)
            save()
            if not row["exact"]:
                raise RuntimeError(f"split8 down rounding decomposition failed at call {index}")
        scratch.freeze()
        print(json.dumps({"correct_calls": len(calls), "max_rel_rms": max(
            row["output"]["B"]["rel_rms"] for row in report["correctness"]),
            "decomposition": report["decomposition"]}), flush=True)
        functions = {"A": [lambda c=c: run(c, control) for c in calls],
                     "B": [lambda c=c: run(c, candidate) for c in calls]}
        report["phase"] = "abba"
        for route in ("A", "B", "B", "A"):
            samples = samples_ms(functions[route], lambda: None, reps=15)
            leg = {"route": route, "samples_ms_per_call": samples,
                   "median_us_per_call": statistics.median(samples) * 1000}
            report["legs"].append(leg)
            print(json.dumps(leg), flush=True)
            save()
        a = [leg["median_us_per_call"] for leg in report["legs"] if leg["route"] == "A"]
        b = [leg["median_us_per_call"] for leg in report["legs"] if leg["route"] == "B"]
        gain = statistics.mean(a) - statistics.mean(b)
        drift_a, drift_b = abs(a[1] - a[0]), abs(b[1] - b[0])
        report.update(mean_leg_gain_us=gain, control_drift_us=drift_a,
                      candidate_drift_us=drift_b, conservative_gain_us=min(a) - max(b),
                      local_gain_exceeds_drift=gain > max(drift_a, drift_b))
        report["phase"] = "complete"
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
