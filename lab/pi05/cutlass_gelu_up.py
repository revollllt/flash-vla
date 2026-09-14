"""Rejected fixed-cfg0 screen: separate up/GELU versus rounded GELU epilogue.

Apply cutlass_gelu_up.patch first; see cutlass_gelu_up.md. Both variants must
live in the same updated Target library. No production route uses this probe.
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
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_backbone, fused_backbone
from flash_vla.inference import build, parse_options, resolve
from flash_vla.runtime.runner import Scratch
from lab.pi05.cutlass_gemm_screen import samples_ms

SITE = "llm_backbone_norm_gated_ffn"


def record_calls(engine, inputs):
    calls = []

    def instrument(name, function):
        if name != SITE:
            return function

        def invoke(x, gate_w, up_w, out, x_norm):
            result = function(x, gate_w, up_w, out, x_norm)
            rows = x.shape[0]
            calls.append((x_norm[:rows].clone(), gate_w, up_w, out[:rows].clone()))
            return result

        return invoke

    engine.stage(**inputs)
    with engine.instrument(instrument):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "llm_backbone":
                engine.run_eager(step.name)
                break
            else:
                engine.replay(step.name)
    torch.cuda.synchronize()
    return calls


class _FusedUpPlan:
    def __init__(self, library, scratch, a, weight, gate, out):
        m, k = a.shape
        n = weight.shape[1]
        size = library.backbone_gelu_up_workspace(m, k, n)
        self.workspace = scratch("backbone_gelu_up_workspace", (max(size, 1),),
                                 torch.uint8, a.device)
        self.tensors = (a, weight, gate, out)
        self.library = library
        self.handle = ctypes.c_void_p()
        cutlass_backbone._check(library.backbone_gelu_up_plan(
            m, k, n, a.data_ptr(), weight.data_ptr(), gate.data_ptr(), out.data_ptr(),
            self.workspace.data_ptr(), torch.cuda.current_stream().cuda_stream,
            ctypes.byref(self.handle)), "backbone_gelu_up_plan")
        self.destroy = weakref.finalize(self, library.backbone_gelu_up_destroy, self.handle)

    def run(self):
        cutlass_backbone._check(self.library.backbone_gelu_up_run(
            self.handle, torch.cuda.current_stream().cuda_stream), "backbone_gelu_up_run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                   **parse_options(args.option))
    calls = record_calls(engine, engine.sample_inputs(args.seed))
    # Both GEMM variants live in this one updated library. The other library
    # contains only the already deployed pointwise control kernels.
    library = cutlass_backbone._library()
    pointwise = fused_backbone._library()
    scratch = Scratch(calls[0][0].device)
    out = scratch("gelu_up_result", calls[0][-1].shape, calls[0][0].dtype, calls[0][0].device)
    gate_plans, up_plans, fused_plans, gates = [], [], [], []

    def linear(plan):
        cutlass_backbone._check(library.backbone_gemm_run(
            plan.handle, torch.cuda.current_stream().cuda_stream), "backbone_gemm_run")

    def control(plan, gate):
        linear(plan)
        cutlass_backbone._check(pointwise.backbone_gelu_mul(
            gate.data_ptr(), out.data_ptr(), out.numel(),
            torch.cuda.current_stream().cuda_stream), "backbone_gelu_mul")

    report = {"source_revision": revision, "seed": args.seed, "options": args.option,
              "base_plan": dict(engine.identity.plan), "calls": len(calls),
              "shape_mkn": [*calls[0][0].shape, calls[0][2].shape[1]],
              "native_library": library._name, "pointwise_library": pointwise._name,
              "scope": "up GEMM plus GELU/product; RMS and gate GEMM precomputed for both",
              "timer": "CUDA graph on capture stream; both routes fully overwrite the same out",
              "per_layer": [], "legs": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    limit = tolerances()["shallow"]
    for index, (a, gate_w, up_w, expected) in enumerate(calls):
        gate = scratch(f"gelu_up_gate_{index}", out.shape, a.dtype, a.device)
        stream = torch.cuda.current_stream().cuda_stream
        gate_plan = cutlass_backbone._Plan(library, scratch, a, gate_w, gate, 0.0, stream)
        gate_plans.append(gate_plan)
        linear(gate_plan)
        up_plan = cutlass_backbone._Plan(library, scratch, a, up_w, out, 0.0, stream)
        fused_plan = _FusedUpPlan(library, scratch, a, up_w, gate, out)
        up_plans.append(up_plan)
        fused_plans.append(fused_plan)
        gates.append(gate)
        control(up_plan, gate)
        control_metrics = error_metrics(expected, out)
        fused_plan.run()
        candidate_metrics = error_metrics(expected, out)
        valid = all(row["rel_rms"] <= limit["rel_rms_max"]
                    and row["cosine_similarity"] >= limit["cosine_min"]
                    for row in (control_metrics, candidate_metrics))
        report["per_layer"].append({"layer": index, "control_vs_actual": control_metrics,
                                    "candidate_vs_actual": candidate_metrics, "correct": valid})
        save()
        if not valid:
            raise RuntimeError(f"GELU epilogue shallow check failed at layer {index}")

    report["linear_workspace_bytes"] = up_plans[0].workspace.numel()
    report["fused_workspace_bytes"] = fused_plans[0].workspace.numel()
    report["materialized_gate_bytes"] = sum(g.numel() * g.element_size() for g in gates)
    report["up_weight_bytes"] = sum(w.numel() * w.element_size() for _, _, w, _ in calls)
    scratch.freeze()
    functions = {"A": [lambda p=p, g=g: control(p, g) for p, g in zip(up_plans, gates)],
                 "B": [p.run for p in fused_plans]}
    print(json.dumps({"correct_layers": len(calls), "max_rel_rms": max(
        row["candidate_vs_actual"]["rel_rms"] for row in report["per_layer"])}), flush=True)
    for route in ("A", "B", "B", "A"):
        samples = samples_ms(functions[route], lambda: None, reps=15)
        leg = {"route": route, "samples_ms_per_call": samples,
               "median_ms_total": statistics.median(samples) * len(calls)}
        report["legs"].append(leg)
        print(json.dumps(leg), flush=True)
        save()


if __name__ == "__main__":
    main()
