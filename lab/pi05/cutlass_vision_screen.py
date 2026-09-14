"""Screen bias-fused linear CUTLASS GEMMs on actual Pi0.5 vision FFN calls.

Configs 1-11 broadcast BF16 bias with ldc=0 into the FP32 epilogue. The
existing deployed config-0 ABI fixes C=D, so its timed path copies bias into
D on every invocation before beta=1 GEMM. Both round only after acc+bias.
This lab probe never changes production routing or loads a second cfg0 kernel.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from unittest.mock import patch

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances


def record_calls(engine, inputs):
    sites = {
        "vision_encoder_norm_ffn_up": "up",
        "vision_encoder_ffn_down_residual": "down",
    }
    calls = {label: [] for label in sites.values()}
    seen = {label: set() for label in sites.values()}
    original_addmm = torch.addmm

    def wrap(name, function):
        if name not in sites:
            return function
        label = sites[name]

        def capture_addmm(bias, a, b, **kwargs):
            if b.data_ptr() not in seen[label]:
                seen[label].add(b.data_ptr())
                calls[label].append((a.clone(), b, bias))
            return original_addmm(bias, a, b, **kwargs)

        def invoke(*args, **kwargs):
            # Patching is confined to an eager selected call and never timed.
            with patch.object(torch, "addmm", capture_addmm):
                return function(*args, **kwargs)
        return invoke

    engine.stage(**inputs)
    with engine.instrument(wrap):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "vision_encoder":
                engine.run_eager(step.name)
                break
            else:
                engine.replay(step.name)
    torch.cuda.synchronize()
    return calls


def screen(cg, label, calls, configs, samples_ms, l2_bytes):
    if not calls:
        raise RuntimeError(f"no torch.addmm calls recorded for {label}")
    weight_bytes = sum(b.numel() * b.element_size() for _, b, _ in calls)
    if weight_bytes <= l2_bytes:
        raise RuntimeError(f"{label} weight cycle {weight_bytes} does not exceed L2 {l2_bytes}")
    outputs = [torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
               for a, b, _ in calls]
    controls = [lambda a=a, b=b, bias=bias, output=output:
                torch.addmm(bias, a, b, out=output)
                for (a, b, bias), output in zip(calls, outputs)]
    for function in controls:
        function()
    expected = [output.clone() for output in outputs]
    rows = []

    def report(route, samples, **extra):
        row = {"route": route, "samples_ms": samples,
               "median_ms": statistics.median(samples), **extra}
        rows.append(row)
        print(json.dumps({"site": label, **{
            key: value for key, value in row.items() if key != "per_layer_metrics"}}), flush=True)

    report("torch_before", samples_ms(controls, lambda: None, reps=15))
    for config, geometry in configs.items():
        try:
            if config == 0:
                # Identical CUTLASS TLS is GNU-unique across these two DSOs.
                # Use the deployed cfg0 library so its launch attributes and
                # cached device ordinal belong to the same kernel entry.
                from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_backbone as cb
                from flash_vla.runtime.runner import Scratch
                native = cb._library()
                plans = [cb._Plan(native, Scratch(a.device), a, b, output, 1.0, cg._stream())
                         for (a, b, _), output in zip(calls, outputs)]
                functions = []
                for plan, (_, _, bias), output in zip(plans, calls, outputs):
                    def invoke(plan=plan, bias=bias, output=output):
                        # C=D requires this reset inside every timed invocation.
                        output.copy_(bias)
                        cb._check(native.backbone_gemm_run(plan.handle, cg._stream()),
                                  "cfg0 run")
                    functions.append(invoke)
                native_path = native._name
                bias_path = "timed broadcast copy into C=D, then beta=1 GEMM"
                copy_bytes = sum(output.numel() * output.element_size() for output in outputs)
            else:
                plans = [cg.plan(a, b, output, config=config, c=bias, beta=1.0,
                                 broadcast_c=True)
                         for (a, b, bias), output in zip(calls, outputs)]
                functions = [lambda plan=plan: cg.run(plan) for plan in plans]
                native_path = cg.library()._name
                bias_path = "beta=1, ldc=0 broadcast in FP32 epilogue"
                copy_bytes = 0
            for function in functions:
                function()
            # The combined 27-layer up tensor exceeds torch.quantile's limit.
            per_layer_metrics = [error_metrics(reference, output)
                                 for reference, output in zip(expected, outputs)]
            metrics = {name: (min if name == "cosine_similarity" else max)(
                layer[name] for layer in per_layer_metrics) for name in per_layer_metrics[0]}
            limit = tolerances()["shallow"]
            valid = (metrics["rel_rms"] <= limit["rel_rms_max"]
                     and metrics["cosine_similarity"] >= limit["cosine_min"])
            if valid:
                report("cutlass", samples_ms(functions, lambda: None, reps=15),
                       config=config, geometry=geometry, metrics=metrics, native_library=native_path,
                       bias_path=bias_path, timed_bias_copy_write_bytes=copy_bytes,
                       expanded_c_logical_read_bytes=copy_bytes,
                       per_layer_metrics=per_layer_metrics)
            else:
                row = {"route": "cutlass", "config": config,
                       "status": "numerical_mismatch", "metrics": metrics,
                       "per_layer_metrics": per_layer_metrics}
                rows.append(row)
                print(json.dumps({"site": label, **row}), flush=True)
        except cg.Unsupported as error:
            row = {"route": "cutlass", "config": config,
                   "status": "unsupported", "reason": str(error)}
            rows.append(row)
            print(json.dumps({"site": label, **row}), flush=True)
        finally:
            torch.cuda.synchronize()
            for handle, workspace in cg._PLANS.values():
                cg._check(cg.library().cutlass_gemm_destroy(handle), "destroy")
            cg._PLANS.clear()
    report("torch_after", samples_ms(controls, lambda: None, reps=15))
    return {
        "site": label, "shape_mkn": [calls[0][0].shape[0], *calls[0][1].shape],
        "dtype": str(calls[0][0].dtype), "bias_dtype": str(calls[0][2].dtype),
        "alpha": 1, "beta": 1,
        "input_source": "first deployed eager torch.addmm for each distinct real weight",
        "weight_rotation_bytes": weight_bytes, "l2_bytes": l2_bytes,
        "calls_per_graph": len(calls), "reps": 15, "workspace": "one workspace per plan",
        "timer": "CUDA graph events on its capture stream",
        "metrics_summary": "worst per-layer value for each metric (minimum cosine)", "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site", choices=("up", "down", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vendor_revision = subprocess.check_output(
        ["git", "-C", str(args.source_checkout / "third_party/cutlass"), "rev-parse", "HEAD"],
        text=True).strip()
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"], text=True).strip()
    # The probe lives in an isolated worktree; inputs must come from the
    # selected deployment checkout, while its timing helper stays local.
    sys.path.insert(0, str(args.source_checkout / "src"))
    from flash_vla.hardware.nvidia.rtx5090 import RTX5090Spec
    from flash_vla.inference import build, parse_options, resolve
    from lab.pi05.cutlass_gemm_screen import samples_ms

    host = args.source_checkout / "src/flash_vla/hardware/nvidia/rtx5090/pi0/backends/cuda/cutlass_gemm.py"
    spec = importlib.util.spec_from_file_location("cutlass_vision_screen_host", host)
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    source = host.parent / "kernels/cutlass_gemm.cu"
    configs = {}
    for entry in re.findall(r"X\(([^)]*EpiLinear)\)", source.read_text()):
        values = [int(value.strip()) for value in entry.split(",")[:-1]]
        configs[values[0]] = {"tile": values[1:4], "warp": values[4:7], "stages": values[7]}
    cg.set_pdl(False)
    print(json.dumps({"native_library": cg.library()._name, "configs": configs}), flush=True)
    engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                   **parse_options(args.option))
    print(json.dumps({"deployment_revision_before_import": revision,
                      "deployment_plan": dict(engine.identity.plan)}), flush=True)
    inputs = engine.sample_inputs(args.seed)
    calls = record_calls(engine, inputs)
    results = []
    for label in (("up", "down") if args.site == "both" else (args.site,)):
        results.append(screen(cg, label, calls[label], configs, samples_ms,
                              RTX5090Spec.L2_CACHE_SIZE_BYTES))
        args.output.write_text(json.dumps({
            "seed": args.seed, "options": args.option, "deployment_revision_before_import": revision,
            "deployment_plan": dict(engine.identity.plan),
            "native_source": str(source), "native_library": cg.library()._name,
            "native_arch": cg._ARCH, "cutlass_vendor_revision": vendor_revision,
            "pdl": False, "results": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
