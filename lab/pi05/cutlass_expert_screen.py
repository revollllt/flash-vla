"""Screen linear CUTLASS GEMMs against current Pi0.5 expert torch.mm calls.

Capture the first use of each real weight from the deployed packed FFN and
down projection. The latter's gate/residual update follows its BF16 GEMM, so
both screens use alpha=1, beta=0. This file never changes production routing.
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
        "action_expert_norm_gated_ffn": "packed_ffn",
        "action_expert_ffn_down_residual": "down",
    }
    calls = {label: [] for label in sites.values()}
    seen = {label: set() for label in sites.values()}
    original_mm = torch.mm

    def wrap(name, function):
        if name not in sites:
            return function
        label = sites[name]

        def capture_mm(a, b, **kwargs):
            if b.data_ptr() not in seen[label]:
                seen[label].add(b.data_ptr())
                calls[label].append((a.clone(), b))
            return original_mm(a, b, **kwargs)

        def invoke(*args, **kwargs):
            # Patching is confined to an eager selected call and never timed.
            with patch.object(torch, "mm", capture_mm):
                return function(*args, **kwargs)
        return invoke

    engine.stage(**inputs)
    with engine.instrument(wrap):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "action_expert":
                engine.run_eager(step.name)
            else:
                engine.replay(step.name)
    torch.cuda.synchronize()
    return calls


def screen(cg, label, calls, configs, samples_ms, l2_bytes):
    if not calls:
        raise RuntimeError(f"no torch.mm calls recorded for {label}")
    weight_bytes = sum(b.numel() * b.element_size() for _, b in calls)
    if weight_bytes <= l2_bytes:
        raise RuntimeError(f"{label} weight cycle {weight_bytes} does not exceed L2 {l2_bytes}")
    outputs = [torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
               for a, b in calls]
    controls = [lambda a=a, b=b, output=output: torch.mm(a, b, out=output)
                for (a, b), output in zip(calls, outputs)]
    for function in controls:
        function()
    expected = torch.cat(outputs)
    rows = []

    def report(route, samples, **extra):
        row = {"route": route, "samples_ms": samples,
               "median_ms": statistics.median(samples), **extra}
        rows.append(row)
        print(json.dumps({"site": label, **row}), flush=True)

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
                plans = [cb._Plan(native, Scratch(a.device), a, b, output, 0.0, cg._stream())
                         for (a, b), output in zip(calls, outputs)]
                functions = [
                    lambda plan=plan: cb._check(
                        native.backbone_gemm_run(plan.handle, cg._stream()), "cfg0 run")
                    for plan in plans]
                native_path = native._name
            else:
                plans = [cg.plan(a, b, output, config=config, beta=0.0)
                         for (a, b), output in zip(calls, outputs)]
                functions = [lambda plan=plan: cg.run(plan) for plan in plans]
                native_path = cg.library()._name
            for function in functions:
                function()
            metrics = error_metrics(expected, torch.cat(outputs))
            limit = tolerances()["shallow"]
            valid = (metrics["rel_rms"] <= limit["rel_rms_max"]
                     and metrics["cosine_similarity"] >= limit["cosine_min"])
            if valid:
                report("cutlass", samples_ms(functions, lambda: None, reps=15),
                       config=config, geometry=geometry, metrics=metrics, native_library=native_path)
            else:
                row = {"route": "cutlass", "config": config,
                       "status": "numerical_mismatch", "metrics": metrics}
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
        "dtype": str(calls[0][0].dtype), "alpha": 1, "beta": 0,
        "input_source": "first deployed eager torch.mm for each distinct real weight",
        "weight_rotation_bytes": weight_bytes, "l2_bytes": l2_bytes,
        "calls_per_graph": len(calls), "reps": 15, "workspace": "one workspace per plan",
        "timer": "CUDA graph events on its capture stream", "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"], text=True).strip()
    # The probe lives in an isolated worktree; inputs must come from the
    # selected deployment checkout, while its timing helper stays local.
    sys.path.insert(0, str(args.source_checkout / "src"))
    from flash_vla.hardware.nvidia.rtx5090 import RTX5090Spec
    from flash_vla.inference import build, parse_options, resolve
    from lab.pi05.cutlass_gemm_screen import samples_ms

    host = args.source_checkout / "src/flash_vla/hardware/nvidia/rtx5090/pi0/backends/cuda/cutlass_gemm.py"
    spec = importlib.util.spec_from_file_location("cutlass_expert_screen_host", host)
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
    for label in ("packed_ffn", "down"):
        results.append(screen(cg, label, calls[label], configs, samples_ms,
                              RTX5090Spec.L2_CACHE_SIZE_BYTES))
        args.output.write_text(json.dumps({
            "seed": args.seed, "options": args.option, "deployment_revision_before_import": revision,
            "deployment_plan": dict(engine.identity.plan),
            "native_source": str(source), "native_library": cg.library()._name,
            "native_arch": cg._ARCH, "pdl": False, "results": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
