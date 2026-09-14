"""Screen existing EpiLinear CUTLASS tiles on real Pi0.5 backbone GEMMs.

This lab-only probe loads a selected checkout's existing Pi0 host interface
to reuse its native build cache. A promising result still needs a component
implementation and deployed validation before Pi0.5 can route through it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090 import RTX5090Spec
from flash_vla.inference import build, parse_options, resolve
from flash_vla.runtime.cuda.graph import StreamGraph


def samples_ms(functions, reset, *, reps=15):
    """Time one graph of distinct real weights on its own capture stream.

    For C=D residuals, restore C before each timed graph so every trial
    starts at the same values. The restore is excluded from both timings.
    """
    graph = StreamGraph()
    graph.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph.stream):
        for _ in range(4):
            reset()
            for function in functions:
                function()
    graph.stream.synchronize()
    with graph.capture():
        for function in functions:
            function()
    graph.stream.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    with torch.cuda.stream(graph.stream):
        for _ in range(reps):
            reset()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / len(functions))
    return samples


def real_calls(engine, inputs, site):
    calls = []

    def record(name, function):
        if site == "gate_up" and name == "llm_backbone_norm_gated_ffn":
            def wrapped(x, gate_w, up_w, out, x_norm):
                result = function(x, gate_w, up_w, out, x_norm)
                if len(calls) < 4:
                    activation = x_norm[:x.shape[0]].clone()
                    calls.extend(((activation, gate_w, None), (activation, up_w, None)))
                return result
            return wrapped
        if site == "down" and name == "llm_backbone_ffn_down_residual":
            def wrapped(x, weight, out):
                if len(calls) < 4:
                    calls.append((x.clone(), weight, out.clone()))
                return function(x, weight, out)
            return wrapped
        return function

    engine.stage(**inputs)
    with engine.instrument(record):
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


def screen(cg, engine, inputs, site, configs):
    calls = real_calls(engine, inputs, site)
    outputs = [torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
               for a, b, _ in calls]

    def reset():
        for (_, _, residual), output in zip(calls, outputs):
            if residual is not None:
                output.copy_(residual)

    reference_functions = []
    for (a, b, residual), output in zip(calls, outputs):
        if residual is None:
            reference_functions.append(lambda a=a, b=b, output=output:
                                       torch.mm(a, b, out=output))
        else:
            reference_functions.append(lambda a=a, b=b, output=output:
                                       torch.addmm(output, a, b, beta=1, alpha=1, out=output))
    reset()
    for function in reference_functions:
        function()
    expected = [output.clone() for output in outputs]
    rows = []

    def report(route, samples, **extra):
        row = {"route": route, "samples_ms": samples,
               "median_ms": statistics.median(samples), **extra}
        rows.append(row)
        print(json.dumps({"site": site, **row}), flush=True)

    report("torch_before", samples_ms(reference_functions, reset))
    for config, geometry in configs.items():
        try:
            plans = [cg.plan(a, b, output, config=config,
                             c=output if residual is not None else None,
                             beta=1.0 if residual is not None else 0.0)
                     for (a, b, residual), output in zip(calls, outputs)]
            functions = [lambda plan=plan: cg.run(plan) for plan in plans]
            reset()
            for function in functions:
                function()
            # Both ends of the four-weight cycle exercise distinct real values.
            metrics = [error_metrics(expected[index], outputs[index]) for index in (0, 3)]
            limit = tolerances()["shallow"]
            valid = all(metric["rel_rms"] <= limit["rel_rms_max"]
                        and metric["cosine_similarity"] >= limit["cosine_min"]
                        for metric in metrics)
            if valid:
                report("cutlass", samples_ms(functions, reset),
                       config=config, geometry=geometry, metrics=metrics)
            else:
                row = {"route": "cutlass", "config": config,
                       "status": "numerical_mismatch", "metrics": metrics}
                rows.append(row)
                print(json.dumps({"site": site, **row}), flush=True)
        except cg.Unsupported as error:
            row = {"route": "cutlass", "config": config,
                   "status": "unsupported", "reason": str(error)}
            rows.append(row)
            print(json.dumps({"site": site, **row}), flush=True)
        finally:
            torch.cuda.synchronize()
            for handle, workspace in cg._PLANS.values():
                cg._check(cg.library().cutlass_gemm_destroy(handle), "destroy")
            cg._PLANS.clear()
    report("torch_after", samples_ms(reference_functions, reset))
    return {
        "site": site, "shape_mkn": [calls[0][0].shape[0], *calls[0][1].shape],
        "dtype": str(calls[0][0].dtype), "alpha": 1,
        "beta": 1 if site == "down" else 0, "c_aliases_d": site == "down",
        "weight_rotation_bytes": sum(b.numel() * b.element_size() for _, b, _ in calls),
        "l2_bytes": RTX5090Spec.L2_CACHE_SIZE_BYTES, "reps": 15, "calls_per_graph": len(calls),
        "timer": "CUDA events on capture stream; residual reset outside timing",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--site", choices=("gate_up", "down", "both"), default="gate_up")
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host = args.source_checkout / "src/flash_vla/hardware/nvidia/rtx5090/pi0/backends/cuda/cutlass_gemm.py"
    spec = importlib.util.spec_from_file_location("cutlass_screen_host", host)
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    source = host.parent / "kernels/cutlass_gemm.cu"
    configs = {}
    for entry in re.findall(r"X\(([^)]*EpiLinear)\)", source.read_text()):
        values = [int(value.strip()) for value in entry.split(",")[:-1]]
        configs[values[0]] = {"tile": values[1:4], "warp": values[4:7], "stages": values[7]}
    cg.set_pdl(False)
    print(json.dumps({"native_library": cg.library()._name, "configs": configs}), flush=True)
    engine = build(resolve("rtx5090/pi05"), "reference", seed=args.seed,
                   **parse_options(args.option))
    inputs = engine.sample_inputs(args.seed)
    results = []
    for site in (("gate_up", "down") if args.site == "both" else (args.site,)):
        results.append(screen(cg, engine, inputs, site, configs))
        args.output.write_text(json.dumps({
            "seed": args.seed, "options": args.option, "native_source": str(source),
            "native_library": cg.library()._name, "native_arch": cg._ARCH,
            "pdl": False, "results": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
