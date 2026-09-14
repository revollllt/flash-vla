"""Check existing cfg0 on 17 actual backbone output projections, then ABBA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_backbone
from flash_vla.inference import build, parse_options, resolve
from flash_vla.runtime.runner import Scratch
from lab.pi05.cutlass_gemm_screen import samples_ms

SITE = "llm_backbone_out_proj_residual"


def record_calls(engine, inputs):
    calls = []

    def instrument(name, function):
        if name != SITE:
            return function

        def invoke(x, weight, out):
            activation, residual = x.clone(), out.clone()
            result = function(x, weight, out)
            calls.append((activation, weight, residual, torch.empty_like(out), result.clone()))
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
    scratch = Scratch(calls[0][0].device)
    # The existing residual closure already accepts this M,K,N; no new route
    # is required to reproduce the rejected cfg0 reuse candidate.
    candidate = cutlass_backbone.make_wrappers(
        scratch, {"llm_backbone_ffn_down_residual"})["llm_backbone_ffn_down_residual"]
    control = getattr(engine.ops, SITE)

    def reset():
        for _, _, residual, out, _ in calls:
            out.copy_(residual)

    functions = {
        route: [lambda fn=fn, x=x, weight=weight, out=out: fn(x, weight, out)
                for x, weight, _, out, _ in calls]
        for route, fn in (("A", control), ("B", candidate))}
    reset()
    for function in functions["B"]:
        function()
    metrics = [error_metrics(expected, out) for _, _, _, out, expected in calls]
    limit = tolerances()["shallow"]
    valid = all(row["rel_rms"] <= limit["rel_rms_max"]
                and row["cosine_similarity"] >= limit["cosine_min"] for row in metrics)
    report = {"source_revision": revision, "seed": args.seed, "options": args.option,
              "base_plan": dict(engine.identity.plan), "site": SITE, "calls": len(calls),
              "shape_mkn": [*calls[0][0].shape, calls[0][1].shape[1]],
              "dtype": str(calls[0][0].dtype), "alpha": 1, "beta": 1, "c_aliases_d": True,
              "weight_bytes": sum(w.numel() * w.element_size() for _, w, _, _, _ in calls),
              "per_layer_metrics": metrics, "correct": valid, "scratch_bytes": scratch.nbytes,
              "timer": "CUDA graph; same residual reset excluded from both routes", "legs": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(json.dumps({"calls": len(calls), "correct": valid,
                      "worst_rel_rms": max(row["rel_rms"] for row in metrics),
                      "lowest_cosine": min(row["cosine_similarity"] for row in metrics)}), flush=True)
    if not valid:
        raise RuntimeError(f"candidate shallow check failed: {SITE}")
    scratch.freeze()
    for route in ("A", "B", "B", "A"):
        samples = samples_ms(functions[route], reset, reps=15)
        leg = {"route": route, "samples_ms_per_call": samples,
               "median_ms_total": statistics.median(samples) * len(calls)}
        report["legs"].append(leg)
        print(json.dumps(leg), flush=True)
        save()


if __name__ == "__main__":
    main()
