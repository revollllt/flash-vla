"""One existing cfg0 reuse on all 18 actual Pi0.5 prefix QKV layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import (
    cutlass_backbone, fused_backbone, fused_prefix_qkv)
from flash_vla.inference import build, resolve
from flash_vla.runtime.runner import Scratch
from measurement.cli import parse_options
from lab.pi05.cutlass_gemm_screen import samples_ms

SITE = "llm_backbone_norm_qkv_rope"
OUTPUTS = ("Q", "K", "V", "x_norm")


def record_calls(engine, inputs):
    calls = []

    def instrument(name, function):
        if name != SITE:
            return function

        def invoke(x, weight, rope, Q, K, V, x_norm):
            saved = {"x": x.clone(), "weight": weight, "rope": rope.clone()}
            result = function(x, weight, rope, Q, K, V, x_norm)
            saved["expected"] = tuple(t.clone() for t in (Q, K, V, x_norm[:x.shape[0]]))
            calls.append(saved)
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
    assert len(calls) == 18, len(calls)
    working = tuple(torch.empty_like(t) for t in calls[0]["expected"])
    q, k, v, normed = working
    rows = calls[0]["x"].shape[0]
    projected = torch.empty((rows, 2560), dtype=normed.dtype, device=normed.device)
    scratch = Scratch(normed.device)
    # This factory requests only its projected buffer. Both routes write the
    # exact same allocation, while native plans keep their usual own scratch.
    def projection_scratch(role, shape, dtype, device):
        return projected

    control = fused_prefix_qkv.make_wrappers(projection_scratch)[SITE]
    gemm_library = cutlass_backbone.library()
    norm_library = fused_backbone.library()
    rope_library = fused_prefix_qkv.library()
    stream = torch.cuda.current_stream().cuda_stream
    plans = [cutlass_backbone._Plan(
        gemm_library, scratch, normed, call["weight"], projected, 0.0, stream)
        for call in calls]

    def candidate(call, plan):
        stream = torch.cuda.current_stream().cuda_stream
        cutlass_backbone.check(norm_library.backbone_rms_norm(
            call["x"].data_ptr(), normed.data_ptr(), rows, stream), "backbone_rms_norm")
        cutlass_backbone.check(
            gemm_library.backbone_gemm_run(plan.handle, stream), "backbone_gemm_run")
        cutlass_backbone.check(rope_library.prefix_rope_scatter(
            projected.data_ptr(), call["rope"].data_ptr(), q.data_ptr(), k.data_ptr(),
            v.data_ptr(), rows, stream), "prefix_rope_scatter")

    functions = {
        "A": [lambda call=call: control(
            call["x"], call["weight"], call["rope"], *working) for call in calls],
        "B": [lambda call=call, plan=plan: candidate(call, plan)
              for call, plan in zip(calls, plans)],
    }
    report = {
        "source_revision": revision, "seed": args.seed, "options": args.option,
        "identity": engine.identity.as_dict(), "site": SITE,
        "shape_mkn": [rows, 2048, 2560], "calls": len(calls), "alpha": 1, "beta": 0,
        "candidate_config": 0, "candidate_library": gemm_library._name,
        "weight_bytes": sum(c["weight"].numel() * c["weight"].element_size() for c in calls),
        "projected_bytes": projected.numel() * projected.element_size(),
        "scratch_bytes": scratch.nbytes,
        "same_addresses": "both routes share normed, BF16 projected, Q, K, V and each immutable input",
        "scope": "complete unchanged norm -> selected GEMM -> unchanged RoPE/scatter",
        "cache_policy": "all 18 real layer weights rotate (180 MiB); no flush; not full-model traffic",
        "timer": "one ABBA; 18 distinct calls per graph; 15 measured replays per leg, first retained",
        "correctness": [], "legs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    limit = tolerances()["shallow"]
    for layer, call in enumerate(calls):
        row = {"layer": layer}
        accepted = True
        for route in ("A", "B"):
            functions[route][layer]()
            outputs = {}
            for name, expected, actual in zip(OUTPUTS, call["expected"], working):
                exact = torch.equal(expected, actual)
                result = {"exact": exact}
                if not exact:
                    metrics = error_metrics(expected, actual)
                    result["metrics"] = metrics
                    passed = (metrics["rel_rms"] <= limit["rel_rms_max"]
                              and metrics["cosine_similarity"] >= limit["cosine_min"])
                    accepted = accepted and passed
                    result["passed"] = passed
                outputs[name] = result
            row[route] = outputs
        report["correctness"].append(row)
        report["status"] = "checking_numerics" if accepted else "numerical_failed"
        save()
        print(json.dumps(row), flush=True)
        assert accepted, row
    report["status"] = "numerics_passed"
    save()
    scratch.freeze()

    for label, route in (("A1", "A"), ("B1", "B"), ("B2", "B"), ("A2", "A")):
        samples = samples_ms(functions[route], lambda: None, reps=15)
        row = {"leg": label, "median_ms_total": statistics.median(samples) * len(calls),
               "samples_ms_per_call": samples}
        report["legs"].append(row)
        save()
        print(json.dumps(row), flush=True)
    a1, b1, b2, a2 = [row["median_ms_total"] for row in report["legs"]]
    report["summary"] = {
        "mean_gain_ms_total": (a1 + a2 - b1 - b2) / 2,
        "conservative_gap_ms_total": min(a1, a2) - max(b1, b2),
        "control_drift_ms_total": abs(a2 - a1),
        "candidate_drift_ms_total": abs(b2 - b1),
    }
    report["status"] = "complete"
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
