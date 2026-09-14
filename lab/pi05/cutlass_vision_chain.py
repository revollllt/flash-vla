"""Check cfg10 on actual 27-layer vision FFN chains, then time A/B/B/A graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.inference import build, parse_options, resolve
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_vision, fused_vision, torch_ops
from flash_vla.runtime.cuda.graph import StreamGraph
from flash_vla.runtime.runner import Scratch
from lab.pi05.cutlass_gemm_screen import samples_ms


def record_calls(engine, inputs):
    calls = {name: [] for name in cutlass_vision.NAMES}

    def wrap(name, function):
        if name not in calls:
            return function

        def invoke(*args):
            values = list(args)
            values[0] = args[0].clone()
            values[-1] = args[-1].clone()
            initial = values[-1].clone() if name.endswith("down_residual") else None
            result = function(*args)
            calls[name].append((tuple(values), initial, result.clone()))
            return result
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


def identical_replay(functions, reset, outputs):
    graph = StreamGraph()
    graph.stream.wait_stream(torch.cuda.current_stream())
    with graph.capture():
        for function in functions:
            function()
    with torch.cuda.stream(graph.stream):
        reset()
        graph.replay()
        first = [output.clone() for output in outputs]
        reset()
        graph.replay()
        graph.stream.synchronize()
        return all(torch.equal(a, b) for a, b in zip(first, outputs))


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
    device = next(iter(calls.values()))[0][0][0].device
    control_scratch, candidate_scratch = Scratch(device), Scratch(device)
    control = fused_vision.make_wrappers(control_scratch, {"vision_encoder_norm_ffn_up"})
    control["vision_encoder_ffn_down_residual"] = torch_ops.vision_encoder_ffn_down_residual
    candidate = cutlass_vision.make_wrappers(candidate_scratch)
    report = {"source_revision": revision, "seed": args.seed, "options": args.option,
              "base_plan": dict(engine.identity.plan), "order": ["A", "B", "B", "A"],
              "scope": "actual 27-layer full norm/GEMM/GELU and GEMM/residual callsites",
              "sites": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    prepared = []
    for name in ("vision_encoder_norm_ffn_up", "vision_encoder_ffn_down_residual"):
        rows = calls[name]
        outputs = [values[-1] for values, _, _ in rows]

        def reset(rows=rows):
            for (values, initial, _) in rows:
                if initial is not None:
                    values[-1].copy_(initial)

        functions = {
            route: [lambda fn=wrappers[name], values=values: fn(*values)
                    for values, _, _ in rows]
            for route, wrappers in (("A", control), ("B", candidate))}
        reset()
        for function in functions["B"]:
            function()
        metrics = [error_metrics(expected, output)
                   for (_, _, expected), output in zip(rows, outputs)]
        limit = tolerances()["shallow"]
        valid = all(row["rel_rms"] <= limit["rel_rms_max"]
                    and row["cosine_similarity"] >= limit["cosine_min"] for row in metrics)
        site = {"name": name, "calls": len(rows), "per_layer_metrics": metrics,
                "correct": valid, "replay_identical": identical_replay(
                    functions["B"], reset, outputs), "legs": []}
        report["sites"].append(site)
        print(json.dumps({"name": name, "calls": len(rows), "correct": valid,
                          "replay_identical": site["replay_identical"],
                          "worst_rel_rms": max(row["rel_rms"] for row in metrics),
                          "lowest_cosine": min(row["cosine_similarity"] for row in metrics)}),
              flush=True)
        save()
        if not valid or not site["replay_identical"]:
            raise RuntimeError(f"candidate chain check failed: {name}")
        prepared.append((site, functions, reset))

    # Both complete chains must pass before either is timed.
    for _, functions, reset in prepared:
        reset()
        for function in functions["A"]:
            function()
    control_scratch.freeze()
    candidate_scratch.freeze()
    for site, functions, reset in prepared:
        for route in report["order"]:
            samples = samples_ms(functions[route], reset, reps=15)
            leg = {"route": route, "samples_ms_per_call": samples,
                   "median_ms_total": statistics.median(samples) * site["calls"]}
            site["legs"].append(leg)
            print(json.dumps({"name": site["name"], **leg}), flush=True)
            save()


if __name__ == "__main__":
    main()
