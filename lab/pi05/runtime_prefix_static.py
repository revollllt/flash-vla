"""Static M896 upper-bound screen on 51 real cfg0 backbone FFN GEMMs.

No runtime selector, pointwise stages, native changes, or production route.
M968 references replay the captured inputs; they are not original GEMM snapshots.
"""
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
from flash_vla.inference import build, resolve
from flash_vla.runtime.runner import Scratch
from measurement.cli import parse_options
from lab.pi05.cutlass_gemm_screen import samples_ms


def record_calls(engine, inputs):
    calls = []

    def instrument(name, function):
        if name == "llm_backbone_norm_gated_ffn":
            def ffn(x, gate_w, up_w, out, x_norm):
                result = function(x, gate_w, up_w, out, x_norm)
                prepared = x_norm[:x.shape[0]].clone()
                layer = len(calls) // 3
                for site, weight in (("gate", gate_w), ("up", up_w)):
                    calls.append(dict(layer=layer, site=site, a=prepared, weight=weight,
                                      residual=None, actual_output=None))
                return result
            return ffn
        if name == "llm_backbone_ffn_down_residual":
            def down(x, weight, out):
                a, residual = x.clone(), out.clone()
                result = function(x, weight, out)
                calls.append(dict(layer=len(calls) // 3, site="down", a=a, weight=weight,
                                  residual=residual, actual_output=out.clone()))
                return result
            return down
        return function

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        seed=args.seed, options=args.option, phase="build",
        rows={"A": 968, "B": 896}, compare_rows=[895, 896], config=0,
        scope="51 isolated real-input GEMMs; no selector, pointwise, or deployment timing",
        reference="untimed M968 cfg0 replay of actual captured inputs; not a GEMM snapshot",
        timer="15xABBA; graph of 51 GEMMs; identical full-M968 down resets excluded from timing",
        addresses="same per-call A/B input and output addresses; distinct outputs per call",
        correctness=[], legs=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def passes(metrics):
        return (metrics["rel_rms"] <= limit["rel_rms_max"]
                and metrics["cosine_similarity"] >= limit["cosine_min"])

    save()
    try:
        engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                       **parse_options(args.option))
        calls = record_calls(engine, engine.sample_inputs(args.seed))
        n_valid = engine.host_state.n_valid
        mask = engine.buffers["mask_bias"][:968]
        report.update(base_plan=dict(engine.identity.plan), calls=len(calls),
                      n_valid=n_valid, prompt_tokens=n_valid - engine.host_state.image_tokens,
                      device_valid_prefix_rows=int((mask == 0).sum().item()),
                      device_mask_896=float(mask[896].item()), phase="prepare")
        save()
        assert len(calls) == 51, len(calls)
        assert [c["site"] for c in calls] == ["gate", "up", "down"] * 17
        assert n_valid == 895 and report["device_valid_prefix_rows"] == 895, report
        assert report["device_mask_896"] < 0
        assert all(c["a"].shape[0] == 968 for c in calls)

        library = cutlass_backbone.library()
        scratch = {route: Scratch(calls[0]["a"].device) for route in ("A", "B")}
        plans = {"A": [], "B": []}
        stream = torch.cuda.current_stream().cuda_stream
        for call in calls:
            a, weight = call["a"], call["weight"]
            call["out"] = torch.empty((968, weight.shape[1]), dtype=a.dtype, device=a.device)
            for route, rows in (("A", 968), ("B", 896)):
                plans[route].append(cutlass_backbone._Plan(
                    library, scratch[route], a[:rows], weight, call["out"][:rows],
                    float(call["residual"] is not None), stream))
        for allocator in scratch.values():
            allocator.freeze()
        report.update(native_library=library._name,
                      scratch_bytes={route: s.nbytes for route, s in scratch.items()},
                      weight_rotation_bytes=sum(c["weight"].numel() * c["weight"].element_size()
                                                for c in calls),
                      shape_mkn=[[968, c["a"].shape[1], c["weight"].shape[1]] for c in calls],
                      phase="correctness")
        save()

        def run(plan):
            cutlass_backbone.check(library.backbone_gemm_run(
                plan.handle, torch.cuda.current_stream().cuda_stream), "backbone_gemm_run")

        def reset_call(call):
            if call["residual"] is not None:
                call["out"].copy_(call["residual"])

        def reset():
            for call in calls:
                reset_call(call)

        limit = tolerances()["shallow"]
        for index, call in enumerate(calls):
            reset_call(call)
            run(plans["A"][index])
            call["reference"] = call["out"].clone()
            row = dict(index=index, layer=call["layer"], site=call["site"])
            valid = True
            if call["actual_output"] is not None:
                row["reference_vs_actual_down"] = error_metrics(
                    call["actual_output"], call["reference"])
                row["reference_vs_actual_down_exact"] = torch.equal(
                    call["actual_output"], call["reference"])
                valid = passes(row["reference_vs_actual_down"])
            reset_call(call)
            run(plans["B"][index])
            row["short_vs_reference"] = {}
            for rows in (895, 896):
                metrics = error_metrics(call["reference"][:rows], call["out"][:rows])
                metrics["exact"] = torch.equal(call["reference"][:rows], call["out"][:rows])
                row["short_vs_reference"][str(rows)] = metrics
                valid = valid and passes(metrics)
            row["correct"] = valid
            report["correctness"].append(row)
            save()
            print(json.dumps({"checked": index + 1, "site": call["site"], "correct": valid,
                              "rel_rms_895": row["short_vs_reference"]["895"]["rel_rms"]}),
                  flush=True)
            if not valid:
                raise RuntimeError(f"static short bucket numerical failure at GEMM {index}")

        functions = {route: [lambda p=p: run(p) for p in route_plans]
                     for route, route_plans in plans.items()}
        report["phase"] = "abba"
        save()
        for route in ("A", "B", "B", "A"):
            samples = samples_ms(functions[route], reset, reps=15)
            leg = dict(route=route, samples_ms_per_call=samples,
                       median_ms_total=statistics.median(samples) * len(calls))
            report["legs"].append(leg)
            print(json.dumps(leg), flush=True)
            save()
        a, b = ([l["median_ms_total"] for l in report["legs"] if l["route"] == route]
                for route in ("A", "B"))
        gain = statistics.mean(a) - statistics.mean(b)
        drift_a, drift_b = abs(a[1] - a[0]), abs(b[1] - b[0])
        report.update(phase="complete", mean_leg_gain_ms=gain, control_drift_ms=drift_a,
                      candidate_drift_ms=drift_b, conservative_gain_ms=min(a) - max(b),
                      static_gain_exceeds_drift=gain > max(drift_a, drift_b))
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
