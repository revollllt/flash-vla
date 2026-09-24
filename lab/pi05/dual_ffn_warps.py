"""Compare only num_warps=4 versus 8 on the deployed dual FFN source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
from unittest.mock import patch

import torch
import triton

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import dual_ffn
from flash_vla.inference import build, resolve
from measurement.cli import parse_options
from lab.pi05.cutlass_gemm_screen import samples_ms

OPTIONS = dict(num_stages=3, enable_fp_fusion=False, enable_reflect_ftz=False)


def record_calls(engine, inputs):
    calls, seen = [], set()
    original_run = dual_ffn._dual_dot.run

    def capture(a, packed, gate_b, up_b, out, rows, **kwargs):
        first = packed.data_ptr() not in seen
        if first:
            saved_a = a.clone()
        compiled = original_run(a, packed, gate_b, up_b, out, rows, **kwargs)
        if first:
            calls.append((saved_a, packed, gate_b, up_b, out.clone()))
            seen.add(packed.data_ptr())
        return compiled

    engine.stage(**inputs)
    with patch.object(dual_ffn._dual_dot, "run", capture):
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
    parser.add_argument("--phase", choices=("compile", "actual"), required=True)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "kernel_source": dual_ffn.__file__, "seed": args.seed, "options": args.option,
        "triton_version": triton.__version__, "tile": [16, 64, 32],
        "grid": [4, 64], "warps": [4, 8], **OPTIONS,
        "phase": args.phase, "resources": [], "correctness": [], "legs": [],
        "scope": "same deployed dual-dot suffix; prepare excluded from both",
        "is_deployment_sequence": False, "reps_per_leg": 15,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        if args.phase == "compile":
            for warps in (4, 8):
                compiled = dual_ffn._dual_dot.warmup(
                    *([torch.bfloat16] * 5), 50, grid=(4, 64),
                    num_warps=warps, **OPTIONS)
                compiled._init_handles()
                paths = {}
                for form in ("ptx", "ttgir"):
                    path = args.output.with_name(args.output.stem + f"-w{warps}.{form}")
                    path.write_text(compiled.asm[form])
                    paths[form] = str(path)
                row = {"warps": warps, "registers": compiled.n_regs,
                       "spills": compiled.n_spills, "shared_bytes": compiled.metadata.shared,
                       **paths}
                report["resources"].append(row)
                print(json.dumps(row), flush=True)
                save()
            report["phase"] = "compiled"
            save()
            return

        engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                       **parse_options(args.option))
        report["deployment_plan"] = dict(engine.identity.plan)
        calls = record_calls(engine, engine.sample_inputs(args.seed))
        report.update(calls=len(calls), weight_rotation_bytes=sum(
            packed.numel() * packed.element_size() for _, packed, _, _, _ in calls))
        output = torch.empty_like(calls[0][-1])

        def run(call, warps):
            a, packed, gate_b, up_b, _ = call
            dual_ffn._dual_dot[(4, 64)](
                a, packed, gate_b, up_b, output, a.shape[0], num_warps=warps, **OPTIONS)

        limit = tolerances()["shallow"]
        report["phase"] = "correctness"
        for index, call in enumerate(calls):
            metrics = {}
            for route, warps in (("A", 4), ("B", 8)):
                run(call, warps)
                metrics[route] = error_metrics(call[-1], output)
                metrics[route]["exact"] = torch.equal(call[-1], output)
            valid = all(m["rel_rms"] <= limit["rel_rms_max"]
                        and m["cosine_similarity"] >= limit["cosine_min"]
                        for m in metrics.values())
            report["correctness"].append({"layer": index, "output": metrics, "correct": valid})
            save()
            if not valid:
                raise RuntimeError(f"dual FFN warps shallow check failed at layer {index}")
        print(json.dumps({"correct_layers": len(calls), "max_rel_rms": max(
            row["output"]["B"]["rel_rms"] for row in report["correctness"])}), flush=True)

        functions = {"A": [lambda c=c: run(c, 4) for c in calls],
                     "B": [lambda c=c: run(c, 8) for c in calls]}
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
