"""One fixed Triton FFN-down tile with a BF16-rounded gated residual.

No production route changes. --compile-only records PTX/resources without
loading the model; the full run then checks actual calls before fixed ABBA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import torch
import triton
import triton.language as tl

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_ffn
from flash_vla.inference import build, parse_options, resolve
from lab.pi05.cutlass_gemm_screen import samples_ms

SITE = "action_expert_ffn_down_residual"
LAUNCH = dict(num_warps=4, num_stages=3,
              enable_fp_fusion=False, enable_reflect_ftz=False)


@triton.jit
def down_mm_residual(A, Weight, Gate, Out):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for start in range(0, 4096, 32):
        a = tl.load(A + rows[:, None] * 4096 + start + kk[None, :],
                    mask=rows[:, None] < 50, other=0.0)
        weight = tl.load(Weight + (start + kk[:, None]) * 1024 + cols[None, :])
        acc = tl.dot(a, weight, acc)
    # Preserve the BF16 GEMM output before independent FP32 multiply and add.
    projected = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    gate = tl.load(Gate + cols).to(tl.float32)
    residual = tl.load(Out + rows[:, None] * 1024 + cols[None, :],
                       mask=rows[:, None] < 50, other=0.0).to(tl.float32)
    updated = projected * gate[None, :]
    updated = updated + residual
    tl.store(Out + rows[:, None] * 1024 + cols[None, :],
             updated.to(tl.bfloat16, fp_downcast_rounding="rtne"),
             mask=rows[:, None] < 50)


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
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": args.seed, "options": args.option, "site": SITE,
        "triton_version": triton.__version__, "shape_mkn": [50, 4096, 1024],
        "tile": [16, 32, 32], "grid": [4, 32], **LAUNCH,
        "phase": "compile", "correctness": [], "decomposition": [], "legs": [],
        "timer": "reset-inclusive CUDA graph; same out, inputs, weights, gates and order",
        "is_deployment_sequence": False, "reps_per_leg": 15,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        # Triton's documented MockTensor dtype warmup uses aligned pointers;
        # all actual operands below are contiguous CUDA allocations.
        compiled = down_mm_residual.warmup(
            torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16,
            grid=(4, 32), **LAUNCH)
        # Loading this compiled function exposes driver-reported registers/spills;
        # this does not launch the kernel or load the model.
        compiled._init_handles()
        ptx = args.output.with_suffix(".ptx")
        ptx.write_text(compiled.asm["ptx"])
        report.update(registers=compiled.n_regs, spills=compiled.n_spills,
                      shared_bytes=compiled.metadata.shared, ptx=str(ptx))
        report["phase"] = "compiled"
        save()
        print(json.dumps({key: report[key] for key in (
            "phase", "registers", "spills", "shared_bytes", "ptx")}), flush=True)
        if args.compile_only:
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
                      residual_stride=list(calls[0]["residual"].stride()))
        control = engine.ops[SITE]
        out = torch.empty_like(calls[0]["residual"])

        def candidate(x, weight, gate, output):
            return down_mm_residual[(4, 32)](x, weight, gate, output, **LAUNCH)

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
            report["correctness"].append({"call": index, "output": metrics,
                                          "correct": valid})
            save()
            if not valid:
                raise RuntimeError(f"FFN down shallow check failed at call {index}")

        # Gate=1/C=0 extracts this kernel's rounded projection without changing
        # its mainloop. The deployed separate residual stage checks the epilogue.
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
                raise RuntimeError(f"FFN down rounding decomposition failed at call {index}")

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
