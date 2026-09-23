"""Screen three Triton replacements for the actual expert QKV BF16 GEMM.

Prepare and factor/bias/RoPE/scatter remain the deployed CUDA functions.
The 36-weight GEMM rotation controls cache pressure; it is not a deployment trace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from eval.metrics import error_metrics
from eval.tolerances import tolerances

TILES = ((32, 32, 32), (16, 64, 32), (16, 32, 32))


@triton.jit
def qkv_mm(A, Weight, Projected, M: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, 1024, BK):
        a = tl.load(A + rows[:, None] * 1024 + start + kk[None, :],
                    mask=rows[:, None] < M, other=0.0)
        weight = tl.load(Weight + (start + kk[:, None]) * 2560 + cols[None, :])
        acc = tl.dot(a, weight, acc)
    # Finish consumes the rounded BF16 projection, not the FP32 accumulator.
    projected = acc.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Projected + rows[:, None] * 2560 + cols[None, :], projected,
             mask=rows[:, None] < M)


def record_calls(engine, inputs):
    calls, seen = [], set()
    original_mm = torch.mm

    def wrap(name, function):
        if name != "action_expert_norm_qkv_rope":
            return function

        def invoke(x, scale, weight, bias, rope, q, k, v, factor):
            if weight.data_ptr() in seen:
                return function(x, scale, weight, bias, rope, q, k, v, factor)
            saved = {"x": x.clone(), "scale": scale, "weight": weight,
                     "bias": bias, "rope": rope.clone()}

            def capture_mm(a, b, **kwargs):
                saved["prepared"] = a.clone()
                result = original_mm(a, b, **kwargs)
                saved["projected"] = result.clone()
                return result

            with patch.object(torch, "mm", capture_mm):
                result = function(x, scale, weight, bias, rope, q, k, v, factor)
            if "prepared" not in saved:
                raise RuntimeError("deployed QKV no longer uses torch.mm; update the probe")
            saved.update(q=q.clone(), k=k.clone(), v=v.clone(), factor=factor.clone())
            calls.append(saved)
            seen.add(weight.data_ptr())
            return result
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(args.source_checkout / "src"))
    from flash_vla.inference import build, parse_options, resolve
    from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_qkv
    from lab.pi05.cutlass_gemm_screen import samples_ms

    engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                   **parse_options(args.option))
    report = {
        "deployment_revision_before_import": revision,
        "deployment_plan": dict(engine.identity.plan),
        "source_qkv": fused_qkv.__file__, "timing_helper": samples_ms.__code__.co_filename,
        "seed": args.seed, "options": args.option, "triton_version": triton.__version__,
        "enable_fp_fusion": False, "enable_reflect_ftz": False,
        "num_warps": 4, "num_stages": 3, "reps_per_leg": 15,
        "input_source": "first actual QKV call per distinct layer weight",
        "scope": "replace only BF16 GEMM; unchanged prepare and finish",
        "cache_pressure_sequence_is_deployment_sequence": False,
        "phase": "record_calls", "results": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(json.dumps(report), flush=True)
    try:
        calls = record_calls(engine, engine.sample_inputs(args.seed))
        if not calls:
            raise RuntimeError("no actual expert QKV calls were recorded")
        first = calls[0]
        scaled = torch.empty_like(first["prepared"])
        projected = torch.empty_like(first["projected"])
        factor = torch.empty_like(first["factor"])
        q, k, v = (torch.empty_like(first[name]) for name in ("q", "k", "v"))
        native = fused_qkv.library()
        report.update(actual_layers=len(calls),
                      shape_mkn=[first["x"].shape[0], 1024, 2560],
                      dtype=str(first["x"].dtype),
                      original_weight_bytes=sum(c["weight"].numel() * c["weight"].element_size()
                                                for c in calls))
        limit = tolerances()["shallow"]
        report["tolerance"] = limit

        def matmul(a, weight, tile):
            if tile is None:
                torch.mm(a, weight, out=projected)
                return None
            bm, bn, bk = tile
            return qkv_mm[(triton.cdiv(a.shape[0], bm), triton.cdiv(2560, bn))](
                a, weight, projected, a.shape[0], bm, bn, bk,
                num_warps=4, num_stages=3, enable_fp_fusion=False, enable_reflect_ftz=False)

        def chain(call, tile):
            rows = call["x"].shape[0]
            stream = torch.cuda.current_stream().cuda_stream
            rc = native.pi05_qkv_prepare(
                call["x"].data_ptr(), call["scale"].data_ptr(), scaled.data_ptr(),
                factor.data_ptr(), rows, stream)
            if rc:
                raise RuntimeError(f"QKV prepare CUDA error {rc}")
            compiled = matmul(scaled, call["weight"], tile)
            rc = native.pi05_qkv_finish(
                projected.data_ptr(), factor.data_ptr(), call["bias"].data_ptr(),
                call["rope"].data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(), rows, stream)
            if rc:
                raise RuntimeError(f"QKV finish CUDA error {rc}")
            return compiled

        def validate(tile):
            metrics = []
            for call in calls:
                compiled = chain(call, tile)
                metrics.append({name: error_metrics(call[name], output)
                                for name, output in (("projected", projected), ("q", q),
                                                     ("k", k), ("v", v), ("factor", factor))})
            valid = all(m["rel_rms"] <= limit["rel_rms_max"]
                        and m["cosine_similarity"] >= limit["cosine_min"]
                        for layer in metrics for m in layer.values())
            return valid, metrics, compiled

        report["phase"] = "control_correctness"
        valid, metrics, _ = validate(None)
        report["control_correct"] = valid
        report["control_per_layer_metrics"] = metrics
        save()
        if not valid:
            raise RuntimeError("recorded QKV chain control mismatch")

        # Same two physical weight banks and order for both A and B.
        # Only setup duplicates weights; neither timed route performs a copy.
        bank0 = [(c["prepared"], c["weight"]) for c in calls]
        bank1 = [(c["prepared"], c["weight"].clone()) for c in calls]
        rotation = bank0 + bank1
        report.update(cache_pressure_calls_per_graph=len(rotation),
                      cache_pressure_weight_bytes=2 * report["original_weight_bytes"],
                      cache_pressure_order="all original layer weights, then all cloned layer weights",
                      l2_bytes=96 * 1024 * 1024)

        def abba(controls, candidates):
            legs = []
            for route, functions in (("A", controls), ("B", candidates),
                                     ("B", candidates), ("A", controls)):
                samples = samples_ms(functions, lambda: None, reps=15)
                leg = {"route": route, "samples_ms_per_call": samples,
                       "median_us_per_call": statistics.median(samples) * 1000}
                legs.append(leg)
                print(json.dumps({"phase": report["phase"], **leg}), flush=True)
            a = [leg["median_us_per_call"] for leg in legs if leg["route"] == "A"]
            b = [leg["median_us_per_call"] for leg in legs if leg["route"] == "B"]
            return {"legs": legs, "mean_leg_delta_us": statistics.mean(a) - statistics.mean(b),
                    "conservative_leg_delta_us": min(a) - max(b),
                    "control_drift_us": abs(a[1] - a[0]),
                    "candidate_drift_us": abs(b[1] - b[0])}

        controls = [lambda a=a, weight=w: matmul(a, weight, None) for a, w in rotation]
        for tile in TILES:
            report["phase"] = f"tile_{tile}_correctness"
            save()
            valid, metrics, compiled = validate(tile)
            result = {"tile": tile, "correct": valid, "per_layer_metrics": metrics,
                      "registers": compiled.n_regs, "spills": compiled.n_spills,
                      "shared_bytes": compiled.metadata.shared,
                      "ctas": triton.cdiv(first["x"].shape[0], tile[0]) * triton.cdiv(2560, tile[1])}
            report["results"].append(result)
            ptx = args.output.with_name(args.output.stem + "-" + "x".join(map(str, tile)) + ".ptx")
            ptx.write_text(compiled.asm["ptx"])
            result["ptx"] = str(ptx)
            print(json.dumps({key: value for key, value in result.items()
                              if key != "per_layer_metrics"}), flush=True)
            save()
            if not valid:
                raise RuntimeError(f"QKV numerical mismatch for tile {tile}")
            candidates = [lambda a=a, weight=w, tile=tile: matmul(a, weight, tile)
                          for a, w in rotation]
            report["phase"] = f"tile_{tile}_cache_pressure"
            result["cache_pressure"] = abba(controls, candidates)
            save()

        winner = max(report["results"],
                     key=lambda result: result["cache_pressure"]["mean_leg_delta_us"])
        report["best_cache_pressure_tile"] = winner["tile"]
        if winner["cache_pressure"]["mean_leg_delta_us"] <= 0:
            report["phase"] = "complete_no_faster_tile"
            save()
            return
        report["phase"] = "winner_original_18_weight_chain"
        # The complete chain was already checked for every tile before its timing.
        report["winner_chain_correct"] = winner["correct"]
        controls = [lambda call=call: chain(call, None) for call in calls]
        candidates = [lambda call=call: chain(call, winner["tile"]) for call in calls]
        report["original_weight_chain"] = {
            "calls_per_graph": len(calls), "weight_bytes": report["original_weight_bytes"],
            "is_deployment_sequence": False,
            "scope": "complete prepare/GEMM/finish chain, first actual call of each layer",
            **abba(controls, candidates)}
        report["phase"] = "complete"
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
