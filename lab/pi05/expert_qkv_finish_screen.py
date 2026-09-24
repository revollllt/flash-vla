"""Test one QKV GEMM epilogue fusion on actual expert calls.

One 16x32x32 dot tile preserves the BF16 projection roundtrip, then performs
factor, bias, adjacent RoPE pairs, and Q/K/V stores. Prepare is unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import triton
import triton.language as tl

from eval.metrics import error_metrics
from eval.tolerances import tolerances


@triton.jit
def qkv_mm_finish(A, Weight, Factor, Bias, Rope, Q, K, V, M: tl.constexpr):
    tile_n = tl.program_id(1)
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tile_n * 32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    acc = tl.zeros((16, 32), tl.float32)
    for start in range(0, 1024, 32):
        a = tl.load(A + rows[:, None] * 1024 + start + kk[None, :],
                    mask=rows[:, None] < M, other=0.0)
        weight = tl.load(Weight + (start + kk[:, None]) * 2560 + cols[None, :])
        acc = tl.dot(a, weight, acc)
    # Keep the store/reload rounding of the separate BF16 GEMM before its epilogue.
    projected = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    factor = tl.load(Factor + rows, mask=rows < M, other=0.0).to(tl.float32)
    values = projected * factor[:, None]
    values = values + tl.load(Bias + cols)[None, :].to(tl.float32)
    if tile_n < 72:
        even, odd = tl.split(tl.reshape(values, (16, 16, 2)))
        pairs = tile_n * 16 + tl.arange(0, 16)
        rope_cols = 2 * (pairs % 128)
        cos = tl.load(Rope + rows[:, None] * 256 + rope_cols[None, :],
                      mask=rows[:, None] < M, other=0.0).to(tl.float32)
        sin = tl.load(Rope + rows[:, None] * 256 + rope_cols[None, :] + 1,
                      mask=rows[:, None] < M, other=0.0).to(tl.float32)
        rotated_even = even * cos - odd * sin
        rotated_odd = odd * cos + even * sin
        values = tl.reshape(tl.join(rotated_even, rotated_odd), (16, 32))
    result = values.to(tl.bfloat16, fp_downcast_rounding="rtne")
    if tile_n < 64:
        tl.store(Q + rows[:, None] * 2048 + cols[None, :], result,
                 mask=rows[:, None] < M)
    elif tile_n < 72:
        # K/V are already suffix pointers from the caller, with prefix offset applied.
        tl.store(K + rows[:, None] * 256 + cols[None, :] - 2048, result,
                 mask=rows[:, None] < M)
    else:
        tl.store(V + rows[:, None] * 256 + cols[None, :] - 2304, result,
                 mask=rows[:, None] < M)


def record_calls(engine, inputs):
    calls, seen = [], set()

    def wrap(name, function):
        if name != "action_expert_norm_qkv_rope":
            return function

        def invoke(x, scale, weight, bias, rope, q, k, v, factor):
            if weight.data_ptr() in seen:
                return function(x, scale, weight, bias, rope, q, k, v, factor)
            # The graph visits layer weights in layer order at the first denoising step.
            layer = len(calls)
            saved = {"x": x.clone(), "scale": scale, "weight": weight,
                     "bias": bias, "rope": rope.clone(),
                     "prefix_k": engine.buffers["prefix_k"][layer].clone(),
                     "prefix_v": engine.buffers["prefix_v"][layer].clone(),
                     "actual_k_offset_elements":
                         (k.data_ptr() - engine.buffers["kv_k"][layer].data_ptr()) // k.element_size(),
                     "actual_v_offset_elements":
                         (v.data_ptr() - engine.buffers["kv_v"][layer].data_ptr()) // v.element_size()}
            result = function(x, scale, weight, bias, rope, q, k, v, factor)
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
    from flash_vla.inference import build, resolve
    from measurement.cli import parse_options
    from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_qkv, triton_qkv
    from lab.pi05.cutlass_gemm_screen import samples_ms

    engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                   **parse_options(args.option))
    report = {"deployment_revision_before_import": revision,
              "deployment_plan": dict(engine.identity.plan),
              "control_source": triton_qkv.__file__,
              "timing_helper": samples_ms.__code__.co_filename,
              "seed": args.seed, "options": args.option, "triton_version": triton.__version__,
              "tile": [16, 32, 32], "num_warps": 4, "num_stages": 3,
              "enable_fp_fusion": False, "enable_reflect_ftz": False,
              "scope": "complete prepare/GEMM/epilogue chain on the first actual call of each layer",
              "is_deployment_sequence": False, "reps_per_leg": 15,
              "phase": "record_calls", "legs": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(json.dumps(report), flush=True)
    try:
        calls = record_calls(engine, engine.sample_inputs(args.seed))
        if not calls:
            raise RuntimeError("no expert QKV calls were recorded")
        first = calls[0]
        scaled = torch.empty_like(first["x"])
        projected = torch.empty((first["x"].shape[0], 2560), dtype=scaled.dtype, device=scaled.device)
        factor = torch.empty_like(first["factor"])
        q = torch.empty_like(first["q"])
        native = fused_qkv.library()
        for call in calls:
            # Copies and view construction occur only in setup, outside every timed graph.
            call["k_cache"] = torch.cat((call["prefix_k"], torch.empty_like(call["k"])))
            call["v_cache"] = torch.cat((call["prefix_v"], torch.empty_like(call["v"])))
            call["k_out"] = call["k_cache"][call["prefix_k"].shape[0]:]
            call["v_out"] = call["v_cache"][call["prefix_v"].shape[0]:]
        report.update(actual_layers=len(calls), calls_per_graph=len(calls),
                      shape_mkn=[first["x"].shape[0], 1024, 2560],
                      weight_rotation_bytes=sum(c["weight"].numel() * c["weight"].element_size()
                                                for c in calls),
                      prefix_rows=[c["prefix_k"].shape[0] for c in calls],
                      actual_k_offset_elements=[c["actual_k_offset_elements"] for c in calls],
                      actual_v_offset_elements=[c["actual_v_offset_elements"] for c in calls],
                      ctas=triton.cdiv(first["x"].shape[0], 16) * 80)

        def run(call, fused):
            rows = call["x"].shape[0]
            stream = torch.cuda.current_stream().cuda_stream
            rc = native.pi05_qkv_prepare(
                call["x"].data_ptr(), call["scale"].data_ptr(), scaled.data_ptr(),
                factor.data_ptr(), rows, stream)
            if rc:
                raise RuntimeError(f"QKV prepare CUDA error {rc}")
            if fused:
                return qkv_mm_finish[(triton.cdiv(rows, 16), 80)](
                    scaled, call["weight"], factor, call["bias"], call["rope"],
                    q, call["k_out"], call["v_out"], rows,
                    num_warps=4, num_stages=3, enable_fp_fusion=False, enable_reflect_ftz=False)
            triton_qkv._qkv_mm[(triton.cdiv(rows, 16), 80)](
                scaled, call["weight"], projected, rows, num_warps=4, num_stages=3,
                enable_fp_fusion=False, enable_reflect_ftz=False)
            rc = native.pi05_qkv_finish(
                projected.data_ptr(), factor.data_ptr(), call["bias"].data_ptr(),
                call["rope"].data_ptr(), q.data_ptr(), call["k_out"].data_ptr(),
                call["v_out"].data_ptr(), rows, stream)
            if rc:
                raise RuntimeError(f"QKV finish CUDA error {rc}")
            return None

        limit = tolerances()["shallow"]
        report["tolerance"] = limit
        for route, fused in (("A", False), ("B", True)):
            report["phase"] = f"{route}_correctness"
            save()
            metrics, prefix_checks = [], []
            for call in calls:
                compiled = run(call, fused)
                metrics.append({name: error_metrics(call[name], output)
                                for name, output in (("q", q), ("k", call["k_out"]),
                                                     ("v", call["v_out"]), ("factor", factor))})
                prefix_checks.append({
                    name: torch.equal(call[name + "_cache"][:call["prefix_" + name].shape[0]],
                                      call["prefix_" + name])
                    for name in ("k", "v")})
            valid = all(m["rel_rms"] <= limit["rel_rms_max"]
                        and m["cosine_similarity"] >= limit["cosine_min"]
                        for layer in metrics for m in layer.values())
            valid = valid and all(all(row.values()) for row in prefix_checks)
            report[route + "_correctness"] = {
                "correct": valid, "per_layer_metrics": metrics,
                "prefix_unchanged_per_layer": prefix_checks}
            if fused:
                ptx = args.output.with_suffix(".ptx")
                ptx.write_text(compiled.asm["ptx"])
                report.update(registers=compiled.n_regs, spills=compiled.n_spills,
                              shared_bytes=compiled.metadata.shared, ptx=str(ptx))
            print(json.dumps({"phase": report["phase"], "correct": valid,
                              "worst_rel_rms": max(m["rel_rms"] for layer in metrics for m in layer.values()),
                              "lowest_cosine": min(m["cosine_similarity"] for layer in metrics for m in layer.values()),
                              "prefix_unchanged": all(all(row.values()) for row in prefix_checks)}), flush=True)
            save()
            if not valid:
                raise RuntimeError(f"QKV finish fusion numerical/prefix mismatch in {route}")

        controls = [lambda call=call: run(call, False) for call in calls]
        candidates = [lambda call=call: run(call, True) for call in calls]
        report["phase"] = "full_chain_abba"
        for route, functions in (("A", controls), ("B", candidates),
                                 ("B", candidates), ("A", controls)):
            samples = samples_ms(functions, lambda: None, reps=15)
            leg = {"route": route, "samples_ms_per_call": samples,
                   "median_us_per_call": statistics.median(samples) * 1000}
            report["legs"].append(leg)
            print(json.dumps(leg), flush=True)
            save()
        a = [leg["median_us_per_call"] for leg in report["legs"] if leg["route"] == "A"]
        b = [leg["median_us_per_call"] for leg in report["legs"] if leg["route"] == "B"]
        report.update(mean_leg_delta_us=statistics.mean(a) - statistics.mean(b),
                      conservative_leg_delta_us=min(a) - max(b),
                      control_drift_us=abs(a[1] - a[0]), candidate_drift_us=abs(b[1] - b[0]))
        report["phase"] = "complete" if report["mean_leg_delta_us"] > 0 else "complete_no_local_gain"
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
