"""Screen a rounded dual-dot FFN suffix against actual packed expert calls.

Only replaces BF16 packed GEMM plus bias/GELU/product; prepare is unchanged.
The fixed tiles test reuse direction and CTA count, without autotuning.
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
from triton.language.extra.cuda import libdevice

from eval.metrics import error_metrics
from eval.tolerances import tolerances

TILES = ((32, 32, 32), (16, 64, 32), (16, 32, 32))


@triton.jit
def dual_dot(A, Packed, GateBias, UpBias, Out, M: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate_acc = tl.zeros((BM, BN), tl.float32)
    up_acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, 1024, BK):
        a = tl.load(A + rows[:, None] * 1024 + (start + kk[None, :]),
                    mask=rows[:, None] < M, other=0.0)
        offsets = (start + kk[:, None]) * 8192 + cols[None, :]
        gate_w = tl.load(Packed + offsets)
        up_w = tl.load(Packed + offsets + 4096)
        gate_acc = tl.dot(a, gate_w, gate_acc)
        up_acc = tl.dot(a, up_w, up_acc)
    # These roundtrips preserve the two BF16 GEMM outputs before either bias.
    gate = gate_acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    up = up_acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    gate = gate + tl.load(GateBias + cols)[None, :].to(tl.float32)
    up = up + tl.load(UpBias + cols)[None, :].to(tl.float32)
    cube = gate * gate * gate
    gelu = 0.5 * gate * (1.0 + libdevice.tanh(
        0.7978845608028654 * (gate + 0.044715 * cube)))
    result = (gelu * up).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Out + rows[:, None] * 4096 + cols[None, :], result,
             mask=rows[:, None] < M)


def record_calls(engine, inputs):
    calls, seen = [], set()
    original_mm = torch.mm

    def wrap(name, function):
        if name != "action_expert_norm_gated_ffn":
            return function

        def invoke(x, scale, gate_w, up_w, gate_b, up_b, out, factor):
            captured = None

            def capture_mm(a, packed, **kwargs):
                nonlocal captured
                if packed.data_ptr() not in seen:
                    seen.add(packed.data_ptr())
                    captured = (a.clone(), packed, gate_b, up_b)
                return original_mm(a, packed, **kwargs)

            with patch.object(torch, "mm", capture_mm):
                result = function(x, scale, gate_w, up_w, gate_b, up_b, out, factor)
            if captured is not None:
                calls.append((*captured, result.clone()))
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
    from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_ffn
    from lab.pi05.cutlass_gemm_screen import samples_ms

    engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                   **parse_options(args.option))
    print(json.dumps({"deployment_revision_before_import": revision,
                      "deployment_plan": dict(engine.identity.plan)}), flush=True)
    calls = record_calls(engine, engine.sample_inputs(args.seed))
    if not calls:
        raise RuntimeError("no packed expert torch.mm operands were recorded")
    # Match the deployed suffix's shared intermediate and shared output buffers.
    a0, packed0, _, _, expected0 = calls[0]
    both = torch.empty((a0.shape[0], 8192), dtype=a0.dtype, device=a0.device)
    output = torch.empty_like(expected0)
    native = fused_ffn.library()

    def control(a, packed, gate_b, up_b):
        torch.mm(a, packed, out=both)
        fused_ffn.check(native.packed_gated_activation_launch(
            both.data_ptr(), both.data_ptr() + 4096 * both.element_size(),
            gate_b.data_ptr(), up_b.data_ptr(), output.data_ptr(), a.shape[0],
            torch.cuda.current_stream().cuda_stream), "packed_gated_activation", a.shape[0])

    controls = [lambda a=a, packed=packed, gb=gb, ub=ub: control(a, packed, gb, ub)
                for a, packed, gb, ub, _ in calls]
    report = {
        "deployment_revision_before_import": revision, "deployment_plan": dict(engine.identity.plan),
        "seed": args.seed, "options": args.option, "calls_per_graph": len(calls),
        "weight_rotation_bytes": sum(p.numel() * p.element_size() for _, p, _, _, _ in calls),
        "input_source": "first actual prepared BF16 activation for each distinct packed weight",
        "scope": "packed GEMM plus gated activation; prepare excluded from both paths",
        "shape_mkn": [a0.shape[0], a0.shape[1], packed0.shape[1]],
        "dtype": str(a0.dtype), "bias_dtype": str(calls[0][2].dtype),
        "triton_version": triton.__version__, "enable_fp_fusion": False,
        "enable_reflect_ftz": False, "num_warps": 4, "num_stages": 3, "results": []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for bm, bn, bk in TILES:
        tile = (bm, bn, bk)

        def candidate(a, packed, gb, ub):
            return dual_dot[(triton.cdiv(a.shape[0], bm), triton.cdiv(4096, bn))](
                a, packed, gb, ub, output, a.shape[0], bm, bn, bk,
                num_warps=4, num_stages=3, enable_fp_fusion=False, enable_reflect_ftz=False)

        metrics = []
        for a, packed, gb, ub, expected in calls:
            compiled = candidate(a, packed, gb, ub)
            metrics.append(error_metrics(expected, output))
        limit = tolerances()["shallow"]
        valid = all(row["rel_rms"] <= limit["rel_rms_max"]
                    and row["cosine_similarity"] >= limit["cosine_min"] for row in metrics)
        ptx_path = args.output.with_name(args.output.stem + f"-{bm}x{bn}x{bk}.ptx")
        ptx_path.write_text(compiled.asm["ptx"])
        result = {"tile": tile, "correct": valid, "per_layer_metrics": metrics,
                  "registers": compiled.n_regs, "spills": compiled.n_spills,
                  "shared_bytes": compiled.metadata.shared, "ptx": str(ptx_path), "legs": []}
        report["results"].append(result)
        print(json.dumps({"tile": tile, "correct": valid, "registers": compiled.n_regs,
                          "spills": compiled.n_spills, "shared_bytes": compiled.metadata.shared,
                          "worst_rel_rms": max(row["rel_rms"] for row in metrics),
                          "lowest_cosine": min(row["cosine_similarity"] for row in metrics)}), flush=True)
        save()
        if not valid:
            raise RuntimeError(f"dual-dot numerical mismatch for tile {tile}")
        candidates = [lambda a=a, packed=p, gb=gb, ub=ub: candidate(a, packed, gb, ub)
                      for a, p, gb, ub, _ in calls]
        for route, functions in (("A", controls), ("B", candidates),
                                 ("B", candidates), ("A", controls)):
            samples = samples_ms(functions, lambda: None, reps=15)
            leg = {"route": route, "samples_ms_per_call": samples,
                   "median_us_per_call": statistics.median(samples) * 1000}
            result["legs"].append(leg)
            print(json.dumps({"tile": tile, **leg}), flush=True)
            save()


if __name__ == "__main__":
    main()
