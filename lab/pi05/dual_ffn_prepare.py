"""Fixed dual FFN with a four-strip RMS prelude and rounded A loads.

Run this isolated file with PYTHONPATH pointing at the deployed checkout so
the control uses its existing native library and unchanged dual_dot.
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
from triton.language.extra.cuda import libdevice

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import dual_ffn, fused_ffn
from flash_vla.inference import build, resolve
from flash_vla.runtime.runner import Scratch
from measurement.cli import parse_options
from lab.pi05.cutlass_gemm_screen import samples_ms

OPTIONS = dict(num_warps=4, num_stages=3, enable_fp_fusion=False, enable_reflect_ftz=False)


@triton.jit
def _dual_dot_with_prepare(X, Scale, Packed, GateBias, UpBias, Out, Factor,
                           M: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    rms_cols = tl.arange(0, 256)
    sumsq = tl.zeros((16, 256), tl.float32)
    for part in tl.static_range(4):
        raw = tl.load(X + rows[:, None] * 1024
                      + rms_cols[None, :] + part * 256,
                      mask=rows[:, None] < M, other=0.0).to(tl.float32)
        sumsq = sumsq + raw * raw
    summed = tl.sum(sumsq, axis=1)
    factor_bf16 = libdevice.rsqrt(summed / 1024.0 + 1e-6).to(
        tl.bfloat16, fp_downcast_rounding="rtne")
    factor = factor_bf16.to(tl.float32)
    if tl.program_id(1) == 0:
        tl.store(Factor + rows, factor_bf16, mask=rows < M)

    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    kk = tl.arange(0, 32)
    gate_acc = tl.zeros((16, 64), tl.float32)
    up_acc = tl.zeros((16, 64), tl.float32)
    for start in range(0, 1024, 32):
        raw_a = tl.load(X + rows[:, None] * 1024 + (start + kk[None, :]),
                        mask=rows[:, None] < M, other=0.0).to(tl.float32)
        normalized = (raw_a * factor[:, None]).to(
            tl.bfloat16, fp_downcast_rounding="rtne")
        scale = tl.load(Scale + start + kk).to(tl.float32)
        a = (normalized.to(tl.float32) * scale[None, :]).to(
            tl.bfloat16, fp_downcast_rounding="rtne")
        offsets = (start + kk[:, None]) * 8192 + cols[None, :]
        gate_w = tl.load(Packed + offsets)
        up_w = tl.load(Packed + offsets + 4096)
        gate_acc = tl.dot(a, gate_w, gate_acc)
        up_acc = tl.dot(a, up_w, up_acc)
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
    calls = []

    def instrument(name, function):
        if name != fused_ffn.NAMES[0]:
            return function

        def invoke(x, scale, gate_w, up_w, gate_b, up_b, out, factor):
            saved_x, saved_scale = x.clone(), scale.clone()
            result = function(x, scale, gate_w, up_w, gate_b, up_b, out, factor)
            calls.append(dict(x=saved_x, scale=saved_scale, gate_w=gate_w, up_w=up_w,
                              gate_b=gate_b, up_b=up_b, expected=out.clone(),
                              factor=factor[:x.shape[0]].clone()))
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
    parser.add_argument("--phase", choices=("compile", "actual"), required=True)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_root = Path(dual_ffn.__file__).resolve().parents[7]
    script_root = Path(__file__).resolve().parents[2]
    report = dict(
        phase=args.phase, seed=args.seed, options=args.option,
        source_revision=subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        probe_revision=subprocess.check_output(
            ["git", "-C", str(script_root), "rev-parse", "HEAD"], text=True).strip(),
        control_source=dual_ffn.__file__, probe_source=str(Path(__file__).resolve()),
        triton_version=triton.__version__, tile=[16, 64, 32], grid=[4, 64],
        **OPTIONS, rms_prelude=[16, 256, 4], resources=[], correctness=[], legs=[],
        scope="native prepare plus original dual_dot versus fused prepare/dual-dot",
        timer="one 15xABBA; graph of 180 actual calls, original 18-weight order",
        shared_addresses="both routes use the same Packed, Out and Factor allocations",
        is_deployment_sequence=False)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        if args.phase == "compile":
            for route, kernel, pointers in (("A", dual_ffn._dual_dot, 5),
                                             ("B", _dual_dot_with_prepare, 7)):
                compiled = kernel.warmup(*([torch.bfloat16] * pointers), 50,
                                         grid=(4, 64), **OPTIONS)
                compiled._init_handles()
                row = dict(route=route, registers=compiled.n_regs,
                           spills=compiled.n_spills, shared_bytes=compiled.metadata.shared)
                for form in ("ptx", "ttgir"):
                    path = args.output.with_name(args.output.stem + f"-{route}.{form}")
                    path.write_text(compiled.asm[form])
                    row[form] = str(path)
                report["resources"].append(row)
                print(json.dumps(row), flush=True)
                save()
                if route == "B" and compiled.n_spills:
                    raise RuntimeError(f"fused prepare resource screen spills: {compiled.n_spills}")
            report["phase"] = "compiled"
            save()
            return

        engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed,
                       **parse_options(args.option))
        calls = record_calls(engine, engine.sample_inputs(args.seed))
        assert len(calls) == 180, len(calls)
        scratch = Scratch(calls[0]["x"].device)
        packed = {}
        for call in calls:
            pair = (call["gate_w"].data_ptr(), call["up_w"].data_ptr())
            if pair not in packed:
                tensor = scratch(f"prepare_dual_weight_{len(packed)}", (1024, 8192),
                                 call["x"].dtype, call["x"].device)
                tensor[:, :4096].copy_(call["gate_w"])
                tensor[:, 4096:].copy_(call["up_w"])
                packed[pair] = tensor
            call["packed"] = packed[pair]
        assert len(packed) == 18, len(packed)
        a = scratch("prepare_dual_a", calls[0]["x"].shape,
                    calls[0]["x"].dtype, calls[0]["x"].device)
        out = torch.empty_like(calls[0]["expected"])
        factor = torch.empty_like(calls[0]["factor"])
        library = fused_ffn.library()
        report.update(phase="correctness", calls=len(calls), weight_pairs=len(packed),
                      packed_bytes=sum(t.numel() * t.element_size() for t in packed.values()),
                      base_plan=dict(engine.identity.plan), native_library=library._name,
                      scratch_bytes=scratch.nbytes)
        save()

        def control(call):
            fused_ffn.check(library.ada_rms_launch(
                call["x"].data_ptr(), call["scale"].data_ptr(), a.data_ptr(),
                factor.data_ptr(), 50, torch.cuda.current_stream().cuda_stream),
                "ada_rms", 50)
            dual_ffn._dual_dot[(4, 64)](
                a, call["packed"], call["gate_b"], call["up_b"], out, 50, **OPTIONS)

        def candidate(call):
            _dual_dot_with_prepare[(4, 64)](
                call["x"], call["scale"], call["packed"], call["gate_b"], call["up_b"],
                out, factor, 50, **OPTIONS)

        limit = tolerances()["shallow"]
        for index, call in enumerate(calls):
            row = dict(index=index, routes={})
            valid = True
            for route, function in (("A", control), ("B", candidate)):
                function(call)
                metrics = {"out": error_metrics(call["expected"], out),
                           "factor": error_metrics(call["factor"], factor),
                           "out_exact": torch.equal(call["expected"], out),
                           "factor_exact": torch.equal(call["factor"], factor)}
                valid = valid and all(
                    metrics[name]["rel_rms"] <= limit["rel_rms_max"]
                    and metrics[name]["cosine_similarity"] >= limit["cosine_min"]
                    for name in ("out", "factor"))
                row["routes"][route] = metrics
            row["correct"] = valid
            report["correctness"].append(row)
            save()
            if (index + 1) % 18 == 0 or not valid:
                print(json.dumps({"checked": index + 1, "correct": valid,
                                  "last_B_out_rel_rms": row["routes"]["B"]["out"]["rel_rms"],
                                  "last_B_factor_exact": row["routes"]["B"]["factor_exact"]}),
                      flush=True)
            if not valid:
                raise RuntimeError(f"fused prepare numerical failure at actual call {index}")

        scratch.freeze()
        report["phase"] = "abba"
        save()
        functions = {"A": [lambda c=c: control(c) for c in calls],
                     "B": [lambda c=c: candidate(c) for c in calls]}
        for route in ("A", "B", "B", "A"):
            samples = samples_ms(functions[route], lambda: None, reps=15)
            leg = dict(route=route, samples_ms_per_call=samples,
                       median_us_per_call=statistics.median(samples) * 1000)
            report["legs"].append(leg)
            print(json.dumps(leg), flush=True)
            save()
        a, b = ([leg["median_us_per_call"] for leg in report["legs"] if leg["route"] == route]
                for route in ("A", "B"))
        gain = statistics.mean(a) - statistics.mean(b)
        drift_a, drift_b = abs(a[1] - a[0]), abs(b[1] - b[0])
        report.update(phase="complete", mean_leg_gain_us=gain, control_drift_us=drift_a,
                      candidate_drift_us=drift_b, conservative_gain_us=min(a) - max(b),
                      local_gain_exceeds_drift=gain > max(drift_a, drift_b))
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
