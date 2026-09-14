"""One static M896 cfg0 screen against shipped Torch M968 QKV/out-projection."""
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
from flash_vla.inference import build, parse_options, resolve
from flash_vla.runtime.runner import Scratch
from lab.pi05.cutlass_gemm_screen import samples_ms

QKV = "llm_backbone_norm_qkv_rope"
OUT = "llm_backbone_out_proj_residual"
ROWS, SHORT_ROWS = 968, 896


def record_calls(engine, inputs):
    """Save both boundaries from one unchanged shipped backbone traversal."""
    calls = {QKV: [], OUT: []}

    def instrument(name, function):
        if name == QKV:
            def qkv(x, weight, rope, q, k, v, x_norm):
                saved = dict(x=x.clone(), weight=weight, rope=rope.clone())
                result = function(x, weight, rope, q, k, v, x_norm)
                saved["expected"] = tuple(t.clone() for t in (q, k, v, x_norm))
                calls[QKV].append(saved)
                return result
            return qkv
        if name == OUT:
            def outproj(x, weight, out):
                saved = dict(x=x.clone(), weight=weight, residual=out.clone())
                result = function(x, weight, out)
                saved["expected"] = out.clone()
                calls[OUT].append(saved)
                return result
            return outproj
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


def qkv_chain(calls, library):
    working = tuple(torch.empty_like(t) for t in calls[0]["expected"])
    q, k, v, normed = working
    projected = torch.empty((ROWS, 2560), dtype=normed.dtype, device=normed.device)
    scratch = Scratch(normed.device)

    def projection_scratch(role, shape, dtype, device):
        return projected

    control = fused_prefix_qkv.make_wrappers(projection_scratch)[QKV]
    norm_library = fused_backbone._library()
    rope_library = fused_prefix_qkv._library()
    stream = torch.cuda.current_stream().cuda_stream
    plans = [cutlass_backbone._Plan(
        library, scratch, normed[:SHORT_ROWS], call["weight"],
        projected[:SHORT_ROWS], 0.0, stream) for call in calls]

    def candidate(call, plan):
        stream = torch.cuda.current_stream().cuda_stream
        cutlass_backbone._check(norm_library.backbone_rms_norm(
            call["x"].data_ptr(), normed.data_ptr(), ROWS, stream), "backbone_rms_norm")
        cutlass_backbone._check(
            library.backbone_gemm_run(plan.handle, stream), "backbone_gemm_run")
        cutlass_backbone._check(rope_library.prefix_rope_scatter(
            projected.data_ptr(), call["rope"].data_ptr(), q.data_ptr(), k.data_ptr(),
            v.data_ptr(), ROWS, stream), "prefix_rope_scatter")

    functions = {
        "A": [lambda c=c: control(c["x"], c["weight"], c["rope"], *working)
              for c in calls],
        "B": [lambda c=c, p=p: candidate(c, p) for c, p in zip(calls, plans)],
    }

    def outputs(call):
        return tuple(zip(("Q", "K", "V", "x_norm"), call["expected"], working))

    return {
        "functions": functions, "outputs": outputs, "reset": projected.zero_,
        "scratch": scratch, "plans": plans, "shape_mkn": [ROWS, 2048, 2560],
        "alpha": 1, "beta": 0,
        "timing_boundary": "full968 RMSNorm -> GEMM A968/B896 -> full968 RoPE/scatter",
        "reset_boundary": "same projected.zero_ before each timed graph, excluded from A and B",
        "finite_tail": "zero projected tail makes full968 scatter write finite zero Q/K/V tail",
        "same_addresses": "A/B share each input/weight plus normed, projected, Q, K and V",
        "cache_policy": "18 real weights rotate, 180 MiB; shared working outputs; no L2 flush",
        "static_limitation": "excluded projected reset supplies a finite tail; no runtime selector",
    }


def outproj_chain(calls, library, control):
    out = torch.empty_like(calls[0]["expected"])
    scratch = Scratch(out.device)
    stream = torch.cuda.current_stream().cuda_stream
    plans = [cutlass_backbone._Plan(
        library, scratch, call["x"][:SHORT_ROWS], call["weight"], out[:SHORT_ROWS],
        1.0, stream) for call in calls]

    def original(call):
        out.copy_(call["residual"])
        control(call["x"], call["weight"], out)

    def candidate(call, plan):
        out.copy_(call["residual"])
        cutlass_backbone._check(library.backbone_gemm_run(
            plan.handle, torch.cuda.current_stream().cuda_stream), "backbone_gemm_run")

    functions = {
        "A": [lambda c=c: original(c) for c in calls],
        "B": [lambda c=c, p=p: candidate(c, p) for c, p in zip(calls, plans)],
    }

    def outputs(call):
        return (("out", call["expected"], out),)

    return {
        "functions": functions, "outputs": outputs, "reset": lambda: None,
        "scratch": scratch, "plans": plans, "shape_mkn": [ROWS, 2048, 2048],
        "alpha": 1, "beta": 1, "c_aliases_d": True,
        "timing_boundary": "per-call full968 residual restore -> GEMM A968/B896",
        "reset_boundary": "identical residual copy included in each A/B timed call",
        "finite_tail": "short GEMM leaves the restored finite residual at rows896:968",
        "same_addresses": "A/B share each input/weight/residual and one common working output",
        "cache_policy": "17 real weights rotate, 136 MiB; shared working output; no L2 flush",
        "static_limitation": "static prefix view; no runtime selector; physical buffers remain968",
    }


def compare(expected, actual, limit):
    exact = torch.equal(expected, actual)
    if exact:
        return {"exact": True, "passed": True}
    metrics = error_metrics(expected, actual)
    passed = (metrics["rel_rms"] <= limit["rel_rms_max"]
              and metrics["cosine_similarity"] >= limit["cosine_min"])
    return {"exact": False, "passed": passed, "metrics": metrics}


def measure_site(site, calls, chain, report, save):
    row = {key: value for key, value in chain.items()
           if key not in ("functions", "outputs", "reset", "scratch", "plans")}
    row.update(calls=len(calls), candidate_config=0, candidate_rows=SHORT_ROWS,
               scratch_bytes=chain["scratch"].nbytes,
               weight_bytes=sum(c["weight"].numel() * c["weight"].element_size() for c in calls),
               correctness=[], legs=[], phase="numerics")
    report["sites"][site] = row
    save()
    limit = tolerances()["shallow"]
    for layer, call in enumerate(calls):
        checked = {"layer": layer}
        passed = True
        for route in ("A", "B"):
            chain["reset"]()
            chain["functions"][route][layer]()
            checked[route] = {}
            for name, expected, actual in chain["outputs"](call):
                entry = {"finite_all": bool(torch.isfinite(actual).all().item())}
                passed = passed and entry["finite_all"]
                if route == "A" or name == "x_norm":
                    entry["full968"] = compare(expected, actual, limit)
                    passed = passed and entry["full968"]["passed"]
                else:
                    # Q flattens token/head rows; compare the same token prefixes.
                    factor = 8 if name == "Q" else 1
                    for rows in (895, 896):
                        entry[str(rows)] = compare(
                            expected[:rows * factor], actual[:rows * factor], limit)
                        passed = passed and entry[str(rows)]["passed"]
                checked[route][name] = entry
        checked["passed"] = passed
        row["correctness"].append(checked)
        save()
        print(json.dumps({"site": site, "layer": layer, "passed": passed}), flush=True)
        if not passed:
            row["phase"] = "numerical_failed"
            save()
            raise RuntimeError(f"{site} numerical failure at layer {layer}")
    chain["scratch"].freeze()
    row["phase"] = "abba"
    save()
    for label, route in (("A1", "A"), ("B1", "B"), ("B2", "B"), ("A2", "A")):
        samples = samples_ms(chain["functions"][route], chain["reset"], reps=15)
        leg = {"leg": label, "median_ms_total": statistics.median(samples) * len(calls),
               "samples_ms_per_call": samples}
        row["legs"].append(leg)
        save()
        print(json.dumps({"site": site, **leg}), flush=True)
    a1, b1, b2, a2 = [leg["median_ms_total"] for leg in row["legs"]]
    row["summary"] = {
        "mean_gain_ms_total": (a1 + a2 - b1 - b2) / 2,
        "control_drift_ms_total": abs(a2 - a1),
        "candidate_drift_ms_total": abs(b2 - b1),
        "conservative_gap_ms_total": min(a1, a2) - max(b1, b2),
    }
    row["phase"] = "complete"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    options = parse_options(args.option)
    # This is the existing real belt-cup workload, never synthetic weights.
    checkpoint = Path(options["converted_checkpoint"])
    assert checkpoint.is_dir(), checkpoint
    runtime_root = Path(cutlass_backbone.__file__).resolve().parents[7]
    probe_root = Path(__file__).resolve().parents[2]
    revision = lambda root: subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "probe_revision": revision(probe_root), "runtime_revision": revision(runtime_root),
        "runtime_source": str(Path(cutlass_backbone.__file__).resolve()),
        "seed": args.seed, "options": args.option, "checkpoint": str(checkpoint),
        "scope": "static M896 screen only; not a deployable dynamic graph",
        "timer": "one fixed ABBA per site, 15 raw samples/leg, first sample retained",
        "phase": "build", "sites": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        engine = build(resolve("rtx5090/pi05"), "shipped", seed=args.seed, **options)
        calls = record_calls(engine, engine.sample_inputs(args.seed))
        mask = engine.buffers["mask_bias"]
        report.update(identity=engine.identity.as_dict(), n_valid=engine.host_state.n_valid,
                      device_valid_prefix_rows=int((mask[:ROWS] == 0).sum().item()),
                      device_mask896=float(mask[896].item()), phase="prepare")
        save()
        assert len(calls[QKV]) == 18 and len(calls[OUT]) == 17
        assert report["n_valid"] == report["device_valid_prefix_rows"] == 895
        assert report["device_mask896"] < 0
        assert dict(engine.identity.plan)[QKV] == "fused-prefix-qkv"
        assert dict(engine.identity.plan)[OUT] == "torch"
        library = cutlass_backbone._library()
        report["candidate_library"] = library._name
        chain = qkv_chain(calls[QKV], library)
        measure_site(QKV, calls[QKV], chain, report, save)
        del chain
        chain = outproj_chain(calls[OUT], library, getattr(engine.ops, OUT))
        measure_site(OUT, calls[OUT], chain, report, save)
        report["phase"] = "complete"
        save()
        print(json.dumps({site: result["summary"] for site, result in report["sites"].items()},
                         indent=2), flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
