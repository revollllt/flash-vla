"""Screen only PV dispatch for actual BF16 P/V with K=1018 versus padded K=1024."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def prepare(args):
    """Capture the first actual deployed probability/V pair outside profiling."""
    from flash_vla.inference import build, resolve

    engine = build(
        resolve("rtx5090/pi05"), "shipped", seed=42,
        converted_checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
        checkpoint_digest=args.checkpoint_id)
    inputs = engine.sample_inputs(42)
    saved = False

    def inspect(name, function):
        if name != "action_expert_attention":
            return function

        def record(Q, K, V, mask, out, prefix_len=None):
            nonlocal saved
            result = function(Q, K, V, mask, out, prefix_len)
            if not saved:
                probabilities = engine.scratch(
                    "pi05_attention_probabilities", (Q.shape[0], K.shape[0]),
                    Q.dtype, Q.device)
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {"probabilities": probabilities.detach().cpu().contiguous(),
                     "values": V.detach().cpu().contiguous()},
                    str(args.snapshot),
                    metadata={"checkpoint": args.checkpoint_id, "seed": "42",
                              "step": "0", "layer": "0",
                              "identity": json.dumps(engine.identity.as_dict())})
                saved = True
            return result

        return record

    engine.stage(**inputs)
    with engine.instrument(inspect):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "action_expert":
                engine.run_eager(step.name)
                if saved:
                    break
            else:
                engine.replay(step.name)
    torch.cuda.synchronize()
    print(json.dumps({"snapshot": str(args.snapshot), "saved": saved}), flush=True)


def cases(args):
    """Prepare both layouts before either timing or tracing begins."""
    with safe_open(str(args.snapshot), framework="pt", device="cpu") as stored:
        probabilities = stored.get_tensor("probabilities").cuda()
        values = stored.get_tensor("values").cuda()
        metadata = stored.metadata()
    padded_p = torch.zeros((400, 1024), dtype=probabilities.dtype, device="cuda")
    padded_v = torch.zeros((1024, 256), dtype=values.dtype, device="cuda")
    padded_p[:, :1018].copy_(probabilities)
    padded_v[:1018].copy_(values)
    original_out = torch.empty((400, 256), dtype=values.dtype, device="cuda")
    padded_out = torch.empty_like(original_out)
    prepared = {"pv_k1018": (probabilities, values, original_out),
                "pv_k1024": (padded_p, padded_v, padded_out)}
    for p, v, out in prepared.values():
        torch.mm(p, v, out=out)
    torch.cuda.synchronize()
    return prepared, metadata


def time_cases(args):
    from eval.metrics import error_metrics
    from eval.tolerances import tolerances
    from flash_vla.bench import bench_gpu_time

    prepared, metadata = cases(args)
    expected, actual = prepared["pv_k1018"][2], prepared["pv_k1024"][2]
    metrics = error_metrics(expected, actual)
    limits = tolerances()["shallow"]
    assert metrics["rel_rms"] <= limits["rel_rms_max"], metrics
    assert metrics["cosine_similarity"] >= limits["cosine_min"], metrics

    def pv(p, v, out):
        torch.mm(p, v, out=out)

    rows = []
    for label in ("pv_k1018", "pv_k1024", "pv_k1024", "pv_k1018"):
        samples = bench_gpu_time(
            pv, input_args=prepared[label], enable_cupti=False,
            use_cuda_graph=True, cold_l2_cache=False, num_iters_within_graph=64,
            dry_run_iters=5, repeat_iters=30)
        rows.append({"case": label, "median_ms": median(samples), "samples_ms": samples})
    report = {"snapshot_metadata": metadata, "pv_output": metrics,
              "bitwise_equal": torch.equal(expected, actual),
              "cache_policy": "one actual P/V pair repeatedly reused; warm-cache screen",
              "timing_boundary": "torch.mm only; all padding/copies outside timing",
              "timer": "fresh CUDA graph on capture stream; ABBA; 64 calls per graph",
              "timings": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "timings": [{k: v for k, v in row.items()
                                           if k != "samples_ms"} for row in rows]}, indent=2))


def trace_cases(args):
    prepared, metadata = cases(args)
    for _ in range(10):
        for p, v, out in prepared.values():
            torch.mm(p, v, out=out)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
    ]) as profiler:
        for label, (p, v, out) in prepared.items():
            with torch.profiler.record_function(label):
                torch.mm(p, v, out=out)
                torch.cuda.synchronize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.out.with_suffix(".trace.json")
    profiler.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    dispatch = {}
    for event in events:
        if event.get("cat") == "user_annotation" and event["name"] in prepared:
            start, end = event["ts"], event["ts"] + event["dur"]
            kernels = [item for item in events if item.get("cat") == "kernel"
                       and start <= item["ts"] and item["ts"] + item["dur"] <= end]
            assert kernels, event
            dispatch[event["name"]] = [
                {"name": item["name"], "grid": item["args"].get("grid"),
                 "block": item["args"].get("block"),
                 "diagnostic_duration_us": item["dur"]} for item in kernels]
    assert set(dispatch) == set(prepared), dispatch
    report = {"snapshot_metadata": metadata, "diagnostic_only": True,
              "trace": str(trace_path), "dispatch": dispatch}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "time", "trace"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.checkpoint is None or args.checkpoint_id is None:
            parser.error("prepare requires --checkpoint and --checkpoint-id")
        prepare(args)
    else:
        if args.out is None:
            parser.error("time/trace requires --out")
        (time_cases if args.mode == "time" else trace_cases)(args)


if __name__ == "__main__":
    main()
