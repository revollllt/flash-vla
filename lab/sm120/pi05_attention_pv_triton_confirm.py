"""Confirm the 16x32x64 PV screen on nine actual denoising step/layer pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.bench import bench_gpu_time
from pi05_attention_pv_triton_probe import _pv


def prepare(args):
    from flash_vla.inference import build, resolve

    engine = build(
        resolve("rtx5090/pi05"), "shipped", seed=42,
        converted_checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
        checkpoint_digest=args.checkpoint_id)
    inputs = engine.sample_inputs(42)
    captured = {}
    index = 0

    def inspect(name, function):
        if name != "action_expert_attention":
            return function

        def record(Q, K, V, mask, out, prefix_len=None):
            nonlocal index
            result = function(Q, K, V, mask, out, prefix_len)
            step, layer = divmod(index, 18)
            if step in (0, 4, 9) and layer in (0, 8, 17):
                label = f"step{step}_layer{layer}"
                p = engine.scratch("pi05_attention_probabilities",
                                   (400, 1018), Q.dtype, Q.device)
                captured[label + "_p"] = p.detach().cpu().contiguous()
                captured[label + "_v"] = V.detach().cpu().contiguous()
            index += 1
            return result

        return record

    engine.stage(**inputs)
    with engine.instrument(inspect):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "action_expert":
                engine.run_eager(step.name)
            else:
                engine.replay(step.name)
    assert len(captured) == 18, (index, captured.keys())
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    save_file(captured, str(args.snapshot), metadata={
        "identity": json.dumps(engine.identity.as_dict()), "seed": "42",
        "checkpoint": args.checkpoint_id})
    print(json.dumps({"snapshot": str(args.snapshot), "pairs": len(captured) // 2}))


def load_pairs(args):
    with safe_open(str(args.snapshot), framework="pt", device="cpu") as stored:
        metadata = stored.metadata()
        pairs = [(name[:-2], stored.get_tensor(name).cuda(),
                  stored.get_tensor(name[:-2] + "_v").cuda())
                 for name in sorted(stored.keys()) if name.endswith("_p")]
    return pairs, metadata


def candidate(p, v, out):
    return _pv[(25, 8)](
        p, v, out, 32, 64, num_warps=4, num_stages=3,
        enable_fp_fusion=False, enable_reflect_ftz=False)


def measure(args):
    pairs, metadata = load_pairs(args)
    limits = tolerances()["shallow"]
    correctness, controls, candidates = [], [], []
    for name, p, v in pairs:
        ref = torch.empty((400, 256), device="cuda", dtype=torch.bfloat16)
        actual = torch.empty_like(ref)
        torch.mm(p, v, out=ref)
        candidate(p, v, actual)
        metrics = error_metrics(ref, actual)
        assert metrics["rel_rms"] <= limits["rel_rms_max"], (name, metrics)
        assert metrics["cosine_similarity"] >= limits["cosine_min"], (name, metrics)
        correctness.append({"pair": name, **metrics, "bitwise_equal": torch.equal(ref, actual)})
        controls.append((p, v, ref))
        candidates.append((p, v, actual))

    def reference(tensors):
        for p, v, out in tensors:
            torch.mm(p, v, out=out)

    def contender(tensors):
        for p, v, out in tensors:
            candidate(p, v, out)

    rows = []
    for name, function, tensors in (
        ("torch", reference, controls), ("triton", contender, candidates),
        ("triton", contender, candidates), ("torch", reference, controls),
    ):
        samples = bench_gpu_time(
            function, input_args=(tensors,), enable_cupti=False,
            use_cuda_graph=True, cold_l2_cache=False, num_iters_within_graph=16,
            dry_run_iters=5, repeat_iters=30)
        rows.append({"case": name, "median_ms_per_9_calls": median(samples),
                     "samples_ms_per_9_calls": samples})
    report = {"snapshot_metadata": metadata, "tile": [16, 32, 64],
              "correctness": correctness, "timings": rows,
              "cache_policy": "repeat nine distinct actual pairs; warm working set, no L2 flush",
              "timer": "ABBA; fresh capture stream; 16 repetitions of nine calls per graph",
              "timing_boundary": "PV only; immutable inputs, overwrite output; JIT excluded"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "timings": [
        {k: v for k, v in row.items() if k != "samples_ms_per_9_calls"} for row in rows]},
                     indent=2), flush=True)


def diagnose(args):
    pairs, metadata = load_pairs(args)
    _, p, v = pairs[0]
    out = torch.empty((400, 256), device="cuda", dtype=torch.bfloat16)
    compiled = candidate(p, v, out)
    for _ in range(10):
        candidate(p, v, out)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
    ]) as profiler:
        candidate(p, v, out)
        torch.cuda.synchronize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_suffix(".trace.json")
    profiler.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    assert len(kernels) == 1, kernels
    ptx_path = args.out.with_suffix(".ptx")
    ptx_path.write_text(compiled.asm["ptx"])
    report = {"snapshot_metadata": metadata, "diagnostic_only": True,
              "trace": str(trace), "ptx": str(ptx_path),
              "kernel": kernels[0]["name"], "grid": kernels[0]["args"]["grid"],
              "block": kernels[0]["args"]["block"],
              "mma_ptx_lines": sorted({line.strip() for line in compiled.asm["ptx"].splitlines()
                                       if "mma.sync" in line or "tcgen05.mma" in line})}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "time", "trace"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-id")
    args = parser.parse_args()
    {"prepare": prepare, "time": measure, "trace": diagnose}[args.mode](args)


if __name__ == "__main__":
    main()
