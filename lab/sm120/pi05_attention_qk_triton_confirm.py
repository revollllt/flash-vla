"""Check the selected 32x32x64 QK at the complete existing attention boundary."""
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
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import fused_attention
from pi05_attention_qk_triton_probe import _qk


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
            step, layer = divmod(index, 18)
            if step in (0, 4, 9) and layer in (0, 8, 17):
                label = f"step{step}_layer{layer}"
                for role, value in (("q", Q), ("k", K), ("v", V), ("mask", mask)):
                    captured[label + "_" + role] = value.detach().cpu().contiguous()
            index += 1
            return function(Q, K, V, mask, out, prefix_len)

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
    assert len(captured) == 36, (index, captured.keys())
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    save_file(captured, str(args.snapshot), metadata={
        "identity": json.dumps(engine.identity.as_dict()), "seed": "42",
        "checkpoint": args.checkpoint_id,
        "capture_boundary": "before actual attention; Q has not been overwritten"})
    print(json.dumps({"snapshot": str(args.snapshot), "pairs": len(captured) // 4}))


def cases(args):
    with safe_open(str(args.snapshot), framework="pt", device="cpu") as stored:
        metadata = stored.metadata()
        labels = [name[:-2] for name in sorted(stored.keys()) if name.endswith("_q")]
        pairs = [(label, *(stored.get_tensor(label + "_" + role).cuda()
                           for role in ("q", "k", "v", "mask"))) for label in labels]
    ref_buffers = {
        "pi05_attention_logits": torch.empty((400, 1018), dtype=torch.float32, device="cuda"),
        "pi05_attention_probabilities": torch.empty((400, 1018), dtype=torch.bfloat16, device="cuda"),
    }

    def scratch(name, shape, dtype, device):
        return ref_buffers[name]

    reference = fused_attention.make_wrappers(scratch)["action_expert_attention"]
    logits = torch.empty_like(ref_buffers["pi05_attention_logits"])
    probabilities = torch.empty_like(ref_buffers["pi05_attention_probabilities"])
    lib = fused_attention._library()

    def candidate(q, k, v, mask, out):
        compiled = _qk[(13, 32)](
            q, k, logits, 32, 64, num_warps=4, num_stages=3,
            enable_fp_fusion=False, enable_reflect_ftz=False)
        rc = lib.pi05_attention_softmax_launch(
            logits.data_ptr(), mask.data_ptr(), probabilities.data_ptr(),
            400, 1018, 0.0625, torch.cuda.current_stream(q.device).cuda_stream)
        if rc:
            raise RuntimeError(f"pi05_attention_softmax CUDA error {rc}")
        torch.mm(probabilities, v, out=out)
        return compiled

    return pairs, metadata, reference, candidate, ref_buffers, logits, probabilities


def measure(args):
    pairs, metadata, reference, candidate, ref_buffers, logits, probabilities = cases(args)
    limits = tolerances()["shallow"]
    correctness, controls, candidates = [], [], []
    for label, q, k, v, mask in pairs:
        ref_q, actual_q = q.clone(), q.clone()
        reference(ref_q, k, v, mask, ref_q, 968)
        candidate(actual_q, k, v, mask, actual_q)
        row = {"pair": label}
        for name, expected, actual in (
            ("logits", ref_buffers["pi05_attention_logits"], logits),
            ("probabilities", ref_buffers["pi05_attention_probabilities"], probabilities),
            ("output", ref_q, actual_q),
        ):
            metrics = error_metrics(expected, actual)
            assert metrics["rel_rms"] <= limits["rel_rms_max"], (label, name, metrics)
            assert metrics["cosine_similarity"] >= limits["cosine_min"], (label, name, metrics)
            row[name] = {**metrics, "bitwise_equal": torch.equal(expected, actual)}
        correctness.append(row)
        controls.append((q, k, v, mask, ref_q))
        candidates.append((q, k, v, mask, actual_q))

    def original(tensors):
        for saved_q, k, v, mask, q in tensors:
            q.copy_(saved_q)
            reference(q, k, v, mask, q, 968)

    def contender(tensors):
        for saved_q, k, v, mask, q in tensors:
            q.copy_(saved_q)
            candidate(q, k, v, mask, q)

    rows = []
    for name, function, tensors in (
        ("control", original, controls), ("candidate", contender, candidates),
        ("candidate", contender, candidates), ("control", original, controls),
    ):
        samples = bench_gpu_time(
            function, input_args=(tensors,), enable_cupti=False,
            use_cuda_graph=True, cold_l2_cache=False, num_iters_within_graph=16,
            dry_run_iters=5, repeat_iters=30)
        rows.append({"case": name, "median_ms_per_9_calls": median(samples),
                     "samples_ms_per_9_calls": samples})
    report = {
        "snapshot_metadata": metadata, "tile": [32, 32, 64],
        "correctness": correctness, "timings": rows,
        "cache_policy": "repeat nine actual pairs; warm working set, no L2 flush",
        "timing_boundary": "identical Q reset + complete attention, out aliases Q",
        "unchanged": "native FP32 masked softmax -> BF16 P and torch.mm(P,V)",
        "timer": "ABBA; fresh capture stream; 16 repetitions of nine calls per graph",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"timings": [
        {key: value for key, value in row.items() if key != "samples_ms_per_9_calls"}
        for row in rows], "all_nine_bitwise": {
            name: all(row[name]["bitwise_equal"] for row in correctness)
            for name in ("logits", "probabilities", "output")}}, indent=2))


def diagnose(args):
    pairs, metadata, _, candidate, _, _, _ = cases(args)
    _, saved_q, k, v, mask = pairs[0]
    q = saved_q.clone()
    compiled = candidate(q, k, v, mask, q)
    for _ in range(10):
        q.copy_(saved_q)
        candidate(q, k, v, mask, q)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
    ]) as profiler:
        q.copy_(saved_q)
        candidate(q, k, v, mask, q)
        torch.cuda.synchronize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_suffix(".trace.json")
    profiler.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    ptx = args.out.with_suffix(".ptx")
    ptx.write_text(compiled.asm["ptx"])
    report = {
        "snapshot_metadata": metadata, "diagnostic_only": True, "trace": str(trace),
        "ptx": str(ptx),
        "reset_copies": [e["name"] for e in events if e.get("cat") == "gpu_memcpy"],
        "kernels": [
            {"name": e["name"], "grid": e["args"]["grid"], "block": e["args"]["block"]}
            for e in kernels],
        "mma_ptx_lines": sorted({line.strip() for line in compiled.asm["ptx"].splitlines()
                                if "mma.sync" in line or "tcgen05.mma" in line}),
    }
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
