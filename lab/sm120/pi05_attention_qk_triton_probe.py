"""Screen three fixed QK tiles while preserving materialized FP32 scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl
from safetensors import safe_open
from safetensors.torch import save_file

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from measurement.kernel_bench import bench_gpu_time


@triton.jit
def _qk(Q, K, Logits, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((32, BN), tl.float32)
    for start in range(0, 256, BK):
        q = tl.load(Q + rows[:, None] * 256 + start + kk[None, :],
                    mask=rows[:, None] < 400, other=0.0)
        k = tl.load(K + cols[None, :] * 256 + start + kk[:, None],
                    mask=cols[None, :] < 1018, other=0.0)
        acc = tl.dot(q, k, acc, out_dtype=tl.float32)
    tl.store(Logits + rows[:, None] * 1018 + cols[None, :], acc,
             mask=(rows[:, None] < 400) & (cols[None, :] < 1018))


def prepare(args):
    from flash_vla.inference import build, resolve

    engine = build(
        resolve("rtx5090/pi05"), "shipped", seed=42,
        converted_checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
        checkpoint_digest=args.checkpoint_id)
    inputs = engine.sample_inputs(42)
    captured = {}

    def inspect(name, function):
        if name != "action_expert_attention":
            return function

        def record(Q, K, V, mask, out, prefix_len=None):
            # Q must be saved before this invocation overwrites its aliased out.
            if not captured:
                captured.update({name: value.detach().cpu().contiguous()
                                 for name, value in (("q", Q), ("k", K),
                                                     ("v", V), ("mask", mask))})
            return function(Q, K, V, mask, out, prefix_len)

        return record

    engine.stage(**inputs)
    with engine.instrument(inspect):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "action_expert":
                engine.run_eager(step.name)
                break
            else:
                engine.replay(step.name)
    assert set(captured) == {"q", "k", "v", "mask"}, captured.keys()
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    save_file(captured, str(args.snapshot), metadata={
        "checkpoint": args.checkpoint_id, "seed": "42", "step": "0", "layer": "0",
        "identity": json.dumps(engine.identity.as_dict()),
        "capture_boundary": "Q/K/V/mask before the first actual attention call"})
    print(json.dumps({"snapshot": str(args.snapshot),
                      "shapes": {name: list(value.shape) for name, value in captured.items()}}))


def measure(args):
    with safe_open(str(args.snapshot), framework="pt", device="cpu") as stored:
        q = stored.get_tensor("q").cuda()
        k = stored.get_tensor("k").cuda()
        metadata = stored.metadata()
    expected = torch.empty((400, 1018), device="cuda", dtype=torch.float32)
    actual = torch.empty_like(expected)
    torch.mm(q, k.T, out_dtype=torch.float32, out=expected)
    limits = tolerances()["shallow"]

    def reference(q, k, logits):
        torch.mm(q, k.T, out_dtype=torch.float32, out=logits)

    results = []
    # Change only BN (416 -> 208 CTAs), then BK (four -> two K iterations).
    for bn, bk in ((32, 64), (64, 64), (64, 128)):
        def candidate(q, k, logits):
            _qk[(13, triton.cdiv(1018, bn))](
                q, k, logits, bn, bk, num_warps=4, num_stages=3,
                enable_fp_fusion=False, enable_reflect_ftz=False)

        candidate(q, k, actual)
        torch.cuda.synchronize()
        metrics = error_metrics(expected, actual)
        assert metrics["rel_rms"] <= limits["rel_rms_max"], metrics
        assert metrics["cosine_similarity"] >= limits["cosine_min"], metrics
        bitwise_equal = torch.equal(expected, actual)

        rows = []
        for name, function, out in (
            ("torch", reference, expected), ("triton", candidate, actual),
            ("triton", candidate, actual), ("torch", reference, expected),
        ):
            samples = bench_gpu_time(
                function, input_args=(q, k, out), enable_cupti=False,
                use_cuda_graph=True, cold_l2_cache=False, num_iters_within_graph=64,
                dry_run_iters=5, repeat_iters=30)
            rows.append({"case": name, "median_ms": median(samples), "samples_ms": samples})
        result = {"tile": [32, bn, bk], "grid": [13, triton.cdiv(1018, bn)],
                  "num_warps": 4, "num_stages": 3, "score_output": metrics,
                  "bitwise_equal": bitwise_equal, "timings": rows}
        results.append(result)
        print(json.dumps({**result, "timings": [
            {key: value for key, value in row.items() if key != "samples_ms"}
            for row in rows]}, indent=2), flush=True)

    report = {
        "snapshot_metadata": metadata,
        "cache_policy": "one actual Q/K pair repeatedly reused; warm-cache screen",
        "timing_boundary": "QK -> FP32 logits only; original softmax/PV excluded",
        "timer": "fresh CUDA graph on capture stream; ABBA per tile; 64 calls per graph",
        "candidates": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "time"))
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
            parser.error("time requires --out")
        measure(args)


if __name__ == "__main__":
    main()
