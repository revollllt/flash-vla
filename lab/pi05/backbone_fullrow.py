"""Fixed full-row backbone attention; explicit M32 or M16, with BF16 score/P."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import triton
import triton.language as tl

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.bench import bench_gpu_time
from flash_vla.hardware.nvidia.rtx5090.pi05.backends.torch_ops import llm_backbone_attention


@triton.jit
def _attention(Q, K, V, Mask, Out, BLOCK_M: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    keys = tl.arange(0, 1024)
    depth = tl.arange(0, 32)
    scores = tl.zeros((BLOCK_M, 1024), tl.float32)
    # One 64 KiB K tile plus 1 or 2 KiB Q tile; never stage K=64 at this N.
    for start in range(0, 256, 32):
        q = tl.load(Q + rows[:, None] * 256 + start + depth[None, :])
        k = tl.load(K + keys[None, :] * 256 + start + depth[:, None],
                    mask=keys[None, :] < 968, other=0.0)
        scores = tl.dot(q, k, scores, out_dtype=tl.float32)
    # The first explicit BF16 boundary precedes FP32 scale/mask/softmax.
    scores = scores.to(tl.bfloat16).to(tl.float32)
    mask = tl.load(Mask + keys, mask=keys < 968, other=0.0).to(tl.float32)
    logits = tl.where(keys[None, :] < 968, scores * 0.0625 + mask[None, :], -float("inf"))
    shifted = logits - tl.max(logits, axis=1)[:, None]
    numerator = tl.exp(shifted)
    probability = (numerator / tl.sum(numerator, axis=1)[:, None]).to(tl.bfloat16)

    channels = tl.arange(0, 256)
    kk = tl.arange(0, 64)
    output = tl.zeros((BLOCK_M, 256), tl.float32)
    for start in range(0, 1024, 64):
        # Full-row normalization and BF16 rounding have already completed.
        indices = tl.broadcast_to((start + kk)[None, :], (BLOCK_M, 64))
        p = tl.gather(probability, indices, axis=1)
        v = tl.load(V + (start + kk[:, None]) * 256 + channels[None, :],
                    mask=start + kk[:, None] < 968, other=0.0)
        output = tl.dot(p, v, output, out_dtype=tl.float32)
    tl.store(Out + rows[:, None] * 256 + channels[None, :], output.to(tl.bfloat16))


def candidate(q, k, v, mask, out, block_m=32):
    return _attention[(7744 // block_m,)](
        q, k, v, mask, out, block_m, num_warps=8, num_stages=1,
        enable_fp_fusion=False, enable_reflect_ftz=False)


def resources(args):
    """Compile the sole geometry before paying for actual-model capture."""
    q = torch.zeros((7744, 256), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((968, 256), dtype=q.dtype, device=q.device)
    v, mask = torch.zeros_like(k), torch.zeros((968,), dtype=q.dtype, device=q.device)
    compiled = candidate(q, k, v, mask, torch.empty_like(q), args.block_m)
    torch.cuda.synchronize()
    report = {"registers": compiled.n_regs, "spills": compiled.n_spills,
              "shared_bytes": compiled.metadata.shared, "block_m": args.block_m,
              "ctas": 7744 // args.block_m, "warps": 8, "stages": 1,
              "scope": "zero tensors only for compile/resources; not actual-input correctness"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".ptx").write_text(compiled.asm["ptx"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def prepare(args):
    """Record real deployed inputs and outputs without feeding candidate values."""
    from flash_vla.inference import build

    engine = build("rtx5090/pi05", "shipped", seed=42,
                   converted_checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
                   checkpoint_digest=args.checkpoint_id)
    inputs = engine.sample_inputs(42)
    captured, layouts = {}, []

    def inspect(name, function):
        if name != "llm_backbone_attention":
            return function

        def record(q, k, v, scale, mask, out):
            assert q.shape == out.shape == (7744, 256)
            assert k.shape == v.shape == (968, 256) and mask.shape == (968,)
            assert all(value.stride() == (256, 1) for value in (q, k, v, out))
            assert scale == 0.0625
            layer = len(layouts)
            label = f"layer{layer:02d}"
            layouts.append({
                "layer": layer, "scale": scale, "q_alias_out": q.data_ptr() == out.data_ptr(),
                "q_address": q.data_ptr(), "out_address": out.data_ptr(),
                "shapes": {role: list(value.shape) for role, value in
                           (("q", q), ("k", k), ("v", v), ("mask", mask), ("out", out))},
                "strides": {role: list(value.stride()) for role, value in
                            (("q", q), ("k", k), ("v", v), ("mask", mask), ("out", out))},
            })
            for role, value in (("q", q), ("k", k), ("v", v), ("mask", mask)):
                captured[f"{label}_{role}"] = value.detach().cpu().contiguous()
            result = function(q, k, v, scale, mask, out)
            captured[f"{label}_out"] = out.detach().cpu().contiguous()
            return result

        return record

    engine.stage(**inputs)
    with engine.instrument(inspect):
        for step in engine.program:
            if step.kind == "host":
                engine.host(step.name, **inputs)
            elif step.name == "llm_backbone":
                engine.run_eager(step.name)
            else:
                engine.replay(step.name)
    assert len(layouts) == 17, layouts
    assert not any(row["q_alias_out"] for row in layouts), layouts
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    save_file(captured, str(args.snapshot), metadata={
        "identity": json.dumps(engine.identity.as_dict()), "seed": "42",
        "checkpoint": args.checkpoint_id, "layouts": json.dumps(layouts),
        "capture": "sample_inputs(42); deployed replay to backbone, eager backbone with original outputs",
    })


def _cases(args):
    with safe_open(str(args.snapshot), framework="pt", device="cpu") as stored:
        metadata = stored.metadata()
        cases = [
            tuple(stored.get_tensor(f"layer{layer:02d}_{role}").cuda()
                  for role in ("q", "k", "v", "mask", "out"))
            for layer in range(17)
        ]
    # Actual backbone uses shared Q/out addresses across layers, without alias.
    q, out = torch.empty_like(cases[0][0]), torch.empty_like(cases[0][0])
    return cases, metadata, q, out


def check(args):
    cases, metadata, q, out = _cases(args)
    limit = tolerances()["shallow"]
    rows = []
    compiled = None
    for layer, (saved_q, k, v, mask, expected) in enumerate(cases):
        q.copy_(saved_q)
        llm_backbone_attention(q, k, v, 0.0625, mask, out)
        control = error_metrics(expected, out)
        q.copy_(saved_q)
        compiled = candidate(q, k, v, mask, out, args.block_m)
        metrics = error_metrics(expected, out)
        finite = bool(torch.isfinite(out).all())
        valid = int((mask == 0).sum()) * 8
        row = {"layer": layer, "control": control, "candidate": metrics,
               "valid_queries": valid, "all_rows_finite": finite,
               "valid_output": error_metrics(expected[:valid], out[:valid])}
        row["passed"] = finite and all(
            m["rel_rms"] < limit["rel_rms_max"] and m["cosine_similarity"] > limit["cosine_min"]
            for m in (control, metrics))
        rows.append(row)
        if not row["passed"]:
            break
    torch.cuda.synchronize()
    report = {
        "phase": "check", "snapshot_metadata": metadata, "tile": [args.block_m, 1024, 32],
        "pv_tile": [args.block_m, 256, 64], "warps": 8, "stages": 1,
        "ctas": 7744 // args.block_m,
        "registers": compiled.n_regs, "spills": compiled.n_spills,
        "shared_bytes": compiled.metadata.shared, "tolerance": limit,
        "rows": rows, "passed": len(rows) == 17 and all(row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".ptx").write_text(compiled.asm["ptx"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def measure(args):
    cases, metadata, q, out = _cases(args)
    # Both routes restore each real Q into the same static address inside timing.
    def chain(route):
        for saved_q, k, v, mask, _ in cases:
            q.copy_(saved_q)
            if route == "control":
                llm_backbone_attention(q, k, v, 0.0625, mask, out)
            else:
                candidate(q, k, v, mask, out, args.block_m)

    rows = []
    for route in ("control", "candidate", "candidate", "control"):
        samples = bench_gpu_time(
            chain, input_args=(route,), enable_cupti=False, use_cuda_graph=True,
            cold_l2_cache=False, num_iters_within_graph=4,
            dry_run_iters=5, repeat_iters=30)
        rows.append({"route": route, "median_ms_per_17": median(samples), "samples_ms": samples})
    a1, b1, b2, a2 = [row["median_ms_per_17"] for row in rows]
    report = {
        "phase": "single-ABBA", "block_m": args.block_m,
        "snapshot_metadata": metadata, "timings": rows,
        "gain_ms_per_17": (a1 + a2 - b1 - b2) / 2,
        "control_drift_ms": abs(a2 - a1), "candidate_drift_ms": abs(b2 - b1),
        "conservative_gain_ms": min(a1, a2) - max(b1, b2),
        "boundary": "Q reset + full BF16-score/BF16-probability attention, same Q/K/V/mask/out addresses",
        "cache": "17 distinct real pairs; repeated working set, no L2 flush; clocks not locked",
        "timer": "CUDA events on fresh capture stream per leg; 4 chains per graph; warm5/repeat30",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("resources", "prepare", "check", "time"))
    parser.add_argument("--block-m", type=int, choices=(16, 32), default=32)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-id")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
        return 0
    return {"resources": resources, "check": check, "time": measure}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
