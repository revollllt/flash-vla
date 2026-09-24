"""Expand the unchanged GR00T reference using the project's cost/floor helpers.

This is a conditional estimate for the current dense computation, not a bound
over every equivalent algorithm. No model weights or GPU are needed.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from flash_vla.inference import build
from flash_vla.runtime.cost import Cost, Invocation, attention, gemm
from flash_vla.hardware.nvidia.rtx5090.spec import ROOFLINE
from tools.profiling.floor import datasheet, load_constants, site_row


def invocations():
    rows = []

    def linear(stage, name, m, k, n, count):
        rows.append((stage, name, gemm(m, k, n), 2 * k * n, count,
                     {"kind": "linear", "m": m, "k": k, "n": n}))

    def attn(stage, name, q, k, d, heads, kv_heads, count):
        rows.append((stage, name, attention(q, k, d, heads, kv_heads), 0, count,
                     {"kind": "attention", "queries": q, "keys": k,
                      "head_dim": d, "query_heads": heads, "kv_heads": kv_heads}))

    s = "vision_encoder"
    linear(s, "patch_projection", 512, 1536, 1024, 1)
    linear(s, "qkv", 512, 1024, 3072, 24)
    linear(s, "out_projection", 512, 1024, 1024, 24)
    linear(s, "ffn_up", 512, 1024, 4096, 24)
    linear(s, "ffn_down", 512, 4096, 1024, 24)
    attn(s, "attention", 256, 256, 64, 16, 16, 24 * 2)
    linear(s, "merger_fc1", 128, 4096, 4096, 4)
    linear(s, "merger_fc2", 128, 4096, 2048, 4)

    s = "llm_backbone"
    linear(s, "query_and_out", 156, 2048, 2048, 16 * 2)
    linear(s, "key_and_value", 156, 2048, 1024, 16 * 2)
    linear(s, "gate_and_up", 156, 2048, 6144, 16 * 2)
    linear(s, "down", 156, 6144, 2048, 16)
    attn(s, "attention", 156, 156, 128, 16, 8, 16)

    s = "action_expert"
    linear(s, "refiner_qkvo", 156, 2048, 2048, 4 * 4)
    linear(s, "refiner_ffn_up", 156, 2048, 8192, 4)
    linear(s, "refiner_ffn_down", 156, 8192, 2048, 4)
    attn(s, "refiner_attention", 156, 156, 64, 32, 32, 4)
    linear(s, "state_fc1", 1, 132, 1024, 1)
    linear(s, "state_fc2", 1, 1024, 1536, 1)
    linear(s, "action_encoder_w1", 40, 132, 1536, 4)
    linear(s, "action_encoder_w2", 40, 3072, 1536, 4)
    linear(s, "action_encoder_w3", 40, 1536, 1536, 4)
    linear(s, "time_fc1", 1, 256, 1536, 4)
    linear(s, "time_fc2", 1, 1536, 1536, 4)
    linear(s, "dit_adaln", 1, 1536, 3072, 32 * 4)
    linear(s, "dit_self_qkvo", 41, 1536, 1536, 16 * 4 * 4)
    attn(s, "dit_self_attention", 41, 41, 48, 32, 32, 16 * 4)
    linear(s, "dit_cross_query_and_out", 41, 1536, 1536, 16 * 4 * 2)
    linear(s, "dit_cross_key_and_value", 156, 2048, 1536, 16 * 4 * 2)
    attn(s, "dit_cross_attention", 41, 156, 48, 32, 32, 16 * 4)
    linear(s, "dit_ffn_up", 41, 1536, 6144, 32 * 4)
    linear(s, "dit_ffn_down", 41, 6144, 1536, 32 * 4)
    linear(s, "output_modulation", 1, 1536, 3072, 4)
    linear(s, "output_projection", 41, 1536, 1024, 4)
    linear(s, "action_decoder_fc1", 41, 1024, 1024, 4)
    linear(s, "action_decoder_fc2", 41, 1024, 132, 4)
    return rows


def estimate():
    constants, _ = load_constants(ROOFLINE.constants_file, ROOFLINE.constant_tags)
    peaks = datasheet(ROOFLINE)
    stages = defaultdict(lambda: defaultdict(float))
    detail = []
    for stage, name, cost, weight_bytes, count, geometry in invocations():
        # Weight streaming assumes these matrices are not persistently retained
        # between invocations. Activations are ideally reused in cache/on chip.
        weight_cost = Cost(weight_bytes, 0, cost.flops)
        streamed = site_row(Invocation(name, weight_cost, count), peaks, constants)
        materialized = site_row(Invocation(name, cost, count), peaks, constants)
        values = {
            "flops": cost.flops * count,
            "streamed_weight_bytes": weight_bytes * count,
            "materialized_bytes": cost.bytes * count,
            "compute_only_ms": cost.flops * count / peaks["bf16_fps"] * 1000,
            "weight_stream_roofline_ms": streamed["roofline_us"] / 1000,
            "materialized_roofline_ms": materialized["roofline_us"] / 1000,
            "measured_capability_estimate_ms": streamed["ceiling_us"] / 1000,
        }
        for key, value in values.items():
            stages[stage][key] += value
        detail.append({"stage": stage, "name": name, "count": count,
                       "geometry": geometry, **values})

    # Independently compare the expanded dense math against all three existing
    # stage declarations; traffic intentionally differs from those declarations.
    engine = build("groot-n17", "reference", declare=True, device="cpu")
    declared = {stage: sum(inv.flops for inv in invs) for stage, invs in engine.costs.items()}
    for stage, values in stages.items():
        if values["flops"] != declared[stage]:
            raise ValueError(f"{stage}: expanded FLOPs {values['flops']} != declared {declared[stage]}")

    totals = {key: sum(stage[key] for stage in stages.values()) for key in next(iter(stages.values()))}
    report = {
        "target": "rtx5090/groot_n17", "reference_commit": "cf2d196",
        "status": "conditional estimate; not measured attainable model performance",
        "workload": {"checkpoint": "GR00T-N1.7-LIBERO/libero_10", "precision": "bf16",
                     "sequence_length": 156, "views": 2, "steps": 4, "action_horizon": 40},
        "method": "flash_vla.runtime.cost.gemm/attention + tools.profiling.floor.site_row",
        "peaks": peaks,
        "measured_primitives": {"stream": constants["stream"], "tensor": constants["tensor"]},
        "assumptions": [
            "Use BF16 input / FP32 accumulation dense throughput, never sparse or FP8 peaks.",
            "Sum ideal per-operation costs for the current dense computation and four denoising steps.",
            "Weight-stream model reads each selected matrix once per invocation; 32 embodiment banks are not all read.",
            "Embedding tables are gathers, not full-table streams; lookup, bias, normalization and other pointwise costs are omitted.",
            "Weight-stream model assumes ideal activation reuse and no persistent weight reuse between invocations; L2 is 96 MiB.",
            "Materialized-traffic variant additionally prices every GEMM/attention input and output at DRAM; this is a sensitivity case, not the chosen denominator.",
            "Launch, scheduling, synchronization, input staging, preprocessing and postprocessing overheads are omitted.",
            "Static-time folding, cross-step KV reuse or other changes to required computation require recomputing this estimate.",
            "Datasheet uses 209.6 TFLOP/s at marketed clocks; the measured primitive uses 253 TFLOP/s at its observed clocks. They are different conditions.",
            "A 90% objective is T <= estimated_floor/0.9 under this model; attaining it is unproven and does not override the run budget.",
        ],
        "stages": stages, "totals": totals,
        "goal_90pct_ms": totals["weight_stream_roofline_ms"] / 0.9,
        "expanded_flops_match_stage_declarations": True,
        "omitted_fp32_rope_flops": 2 * 3 * 64 * 156,
        "execution_check_record": "results/groot-n17-rtx5090/reference/roofline-execution-check.json",
        "operations": detail,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/groot-n17/roofline.json"))
    args = parser.parse_args()
    report = estimate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "stages": report["stages"],
                      "totals": report["totals"], "goal_90pct_ms": report["goal_90pct_ms"]}, indent=2))


if __name__ == "__main__":
    main()
