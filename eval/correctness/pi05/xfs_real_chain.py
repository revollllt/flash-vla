"""Real FFN-prefix integration experiment for the persistent GatedProjection.

Both paths start from the real Pi0.5 attention output-projection residual
kernel and the same tensors:

    XFS path: residual GEMM -> RMS/XFS producer -> persistent GU-only
    baseline: residual GEMM -> RMS factor       -> TileLang gated FFN

This is an opt-in harness only; it does not change the default pipeline.
"""
from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from flash_vla.hardware.nvidia.h100.pi05.backends.cuda.taskloop import (
    FFNTaskloop,
    N_COUNTERS,
    build_table,
)
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels import adarms


M, M_PAD, K, FF = 50, 64, 1024, 4096


def _rand(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
            * 0.05).bfloat16()


def _pack_gate_up(gate_w: torch.Tensor, up_w: torch.Tensor) -> torch.Tensor:
    """[K,FF] pairs -> task-major [FF/32*K,64], matching persistent GU."""
    def tiles(weight: torch.Tensor) -> torch.Tensor:
        return (weight.reshape(K, FF // 32, 32).permute(1, 0, 2)
                .contiguous().view(-1, 32))

    return torch.cat((tiles(gate_w), tiles(up_w)), dim=1).contiguous()


def _make_case(gen: torch.Generator) -> dict[str, torch.Tensor]:
    gate_w = _rand(gen, K, FF)
    up_w = _rand(gen, K, FF)
    return {
        # Real decoder_out_proj_residual inputs.
        "attention": _rand(gen, M, K),
        "attention_weight": _rand(gen, K, K),
        "attention_gate": _rand(gen, K),
        "residual_seed": _rand(gen, M, K),
        # Next FFN's folded AdaRMS parameters and projections.
        "ffn_scale": (1.0 + _rand(gen, K)).bfloat16(),
        "gate_w": gate_w,
        "up_w": up_w,
        "gate_b": _rand(gen, FF),
        "up_b": _rand(gen, FF),
        "packed_gate_up": _pack_gate_up(gate_w, up_w),
        # Independent destinations keep the two paths comparable after capture.
        "x_xfs": torch.empty((M, K), dtype=torch.bfloat16, device="cuda"),
        "x_base": torch.empty((M, K), dtype=torch.bfloat16, device="cuda"),
        "factor": torch.empty((M,), dtype=torch.bfloat16, device="cuda"),
        "xfs": torch.empty((K, M_PAD), dtype=torch.bfloat16, device="cuda"),
        "hidden_xfs": torch.empty((M_PAD, FF), dtype=torch.bfloat16, device="cuda"),
        "hidden_base": torch.empty((M, FF), dtype=torch.bfloat16, device="cuda"),
    }


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    ref = reference.float().flatten()
    act = actual.float().flatten()
    return F.cosine_similarity(ref, act, dim=0).item(), (ref - act).abs().max().item()


def _capture(cases, body) -> torch.cuda.CUDAGraph:
    for case in cases:
        body(case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            body(case)
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, launches: int, reps: int) -> float:
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / launches)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", type=int, default=3)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()

    gen = torch.Generator(device="cuda").manual_seed(47)
    cases = [_make_case(gen) for _ in range(args.sets)]
    table = build_table("gu").to("cuda")
    taskloop = FFNTaskloop(verbose=True)

    # GU-only does not consume these DownResidual operands. Keep valid shared
    # allocations because they remain part of the stable taskloop C ABI.
    legacy_factor = torch.empty((M_PAD,), dtype=torch.bfloat16, device="cuda")
    down_weight = torch.empty((FF, K), dtype=torch.bfloat16, device="cuda")
    down_gate = torch.empty((K,), dtype=torch.bfloat16, device="cuda")
    residual_out = torch.empty((M_PAD, K), dtype=torch.bfloat16, device="cuda")
    counters = torch.empty((N_COUNTERS,), dtype=torch.int32, device="cuda")
    hidden_ready, down_ready = taskloop.readiness_counter_buffers(counters)

    tilelang_gate = wrappers._compiled(
        adarms.tl_ada_scaled_gate, M=M, N=FF, K=K, **wrappers._DEC_GATE)

    def xfs_path(case) -> None:
        case["x_xfs"].copy_(case["residual_seed"])
        wrappers.decoder_out_proj_residual(
            case["attention"], case["attention_weight"],
            case["attention_gate"], case["x_xfs"])
        wrappers.decoder_rms_xfs(
            case["x_xfs"], case["ffn_scale"], hidden_ready, down_ready,
            case["xfs"])
        taskloop.launch(
            table, case["xfs"], legacy_factor, case["ffn_scale"],
            case["packed_gate_up"], case["packed_gate_up"],
            case["gate_b"], case["up_b"], down_weight, down_gate,
            case["hidden_xfs"], residual_out, counters,
            zero_counters=False,
        )

    def baseline_path(case) -> None:
        case["x_base"].copy_(case["residual_seed"])
        wrappers.decoder_out_proj_residual(
            case["attention"], case["attention_weight"],
            case["attention_gate"], case["x_base"])
        wrappers._rms_factor(case["x_base"], case["factor"])
        tilelang_gate(
            case["x_base"], case["factor"], case["ffn_scale"],
            case["gate_w"], case["up_w"], case["gate_b"], case["up_b"],
            case["hidden_base"],
        )

    # One real-chain execution supplies the semantic comparison before timing.
    for case in cases:
        xfs_path(case)
        baseline_path(case)
    torch.cuda.synchronize()

    for index, case in enumerate(cases):
        residual_exact = torch.equal(case["x_xfs"], case["x_base"])
        old_xfs = torch.zeros_like(case["xfs"])
        old_xfs[:, :M] = (
            ((case["x_base"] * case["factor"][:, None]).bfloat16()
             * case["ffn_scale"][None, :]).bfloat16().T
        )
        producer_exact = torch.equal(case["xfs"], old_xfs)
        padding_nonzero = torch.count_nonzero(case["xfs"][:, M:]).item()
        hidden_cos, hidden_max = _metrics(
            case["hidden_base"], case["hidden_xfs"][:M])
        print(
            f"[chain] set={index} residual_exact={residual_exact} "
            f"producer_exact={producer_exact} pad_nonzero={padding_nonzero} "
            f"hidden_cos={hidden_cos:.7f} hidden_max_abs={hidden_max:.6g}",
            flush=True,
        )
        if not residual_exact or not producer_exact or padding_nonzero:
            raise SystemExit(f"chain semantic mismatch in set {index}")
        if hidden_cos < 0.999:
            raise SystemExit(f"persistent/TileLang hidden mismatch in set {index}")

    graph_xfs = _capture(cases, xfs_path)
    graph_base = _capture(cases, baseline_path)
    xfs_us = _time_graph(graph_xfs, len(cases), args.reps)
    base_us = _time_graph(graph_base, len(cases), args.reps)
    print(
        f"[chain] sets={len(cases)} xfs_persistent_gu={xfs_us:.3f} us "
        f"factor_tilelang_gate={base_us:.3f} us delta={xfs_us - base_us:+.3f} us",
        flush=True,
    )


if __name__ == "__main__":
    main()
