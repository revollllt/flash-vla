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
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels import (
    adarms,
    xfs as xfs_kernels,
)
from .xfs_producer import out_proj_residual_rms_xfs_reference


M, M_PAD, ATTENTION_K, D, FF = 50, 64, 2048, 1024, 4096


def _rand(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
            * 0.05).bfloat16()


def _pack_gate_up(gate_w: torch.Tensor, up_w: torch.Tensor) -> torch.Tensor:
    """[D,FF] pairs -> task-major [FF/32*D,64], matching persistent GU."""
    def tiles(weight: torch.Tensor) -> torch.Tensor:
        return (weight.reshape(D, FF // 32, 32).permute(1, 0, 2)
                .contiguous().view(-1, 32))

    return torch.cat((tiles(gate_w), tiles(up_w)), dim=1).contiguous()


def _make_case(gen: torch.Generator) -> dict[str, torch.Tensor]:
    gate_w = _rand(gen, D, FF)
    up_w = _rand(gen, D, FF)
    return {
        # Real decoder_out_proj_residual inputs.
        "attention": _rand(gen, M, ATTENTION_K),
        "attention_weight": _rand(gen, ATTENTION_K, D),
        "attention_gate": _rand(gen, D),
        "residual_seed": _rand(gen, M, D),
        # Next FFN's folded AdaRMS parameters and projections.
        "ffn_scale": (1.0 + _rand(gen, D)).bfloat16(),
        "gate_w": gate_w,
        "up_w": up_w,
        "gate_b": _rand(gen, FF),
        "up_b": _rand(gen, FF),
        "packed_gate_up": _pack_gate_up(gate_w, up_w),
        # Independent destinations keep the two paths comparable after capture.
        "x_xfs": torch.empty((M, D), dtype=torch.bfloat16, device="cuda"),
        "x_fused": torch.empty((M, D), dtype=torch.bfloat16, device="cuda"),
        "x_base": torch.empty((M, D), dtype=torch.bfloat16, device="cuda"),
        "factor": torch.empty((M,), dtype=torch.bfloat16, device="cuda"),
        "xfs": torch.empty((D, M_PAD), dtype=torch.bfloat16, device="cuda"),
        "xfs_fused": torch.empty((D, M_PAD), dtype=torch.bfloat16, device="cuda"),
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
    parser.add_argument("--task-table", choices=("gu", "full"), default="gu")
    args = parser.parse_args()

    gen = torch.Generator(device="cuda").manual_seed(47)
    cases = [_make_case(gen) for _ in range(args.sets)]
    table = build_table(args.task_table).to("cuda")
    taskloop = FFNTaskloop(verbose=True)

    # Keep one set of valid DownResidual operands for both task-table modes;
    # they remain part of the stable taskloop C ABI even in GU-only mode.
    legacy_factor = torch.empty((M_PAD,), dtype=torch.bfloat16, device="cuda")
    down_weight = _rand(gen, FF, D)
    down_gate = _rand(gen, D)
    residual_out = torch.empty((M_PAD, D), dtype=torch.bfloat16, device="cuda")
    counters = torch.empty((N_COUNTERS,), dtype=torch.int32, device="cuda")
    hidden_ready, down_ready = taskloop.readiness_counter_buffers(counters)
    square_partials = torch.empty(
        (4, 32, 16), dtype=torch.float32, device="cuda")

    tilelang_gate = wrappers._compiled(
        adarms.tl_ada_scaled_gate, M=M, N=FF, K=D, **wrappers._DEC_GATE)
    partial_producer = wrappers._compiled(
        xfs_kernels.tl_out_proj_residual_partials,
        M=M, N=D, K=ATTENTION_K, **wrappers._DEC_OUT_PROJ_PARTIALS)
    partial_xfs = wrappers._compiled(
        xfs_kernels.tl_rms_xfs_from_partials,
        M=M, N=D, **wrappers._DEC_XFS_FROM_PARTIALS)

    def legacy_out_proj(case) -> None:
        case["x_xfs"].copy_(case["residual_seed"])
        wrappers.decoder_out_proj_residual(
            case["attention"], case["attention_weight"],
            case["attention_gate"], case["x_xfs"])

    def legacy_xfs(case) -> None:
        wrappers.decoder_rms_xfs(
            case["x_xfs"], case["ffn_scale"], hidden_ready, down_ready,
            case["xfs"], trigger_programmatic_launch=True)

    def legacy_xfs_before_reset_fusion(case) -> None:
        wrappers.decoder_rms_xfs(
            case["x_xfs"], case["ffn_scale"], hidden_ready, down_ready,
            case["xfs"], reset_readiness=False)

    def out_proj_partials(case) -> None:
        case["x_fused"].copy_(case["residual_seed"])
        partial_producer(
            case["attention"], case["attention_weight"],
            case["attention_gate"], case["x_fused"],
            hidden_ready, down_ready, square_partials)

    def xfs_from_partials(case) -> None:
        partial_xfs(
            case["x_fused"], case["ffn_scale"], square_partials,
            case["xfs_fused"])

    def legacy_producer(case) -> None:
        legacy_out_proj(case)
        legacy_xfs(case)

    def fused_producer(case) -> None:
        out_proj_partials(case)
        xfs_from_partials(case)

    def xfs_path(case) -> None:
        legacy_producer(case)
        taskloop.launch(
            table, case["xfs"], legacy_factor, case["ffn_scale"],
            case["packed_gate_up"], case["packed_gate_up"],
            case["gate_b"], case["up_b"], down_weight, down_gate,
            case["hidden_xfs"], residual_out, counters,
            zero_counters=False,
            use_programmatic_dependency=True,
        )

    def standalone_reset_path(case) -> None:
        legacy_out_proj(case)
        legacy_xfs_before_reset_fusion(case)
        taskloop.launch(
            table, case["xfs"], legacy_factor, case["ffn_scale"],
            case["packed_gate_up"], case["packed_gate_up"],
            case["gate_b"], case["up_b"], down_weight, down_gate,
            case["hidden_xfs"], residual_out, counters,
            zero_counters=True,
        )

    def fused_xfs_path(case) -> None:
        fused_producer(case)
        taskloop.launch(
            table, case["xfs_fused"], legacy_factor, case["ffn_scale"],
            case["packed_gate_up"], case["packed_gate_up"],
            case["gate_b"], case["up_b"], down_weight, down_gate,
            case["hidden_xfs"], residual_out, counters,
            zero_counters=False,
            use_programmatic_dependency=True,
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
        hidden_ready.fill_(7)
        down_ready.fill_(9)
        fused_producer(case)
        xfs_path(case)
        baseline_path(case)
    torch.cuda.synchronize()

    for index, case in enumerate(cases):
        residual_exact = torch.equal(case["x_xfs"], case["x_base"])
        fused_residual_exact = torch.equal(case["x_fused"], case["x_xfs"])
        fused_xfs_exact = torch.equal(case["xfs_fused"], case["xfs"])
        fused_pad_nonzero = torch.count_nonzero(case["xfs_fused"][:, M:]).item()
        old_xfs = torch.zeros_like(case["xfs"])
        old_xfs[:, :M] = (
            ((case["x_base"] * case["factor"][:, None]).bfloat16()
             * case["ffn_scale"][None, :]).bfloat16().T
        )
        producer_exact = torch.equal(case["xfs"], old_xfs)
        padding_nonzero = torch.count_nonzero(case["xfs"][:, M:]).item()
        hidden_cos, hidden_max = _metrics(
            case["hidden_base"], case["hidden_xfs"][:M])
        torch_x, torch_xfs = out_proj_residual_rms_xfs_reference(
            case["attention"], case["attention_weight"],
            case["attention_gate"], case["residual_seed"],
            case["ffn_scale"])
        torch_x_cos, torch_x_max = _metrics(torch_x, case["x_base"])
        torch_xfs_cos, torch_xfs_max = _metrics(torch_xfs, case["xfs"])
        print(
            f"[chain] set={index} residual_exact={residual_exact} "
            f"producer_exact={producer_exact} pad_nonzero={padding_nonzero} "
            f"fused_residual_exact={fused_residual_exact} "
            f"fused_xfs_exact={fused_xfs_exact} "
            f"fused_pad_nonzero={fused_pad_nonzero} "
            f"hidden_cos={hidden_cos:.7f} hidden_max_abs={hidden_max:.6g} "
            f"torch_x_cos={torch_x_cos:.7f} torch_x_max={torch_x_max:.6g} "
            f"torch_xfs_cos={torch_xfs_cos:.7f} "
            f"torch_xfs_max={torch_xfs_max:.6g}",
            flush=True,
        )
        if (not residual_exact or not producer_exact or padding_nonzero
                or not fused_residual_exact or not fused_xfs_exact
                or fused_pad_nonzero):
            raise SystemExit(f"chain semantic mismatch in set {index}")
        if hidden_cos < 0.999:
            raise SystemExit(f"persistent/TileLang hidden mismatch in set {index}")

    graph_xfs = _capture(cases, xfs_path)
    graph_fused_xfs = _capture(cases, fused_xfs_path)
    graph_standalone_reset = _capture(cases, standalone_reset_path)
    graph_base = _capture(cases, baseline_path)
    graph_legacy_producer = _capture(cases, legacy_producer)
    graph_fused_producer = _capture(cases, fused_producer)
    split_graphs = {
        "legacy_out_proj": _capture(cases, legacy_out_proj),
        "legacy_xfs": _capture(cases, legacy_xfs),
        "partial_producer": _capture(cases, out_proj_partials),
        "partial_xfs": _capture(cases, xfs_from_partials),
    }
    standalone_reset_us = _time_graph(
        graph_standalone_reset, len(cases), args.reps)
    fused_cumulative_us = _time_graph(
        graph_fused_xfs, len(cases), args.reps)
    closing_standalone_reset_us = _time_graph(
        graph_standalone_reset, len(cases), args.reps)
    xfs_us = _time_graph(graph_xfs, len(cases), args.reps)
    fused_xfs_us = _time_graph(graph_fused_xfs, len(cases), args.reps)
    closing_xfs_us = _time_graph(graph_xfs, len(cases), args.reps)
    base_us = _time_graph(graph_base, len(cases), args.reps)
    legacy_producer_us = _time_graph(
        graph_legacy_producer, len(cases), args.reps)
    fused_producer_us = _time_graph(
        graph_fused_producer, len(cases), args.reps)
    split_us = {
        name: _time_graph(graph, len(cases), args.reps)
        for name, graph in split_graphs.items()
    }
    print(
        f"[chain] sets={len(cases)} xfs_persistent_gu={xfs_us:.3f} us "
        f"factor_tilelang_gate={base_us:.3f} us delta={xfs_us - base_us:+.3f} us",
        flush=True,
    )
    xfs_midpoint = (xfs_us + closing_xfs_us) / 2.0
    print(
        f"[chain-pdl] legacy={xfs_us:.3f} us "
        f"fused={fused_xfs_us:.3f} us closing_legacy={closing_xfs_us:.3f} us "
        f"midpoint_delta={fused_xfs_us - xfs_midpoint:+.3f} us",
        flush=True,
    )
    reset_midpoint = (
        standalone_reset_us + closing_standalone_reset_us) / 2.0
    print(
        f"[chain-cumulative] standalone_reset={standalone_reset_us:.3f} us "
        f"fused={fused_cumulative_us:.3f} us "
        f"closing_standalone_reset={closing_standalone_reset_us:.3f} us "
        f"midpoint_gain={reset_midpoint - fused_cumulative_us:+.3f} us",
        flush=True,
    )
    print(
        f"[producer] legacy={legacy_producer_us:.3f} us "
        f"fused={fused_producer_us:.3f} us "
        f"delta={fused_producer_us - legacy_producer_us:+.3f} us",
        flush=True,
    )
    print(
        "[producer-split] "
        + " ".join(f"{name}={value:.3f} us"
                   for name, value in split_us.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
