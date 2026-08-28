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
    COUNTER_ARRIVE,
    DOWN_RESIDUAL_SPLIT,
    FFNTaskloop,
    N_COUNTERS,
    N_CTAS_FULL,
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
        "x_cooperative": torch.empty(
            (M, D), dtype=torch.bfloat16, device="cuda"),
        "x_base": torch.empty((M, D), dtype=torch.bfloat16, device="cuda"),
        "factor": torch.empty((M,), dtype=torch.bfloat16, device="cuda"),
        "xfs": torch.empty((D, M_PAD), dtype=torch.bfloat16, device="cuda"),
        "xfs_fused": torch.empty((D, M_PAD), dtype=torch.bfloat16, device="cuda"),
        "xfs_cooperative": torch.empty(
            (D, M_PAD), dtype=torch.bfloat16, device="cuda"),
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


def _time_graph_replays(
        graph: torch.cuda.CUDAGraph, replays: int, reps: int,
) -> tuple[float, float, float]:
    """Return median total/per-replay GPU time and their arithmetic residual."""
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    total_us = statistics.median(samples)
    per_replay_us = total_us / replays
    residual_us = abs(total_us - per_replay_us * replays)
    return total_us, per_replay_us, residual_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", type=int, default=3)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--task-table", choices=("gu", "full"), default="gu")
    parser.add_argument("--cooperative-candidate", action="store_true")
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

    def cooperative_fused_producer(case) -> None:
        case["x_cooperative"].copy_(case["residual_seed"])
        wrappers.decoder_out_proj_residual_rms_xfs(
            case["attention"], case["attention_weight"],
            case["attention_gate"], case["x_cooperative"],
            case["ffn_scale"], hidden_ready, down_ready,
            square_partials, case["xfs_cooperative"],
        )

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

    def cooperative_xfs_path(case) -> None:
        cooperative_fused_producer(case)
        taskloop.launch(
            table, case["xfs_cooperative"], legacy_factor, case["ffn_scale"],
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
        split_counters_exact = True
        cooperative_counters_exact = True
        cooperative_residual_exact = True
        cooperative_xfs_exact = True
        cooperative_pad_nonzero = 0
        if args.cooperative_candidate:
            hidden_ready.fill_(7)
            down_ready.fill_(9)
            fused_producer(case)
            torch.cuda.synchronize()
            split_counters_exact = (
                torch.count_nonzero(hidden_ready).item() == 0
                and torch.count_nonzero(down_ready).item() == 0)
            hidden_ready.fill_(7)
            down_ready.fill_(9)
            cooperative_fused_producer(case)
            torch.cuda.synchronize()
            cooperative_counters_exact = (
                torch.count_nonzero(hidden_ready).item() == 0
                and torch.count_nonzero(down_ready).item() == 0)
            cooperative_residual_exact = torch.equal(
                case["x_cooperative"], case["x_fused"])
            cooperative_xfs_exact = torch.equal(
                case["xfs_cooperative"], case["xfs_fused"])
            cooperative_pad_nonzero = torch.count_nonzero(
                case["xfs_cooperative"][:, M:]).item()
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
            f"coop_residual_exact={cooperative_residual_exact} "
            f"coop_xfs_exact={cooperative_xfs_exact} "
            f"coop_pad_nonzero={cooperative_pad_nonzero} "
            f"split_counters_exact={split_counters_exact} "
            f"coop_counters_exact={cooperative_counters_exact} "
            f"fused_pad_nonzero={fused_pad_nonzero} "
            f"hidden_cos={hidden_cos:.7f} hidden_max_abs={hidden_max:.6g} "
            f"torch_x_cos={torch_x_cos:.7f} torch_x_max={torch_x_max:.6g} "
            f"torch_xfs_cos={torch_xfs_cos:.7f} "
            f"torch_xfs_max={torch_xfs_max:.6g}",
            flush=True,
        )
        if (not residual_exact or not producer_exact or padding_nonzero
                or not fused_residual_exact or not fused_xfs_exact
                or not cooperative_residual_exact or not cooperative_xfs_exact
                or cooperative_pad_nonzero or not split_counters_exact
                or not cooperative_counters_exact
                or fused_pad_nonzero):
            raise SystemExit(f"chain semantic mismatch in set {index}")
        if hidden_cos < 0.999:
            raise SystemExit(f"persistent/TileLang hidden mismatch in set {index}")

    if args.cooperative_candidate:
        graph_standalone_reset = _capture(cases, standalone_reset_path)
        graph_split_full = _capture(cases, fused_xfs_path)
        graph_cooperative_full = _capture(cases, cooperative_xfs_path)
        graph_split_producer = _capture(cases, fused_producer)
        graph_cooperative_producer = _capture(
            cases, cooperative_fused_producer)
        graph_standalone_180 = _capture(cases[:1], standalone_reset_path)
        graph_cooperative_180 = _capture(cases[:1], cooperative_xfs_path)

        hidden_ready.fill_(17)
        down_ready.fill_(19)
        for _ in range(20):
            graph_cooperative_full.replay()
        torch.cuda.synchronize()
        replay_hidden_terminal = bool(torch.all(
            hidden_ready == COUNTER_ARRIVE).item())
        replay_down_terminal = bool(torch.all(
            down_ready == DOWN_RESIDUAL_SPLIT - 1).item())
        print(
            f"[cooperative-replay-counters] hidden="
            f"{hidden_ready.min().item()}..{hidden_ready.max().item()} "
            f"down={down_ready.min().item()}..{down_ready.max().item()} "
            f"exact={replay_hidden_terminal and replay_down_terminal}",
            flush=True,
        )
        if not replay_hidden_terminal or not replay_down_terminal:
            raise SystemExit("cooperative replay counter terminal mismatch")

        old_180_open = _time_graph_replays(
            graph_standalone_180, replays=180, reps=args.reps)
        cooperative_180 = _time_graph_replays(
            graph_cooperative_180, replays=180, reps=args.reps)
        old_180_close = _time_graph_replays(
            graph_standalone_180, replays=180, reps=args.reps)
        old_180_total_us = (old_180_open[0] + old_180_close[0]) / 2.0
        old_180_per_call_us = old_180_total_us / 180.0
        old_180_residual_us = abs(
            old_180_total_us - old_180_per_call_us * 180.0)
        total_180_gain_us = old_180_total_us - cooperative_180[0]
        per_call_180_gain_us = (
            old_180_per_call_us - cooperative_180[1])
        print(
            f"[production-180-replay] old_open_total={old_180_open[0]:.6f} us "
            f"old_close_total={old_180_close[0]:.6f} us "
            f"old_midpoint_total={old_180_total_us:.6f} us "
            f"cooperative_total={cooperative_180[0]:.6f} us "
            f"total_gain={total_180_gain_us:+.6f} us "
            f"old_per_call={old_180_per_call_us:.9f} us "
            f"cooperative_per_call={cooperative_180[1]:.9f} us "
            f"per_call_gain={per_call_180_gain_us:+.9f} us "
            f"old_abs_total_minus_per_call_x180={old_180_residual_us:.12f} us "
            f"cooperative_abs_total_minus_per_call_x180="
            f"{cooperative_180[2]:.12f} us",
            flush=True,
        )

        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]) as trace:
            graph_cooperative_180.replay()
            torch.cuda.synchronize()
        kernel_events = [
            event for event in trace.key_averages()
            if (getattr(event, "device_time_total", 0) > 0
                or getattr(event, "self_device_time_total", 0) > 0)
        ]
        kernel_counts = {
            event.key: int(getattr(event, "count", 1))
            for event in kernel_events
        }
        fused_nodes = {
            name: count for name, count in kernel_counts.items()
            if "tl_out_proj_residual_rms_xfs" in name
        }
        split_nodes = {
            name: count for name, count in kernel_counts.items()
            if ("tl_out_proj_residual_partials" in name
                or "tl_rms_xfs_from_partials" in name)
        }
        reset_nodes = {
            name: count for name, count in kernel_counts.items()
            if "reset_ffn_counters_kernel" in name
        }
        consumer_nodes = {
            name: count for name, count in kernel_counts.items()
            if "ffn_taskloop_kernel" in name
        }
        fused_launches = sum(fused_nodes.values())
        consumer_launches = sum(consumer_nodes.values())
        print(
            f"[production-graph-kernels] names={sorted(kernel_counts)}",
            flush=True,
        )
        print(
            f"[production-graph-shape] fused_nodes={fused_nodes} "
            f"split_nodes={split_nodes} reset_nodes={reset_nodes} "
            f"consumer_nodes={consumer_nodes} consumer_grid_ctas={N_CTAS_FULL}",
            flush=True,
        )
        if fused_launches != 1:
            raise SystemExit(
                f"production graph expected one fused producer, got "
                f"{fused_launches}")
        if split_nodes:
            raise SystemExit("production graph still contains split producers")
        if reset_nodes:
            raise SystemExit("production graph still contains standalone reset")
        if N_CTAS_FULL != 132 or consumer_launches != 1:
            raise SystemExit(
                f"production graph consumer mismatch: grid={N_CTAS_FULL}, "
                f"launches={consumer_launches}")

        standalone_open_us = _time_graph(
            graph_standalone_reset, len(cases), args.reps)
        cooperative_cumulative_forward_us = _time_graph(
            graph_cooperative_full, len(cases), args.reps)
        standalone_close_us = _time_graph(
            graph_standalone_reset, len(cases), args.reps)
        cooperative_cumulative_open_us = _time_graph(
            graph_cooperative_full, len(cases), args.reps)
        standalone_reverse_us = _time_graph(
            graph_standalone_reset, len(cases), args.reps)
        cooperative_cumulative_close_us = _time_graph(
            graph_cooperative_full, len(cases), args.reps)
        standalone_midpoint_us = (
            standalone_open_us + standalone_close_us) / 2.0
        cooperative_cumulative_midpoint_us = (
            cooperative_cumulative_open_us
            + cooperative_cumulative_close_us) / 2.0
        cumulative_forward_gain_us = (
            standalone_midpoint_us - cooperative_cumulative_forward_us)
        cumulative_reverse_gain_us = (
            standalone_reverse_us - cooperative_cumulative_midpoint_us)
        minimum_cumulative_gain_us = min(
            cumulative_forward_gain_us, cumulative_reverse_gain_us)

        split_full_open_us = _time_graph(
            graph_split_full, len(cases), args.reps)
        cooperative_full_forward_us = _time_graph(
            graph_cooperative_full, len(cases), args.reps)
        split_full_close_us = _time_graph(
            graph_split_full, len(cases), args.reps)
        cooperative_full_open_us = _time_graph(
            graph_cooperative_full, len(cases), args.reps)
        split_full_reverse_us = _time_graph(
            graph_split_full, len(cases), args.reps)
        cooperative_full_close_us = _time_graph(
            graph_cooperative_full, len(cases), args.reps)
        split_full_midpoint_us = (
            split_full_open_us + split_full_close_us) / 2.0
        cooperative_full_midpoint_us = (
            cooperative_full_open_us + cooperative_full_close_us) / 2.0
        forward_gain_us = (
            split_full_midpoint_us - cooperative_full_forward_us)
        reverse_gain_us = (
            split_full_reverse_us - cooperative_full_midpoint_us)

        split_producer_open_us = _time_graph(
            graph_split_producer, len(cases), args.reps)
        cooperative_producer_forward_us = _time_graph(
            graph_cooperative_producer, len(cases), args.reps)
        split_producer_close_us = _time_graph(
            graph_split_producer, len(cases), args.reps)
        cooperative_producer_open_us = _time_graph(
            graph_cooperative_producer, len(cases), args.reps)
        split_producer_reverse_us = _time_graph(
            graph_split_producer, len(cases), args.reps)
        cooperative_producer_close_us = _time_graph(
            graph_cooperative_producer, len(cases), args.reps)
        split_producer_midpoint_us = (
            split_producer_open_us + split_producer_close_us) / 2.0
        cooperative_producer_midpoint_us = (
            cooperative_producer_open_us
            + cooperative_producer_close_us) / 2.0
        forward_regression_us = (
            cooperative_producer_forward_us - split_producer_midpoint_us)
        reverse_regression_us = (
            cooperative_producer_midpoint_us - split_producer_reverse_us)
        minimum_full_gain_us = min(forward_gain_us, reverse_gain_us)
        maximum_producer_regression_us = max(
            forward_regression_us, reverse_regression_us)
        print(
            f"[cooperative-cumulative-forward] standalone_open="
            f"{standalone_open_us:.3f} us "
            f"cooperative={cooperative_cumulative_forward_us:.3f} us "
            f"standalone_close={standalone_close_us:.3f} us "
            f"gain={cumulative_forward_gain_us:+.3f} us",
            flush=True,
        )
        print(
            f"[cooperative-cumulative-reverse] cooperative_open="
            f"{cooperative_cumulative_open_us:.3f} us "
            f"standalone={standalone_reverse_us:.3f} us "
            f"cooperative_close={cooperative_cumulative_close_us:.3f} us "
            f"gain={cumulative_reverse_gain_us:+.3f} us",
            flush=True,
        )
        print(
            f"[cooperative-full-forward] split_open={split_full_open_us:.3f} us "
            f"cooperative={cooperative_full_forward_us:.3f} us "
            f"split_close={split_full_close_us:.3f} us "
            f"gain={forward_gain_us:+.3f} us",
            flush=True,
        )
        print(
            f"[cooperative-full-reverse] cooperative_open="
            f"{cooperative_full_open_us:.3f} us "
            f"split={split_full_reverse_us:.3f} us "
            f"cooperative_close={cooperative_full_close_us:.3f} us "
            f"gain={reverse_gain_us:+.3f} us",
            flush=True,
        )
        print(
            f"[cooperative-producer-forward] split_open="
            f"{split_producer_open_us:.3f} us "
            f"cooperative={cooperative_producer_forward_us:.3f} us "
            f"split_close={split_producer_close_us:.3f} us "
            f"regression={forward_regression_us:+.3f} us",
            flush=True,
        )
        print(
            f"[cooperative-producer-reverse] cooperative_open="
            f"{cooperative_producer_open_us:.3f} us "
            f"split={split_producer_reverse_us:.3f} us "
            f"cooperative_close={cooperative_producer_close_us:.3f} us "
            f"regression={reverse_regression_us:+.3f} us",
            flush=True,
        )
        accepted = (
            minimum_cumulative_gain_us >= 1.0
            and minimum_full_gain_us >= 0.20
            and maximum_producer_regression_us <= 0.05)
        print(
            f"[cooperative-decision] accepted={accepted} "
            f"minimum_cumulative_gain={minimum_cumulative_gain_us:+.3f} us "
            f"minimum_full_gain={minimum_full_gain_us:+.3f} us "
            f"maximum_producer_regression="
            f"{maximum_producer_regression_us:+.3f} us",
            flush=True,
        )
        return

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
