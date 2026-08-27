"""Real full FFN PDL chain and same-job critical-path decomposition."""
from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from flash_vla.hardware.nvidia.h100.pi05.backends.cuda.taskloop import (
    COUNTER_ARRIVE, DOWN_RESIDUAL_SPLIT, FFNTaskloop, N_COUNTERS, build_table,
)
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers
from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang.kernels import adarms


M, M_PAD, D, FF = 50, 64, 1024, 4096


def randn(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
            * 0.05).bfloat16()


def pack_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    def tiles(weight: torch.Tensor) -> torch.Tensor:
        return (weight.reshape(D, FF // 32, 32).permute(1, 0, 2)
                .contiguous().view(-1, 32))
    return torch.cat((tiles(gate), tiles(up)), dim=1).contiguous()


def pack_down(weight: torch.Tensor) -> torch.Tensor:
    return (weight.reshape(FF, D // 32, 32).permute(1, 0, 2)
            .contiguous().view(FF, D))


def make_case(gen: torch.Generator) -> dict[str, torch.Tensor]:
    gate_w = randn(gen, D, FF)
    up_w = randn(gen, D, FF)
    down_w = randn(gen, FF, D)
    x = randn(gen, M, D)
    x_pad = torch.zeros((M_PAD, D), dtype=torch.bfloat16, device="cuda")
    x_pad[:M].copy_(x)
    return {
        "x": x,
        "x_pad_seed": x_pad,
        "scale": (1.0 + randn(gen, D)).bfloat16(),
        "gate_w": gate_w,
        "up_w": up_w,
        "packed_gate_up": pack_gate_up(gate_w, up_w),
        "gate_b": randn(gen, FF),
        "up_b": randn(gen, FF),
        "down_w": down_w,
        "packed_down": pack_down(down_w),
        "down_gate": randn(gen, D),
        "factor": torch.empty((M,), dtype=torch.bfloat16, device="cuda"),
        "xfs": torch.empty((D, M_PAD), dtype=torch.bfloat16, device="cuda"),
        "hidden_full": torch.empty((M_PAD, FF), dtype=torch.bfloat16, device="cuda"),
        "hidden_gu": torch.empty((M_PAD, FF), dtype=torch.bfloat16, device="cuda"),
        "hidden_base": torch.empty((M, FF), dtype=torch.bfloat16, device="cuda"),
        "hidden_dr": torch.zeros((M_PAD, FF), dtype=torch.bfloat16, device="cuda"),
        "out_full": x_pad.clone(),
        "out_base": x.clone(),
        "out_dr": x_pad.clone(),
        "out_dummy": torch.empty((M_PAD, D), dtype=torch.bfloat16, device="cuda"),
        "counters_full": torch.empty((N_COUNTERS,), dtype=torch.int32, device="cuda"),
        "counters_gu": torch.empty((N_COUNTERS,), dtype=torch.int32, device="cuda"),
        "counters_dr": torch.empty((N_COUNTERS,), dtype=torch.int32, device="cuda"),
    }


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    ref = reference.float().flatten()
    act = actual.float().flatten()
    return (F.cosine_similarity(ref, act, dim=0).item(),
            (ref - act).abs().max().item())


def capture(cases, body) -> torch.cuda.CUDAGraph:
    for case in cases:
        body(case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            body(case)
    return graph


def time_graph(core: torch.cuda.CUDAGraph, launches: int, reps: int,
               prepare: torch.cuda.CUDAGraph | None = None) -> float:
    for _ in range(5):
        if prepare is not None:
            prepare.replay()
        core.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        if prepare is not None:
            prepare.replay()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        core.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / launches)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", type=int, default=3)
    parser.add_argument("--reps", type=int, default=40)
    args = parser.parse_args()

    gen = torch.Generator(device="cuda").manual_seed(79)
    cases = [make_case(gen) for _ in range(args.sets)]
    tables = {mode: build_table(mode).to("cuda") for mode in ("full", "gu", "dr")}
    taskloop = FFNTaskloop(verbose=True)
    tilelang_gate = wrappers._compiled(
        adarms.tl_ada_scaled_gate, M=M, N=FF, K=D, **wrappers._DEC_GATE)

    def persistent(
            case, mode: str, use_pdl: bool,
            reset_in_producer: bool = True) -> None:
        hidden = case["hidden_full"] if mode == "full" else case["hidden_gu"]
        output = case["out_full"] if mode == "full" else case["out_dummy"]
        counters = case["counters_full"] if mode == "full" else case["counters_gu"]
        hidden_ready, down_ready = taskloop.readiness_counter_buffers(counters)
        if not reset_in_producer:
            taskloop.reset_counters(counters)
        wrappers.decoder_rms_xfs(
            case["x"], case["scale"], hidden_ready, down_ready, case["xfs"],
            trigger_programmatic_launch=use_pdl,
            reset_readiness=reset_in_producer)
        taskloop.launch(
            tables[mode], case["xfs"], case["factor"], case["scale"],
            case["packed_gate_up"], case["packed_gate_up"],
            case["gate_b"], case["up_b"], case["packed_down"],
            case["down_gate"], hidden, output, counters,
            zero_counters=False, use_programmatic_dependency=use_pdl,
        )

    def baseline(case) -> None:
        wrappers._rms_factor(case["x"], case["factor"])
        tilelang_gate(
            case["x"], case["factor"], case["scale"],
            case["gate_w"], case["up_w"], case["gate_b"], case["up_b"],
            case["hidden_base"],
        )
        wrappers.decoder_ffn_down_residual(
            case["hidden_base"], case["down_w"], case["down_gate"],
            case["out_base"],
        )

    def prepare_full(case) -> None:
        case["out_full"].copy_(case["x_pad_seed"])

    def prepare_base(case) -> None:
        case["out_base"].copy_(case["x"])

    def prepare_dr(case) -> None:
        case["out_dr"].copy_(case["x_pad_seed"])
        taskloop.reset_counters(case["counters_dr"])
        case["counters_dr"].fill_(COUNTER_ARRIVE)

    def dr_only(case) -> None:
        taskloop.launch(
            tables["dr"], case["xfs"], case["factor"], case["scale"],
            case["packed_gate_up"], case["packed_gate_up"],
            case["gate_b"], case["up_b"], case["packed_down"],
            case["down_gate"], case["hidden_dr"], case["out_dr"],
            case["counters_dr"], zero_counters=False,
        )

    # Semantic run.
    for case in cases:
        prepare_full(case)
        persistent(case, "full", True)
        prepare_base(case)
        baseline(case)
    torch.cuda.synchronize()
    for index, case in enumerate(cases):
        hidden_cos, hidden_max = metrics(case["hidden_base"], case["hidden_full"][:M])
        output_cos, output_max = metrics(case["out_base"], case["out_full"][:M])
        print(
            f"[full-parity] set={index} hidden_cos={hidden_cos:.7f} "
            f"hidden_max={hidden_max:.6g} output_cos={output_cos:.7f} "
            f"output_max={output_max:.6g}", flush=True,
        )
        if hidden_cos < 0.999 or output_cos < 0.999:
            raise SystemExit(f"full parity failure set={index}")
        case["hidden_dr"].zero_()
        case["hidden_dr"][:M].copy_(case["hidden_full"][:M])

    # DR-only semantic run against the persistent-full result, using the exact
    # same hidden tensor produced by GU.
    for case in cases:
        prepare_dr(case)
        dr_only(case)
    torch.cuda.synchronize()
    for index, case in enumerate(cases):
        dr_cos, dr_max = metrics(case["out_full"][:M], case["out_dr"][:M])
        print(f"[dr-parity] set={index} cos={dr_cos:.7f} max={dr_max:.6g}", flush=True)
        if dr_cos < 0.9999:
            raise SystemExit(f"DR parity failure set={index}")

    # Core graphs. Residual restores and DR prerequisite preparation are replayed
    # immediately before timing but sit before the begin event.
    full_prepare_graph = capture(cases, prepare_full)
    base_prepare_graph = capture(cases, prepare_base)
    dr_prepare_graph = capture(cases, prepare_dr)
    full_graph = capture(cases, lambda case: persistent(case, "full", True))
    reset_kernel_graph = capture(
        cases, lambda case: persistent(
            case, "full", True, reset_in_producer=False))
    baseline_graph = capture(cases, baseline)
    gu_graph = capture(cases, lambda case: persistent(case, "gu", True))
    dr_graph = capture(cases, dr_only)

    # Poison both readiness arrays, then prove repeated producer -> consumer
    # graph replay remains numerically identical rather than merely deadlock-free.
    for case in cases:
        hidden_ready, down_ready = taskloop.readiness_counter_buffers(
            case["counters_full"])
        hidden_ready.fill_(17)
        down_ready.fill_(19)
    for _ in range(20):
        full_prepare_graph.replay()
        full_graph.replay()
    base_prepare_graph.replay()
    baseline_graph.replay()
    torch.cuda.synchronize()
    for index, case in enumerate(cases):
        hidden_cos, hidden_max = metrics(
            case["hidden_base"], case["hidden_full"][:M])
        output_cos, output_max = metrics(case["out_base"], case["out_full"][:M])
        print(
            f"[replay-parity] set={index} hidden_cos={hidden_cos:.7f} "
            f"hidden_max={hidden_max:.6g} output_cos={output_cos:.7f} "
            f"output_max={output_max:.6g}", flush=True,
        )
        if hidden_cos < 0.999 or output_cos < 0.999:
            raise SystemExit(f"replay parity failure set={index}")
        hidden_ready, down_ready = taskloop.readiness_counter_buffers(
            case["counters_full"])
        hidden_terminal = bool(torch.all(hidden_ready == COUNTER_ARRIVE).item())
        down_terminal = bool(torch.all(
            down_ready == DOWN_RESIDUAL_SPLIT - 1).item())
        print(
            f"[replay-counters] set={index} "
            f"hidden={hidden_ready.min().item()}..{hidden_ready.max().item()} "
            f"down={down_ready.min().item()}..{down_ready.max().item()}",
            flush=True,
        )
        if not hidden_terminal or not down_terminal:
            raise SystemExit(f"replay counter terminal-state failure set={index}")

    # Profile the fused-pair test graph. The production decoder graph is checked
    # separately by the targeted end-to-end profiler.
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as trace:
        full_prepare_graph.replay()
        full_graph.replay()
        torch.cuda.synchronize()
    kernel_names = sorted({
        event.key for event in trace.key_averages()
        if (getattr(event, "device_time_total", 0) > 0
            or getattr(event, "self_device_time_total", 0) > 0)
    })
    reset_nodes = [name for name in kernel_names
                   if "reset_ffn_counters_kernel" in name]
    print(f"[graph-kernels] count={len(kernel_names)} reset_nodes={reset_nodes}",
          flush=True)
    if reset_nodes:
        raise SystemExit("production graph still contains standalone counter reset")

    measurements = {}
    for order in (("full", "reset_kernel", "base", "gu", "dr"),
                  ("dr", "gu", "base", "reset_kernel", "full")):
        for name in order:
            graph, prep = {
                "full": (full_graph, full_prepare_graph),
                "reset_kernel": (reset_kernel_graph, full_prepare_graph),
                "base": (baseline_graph, base_prepare_graph),
                "gu": (gu_graph, None),
                "dr": (dr_graph, dr_prepare_graph),
            }[name]
            measurements.setdefault(name, []).append(
                time_graph(graph, len(cases), args.reps, prep))
        print("[full-bench] " + " ".join(
            f"{name}={measurements[name][-1]:.3f}us"
            for name in ("full", "reset_kernel", "base", "gu", "dr")),
              flush=True)

    values = {name: statistics.median(samples)
              for name, samples in measurements.items()}
    print(
        f"[critical] full={values['full']:.3f}us base={values['base']:.3f}us "
        f"delta={values['full'] - values['base']:+.3f}us "
        f"reset_kernel={values['reset_kernel']:.3f}us "
        f"reset_gain={values['reset_kernel'] - values['full']:+.3f}us "
        f"gu_prefix={values['gu']:.3f}us dr_only={values['dr']:.3f}us "
        f"join_tail={values['full'] - values['gu']:.3f}us",
        flush=True,
    )


if __name__ == "__main__":
    main()
