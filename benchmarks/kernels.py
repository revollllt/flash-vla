"""Per-kernel GPU time for the Pi0 decoder kernels: python -m benchmarks kernels.

Where `e2e` times the whole graph replay and `profile` attributes time inside
it, this command benchmarks one kernel launch at a time, the way FlashInfer
benchmarks its kernels:

    python -m benchmarks kernels --all
    python -m benchmarks kernels --case decoder_attention --timer cupti
    python -m benchmarks kernels --all --csv results.csv

Each case is a small closure that issues exactly one kernel launch against the
benchmark buffers, plus the shape metadata (FLOPs / bytes) needed to derive the
headline metrics. Buffers and weights come from `benchmarks.synthetic`, so the
numbers are comparable with the e2e benchmarks.

The wrapper selection follows the production binding: `op_table(fused)` resolves
each call site to its backend, exactly as `pipeline.py` does, so a case measures
the kernel the way the engine actually runs it (unfused by default; pass
`--fused` for the FlashDecoding paths).

The timing machinery itself is not here -- it is generic and lives in the
library at `flash_vla.bench`, so a kernel that is not part of flash-vla can be
measured with the same methodology. See the `benchmark-kernel` skill.

Timing backends (--timer):
  cupti      hardware-level GPU kernel time (needs cupti-python>=13; fallback: events)
  cudagraph  CUDA-graph-amortised timing, rotating-buffer cold L2
  events     CUDA events, L2-flush cold L2

Every case reports, per FlashInfer's format:
  median time <ms>; std <ms>; achieved tflops <TFLOPs/sec>; achieved tb_per_sec <TB/sec>
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

from flash_vla.bench import KernelResult, bench_gpu_time, render_table, write_csv
from flash_vla.hardware.nvidia.h100.pi0 import op_table

from .metrics import require_cuda
from .synthetic import decoder_buffers, decoder_weights, encoder_seq_len


# Decoder shapes the kernels are tuned at (num_views=3, chunk_size=50, prompt_len=0).
NUM_HEADS = 8
HEAD_DIM = 256
CHUNK = 50
ENC_LEN = encoder_seq_len(3, 0)          # 768
TOTAL = ENC_LEN + CHUNK + 1              # 819

Case = tuple[str, Callable[[], None], dict[str, Any]]


def _attn_flops() -> int:
    """Decoder attention FLOPs (scores + attn-V) at the tuned shape, causal=0."""
    q = CHUNK + 1
    return 2 * q * TOTAL * NUM_HEADS * HEAD_DIM * 2

def _attn_bytes() -> int:
    """Q+K+V+O bytes read/written by the decoder attention kernel."""
    itemsize = 2  # bf16
    q = CHUNK + 1
    qb = q * NUM_HEADS * HEAD_DIM * itemsize
    kv = TOTAL * NUM_HEADS * HEAD_DIM * itemsize
    return qb + kv + kv + qb


def _gemm_bytes(m: int, n: int, k: int) -> int:
    """Read x + weight, write out, for a GEMM with the given dims (bf16)."""
    return (m * k + k * n + m * n) * 2


def build_cases(fused: bool = True) -> list[Case]:
    """Build the built-in case list. Buffers are shared across all cases."""
    buffers = decoder_buffers()
    weights = decoder_weights()
    ops = op_table(fused)

    # --- decoder GEMMs -----------------------------------------------------
    def action_in_proj():
        ops.decoder_action_in_proj(
            buffers["diffusion_noise"], weights["decoder_action_fused_in_proj_w"],
            weights["decoder_action_fused_time_biases"][0], buffers["decoder_x_buf"],
        )

    def action_mlp():
        ops.decoder_action_mlp(
            buffers["decoder_x_buf"], weights["decoder_action_mlp_w"],
            weights["decoder_action_mlp_b"], buffers["decoder_x_buf"],
        )

    def out_proj_residual():
        ops.decoder_out_proj_residual(
            buffers["decoder_q_buf"].view(-1, 2048), weights["decoder_attn_o_w"][0],
            buffers["decoder_x"],
        )

    # --- decoder attention ------------------------------------------------
    def attention():
        # Same call shape as pipeline.py: Q and out are both decoder_q_buf.
        ops.decoder_attention(
            buffers["decoder_q_buf"],
            buffers["encoder_K"][0], buffers["encoder_V"][0],
            buffers["decoder_attn_buf"], buffers["decoder_q_buf"],
            ENC_LEN,
        )

    # --- decoder norm+gated FFN -------------------------------------------
    def norm_gated_ffn():
        ops.decoder_norm_gated_ffn(
            buffers["decoder_x"], weights["decoder_ffn_gate_w"][0],
            weights["decoder_ffn_up_w"][0], buffers["decoder_hidden"],
            buffers["decoder_norm_factor_buf"],
        )

    # --- decoder state proj (the single state token) ----------------------
    def state_proj():
        ops.decoder_state_proj(
            buffers["observation_state_normalized"],
            weights["decoder_state_in_proj_w"], weights["decoder_state_in_proj_b"],
            buffers["decoder_state_buf"],
        )

    cases: list[Case] = [
        (
            "decoder_attention",
            attention,
            {"flops": _attn_flops(), "bytes": _attn_bytes()},
        ),
        (
            "decoder_norm_gated_ffn",
            norm_gated_ffn,
            {"flops": 2 * (CHUNK + 1) * 1024 * 4096,
             "bytes": _gemm_bytes(CHUNK + 1, 4096, 1024)},
        ),
        (
            "decoder_action_mlp",
            action_mlp,
            {"flops": 2 * CHUNK * 1024 * 1024,
             "bytes": _gemm_bytes(CHUNK, 1024, 1024)},
        ),
        (
            "decoder_action_in_proj",
            action_in_proj,
            {"flops": 2 * CHUNK * 32 * 1024,
             "bytes": _gemm_bytes(CHUNK, 1024, 32)},
        ),
        (
            "decoder_out_proj_residual",
            out_proj_residual,
            {"flops": 2 * (CHUNK + 1) * 1024 * 2048,
             "bytes": _gemm_bytes(CHUNK + 1, 1024, 2048)},
        ),
        (
            "decoder_state_proj",
            state_proj,
            {"flops": 2 * 1 * 1024 * 32,
             "bytes": _gemm_bytes(1, 1024, 32)},
        ),
    ]
    return cases


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks kernels",
                                description=__doc__.split("\n")[0])
    p.add_argument("--case", action="append", default=[],
                   help="benchmark a specific case (repeatable); default: all")
    p.add_argument("--all", action="store_true", help="benchmark every built-in case")
    p.add_argument("--timer", choices=["cupti", "cudagraph", "events"], default="cupti",
                   help="timing backend (default: cupti, auto-fallback to events)")
    p.add_argument("--fused", action="store_true", help="use fused wrappers (FlashDecoding)")
    p.add_argument("--reps", type=int, default=None,
                   help="measurement iterations (default: adaptive from target time)")
    p.add_argument("--repeat-time-ms", type=int, default=100,
                   help="target measurement duration in ms when --reps is unset")
    p.add_argument("--dry-run-time-ms", type=int, default=25,
                   help="target warmup duration in ms when --reps is unset")
    p.add_argument("--csv", default=None, help="append results to a CSV file")
    a = p.parse_args(argv)

    require_cuda()

    cases = build_cases(fused=a.fused)
    if a.case:
        wanted = set(a.case)
        cases = [c for c in cases if c[0] in wanted]
        missing = wanted - {c[0] for c in cases}
        if missing:
            print(f"error: unknown case(s): {sorted(missing)}", file=sys.stderr)
            print(f"available: {[c[0] for c in build_cases(fused=a.fused)]}", file=sys.stderr)
            return 1

    if not cases:
        print("no cases selected; use --all or --case <name>", file=sys.stderr)
        return 1

    results = []
    for label, fn, meta in cases:
        samples = bench_gpu_time(
            fn,
            repeat_iters=a.reps,
            dry_run_time_ms=a.dry_run_time_ms,
            repeat_time_ms=a.repeat_time_ms,
            enable_cupti=(a.timer == "cupti"),
            use_cuda_graph=(a.timer == "cudagraph"),
        )
        r = KernelResult(label, samples, flops=meta.get("flops"), bytes=meta.get("bytes"))
        results.append(r)
        print(r.perf_line())

    print()
    print(render_table(results))

    if a.csv:
        write_csv(a.csv, results)
        print(f"appended to {a.csv}")

    return 0


__all__ = ["build_cases", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
