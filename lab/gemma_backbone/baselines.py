"""Library baselines for the Gemma backbone's GEMM call sites, both Targets.

    python -m lab.gemma_backbone.baselines --csv out.csv

Times cuBLAS through torch at the exact production shapes, so the candidate
loop compares a hand-written kernel against the best library implementation of
the same maths rather than against the incumbent alone. Weights rotate cold
(`bench_gpu_time`'s rotating buffers), which is the regime that makes a weight
stream honest; the numbers here are therefore NOT comparable to the in-graph
attribution of `benchmarks profile`, where the activation is L2-resident
because its producer just wrote it. Both regimes appear in the contract's
baseline table and each is read for what it answers.

The gated FFN is timed in two library forms -- two separate GEMMs plus a fused
`gelu_tanh(gate) * up` pass, and the same with the multiply folded into one
`torch.compile` region -- because the fused dual-GEMM is the candidate those
forms have to beat.
"""
from __future__ import annotations

import argparse

import torch

from flash_vla.bench import KernelResult, bench_gpu_time, render_table, write_csv

#: Model constants of the Gemma backbone, shared by both Targets.
K_DIM, FFN, QKV_WIDTH = 2048, 16384, 2560
#: Prefix rows per Target: Pi0.5 carries a 200-token prompt, Pi0 none.
ROWS = {"pi05": 968, "pi0": 768}

GELU_C0 = 1.5957691216057308
GELU_C1 = 0.044715


def gelu_tanh(v: torch.Tensor) -> torch.Tensor:
    """The backbone's activation, spelled as the TileLang kernel spells it.

    `v * sigmoid(C0 * v * (1 + C1 * v^2))` is the sigmoid form of the tanh
    GELU approximation; it must be evaluated in fp32 on the accumulator and
    rounded once, so the caller passes an fp32 tensor and rounds the product.
    """
    return v * torch.sigmoid(GELU_C0 * v * (1.0 + GELU_C1 * v * v))


def _gemm_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def _bf16(*shape, device) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.bfloat16, device=device)


def _cases(target: str, device) -> list[tuple[str, callable, tuple, int, int]]:
    """(label, fn, args, flops, bytes) for one Target's production shapes."""
    m = ROWS[target]
    itemsize = 2
    cases = []

    # --- QKV projection: the GEMM inside llm_backbone_norm_qkv_rope ---------
    x = _bf16(m, K_DIM, device=device)
    qkv_w = _bf16(K_DIM, QKV_WIDTH, device=device)
    qkv_out = _bf16(m, QKV_WIDTH, device=device)
    cases.append((
        f"{target}/qkv cublas mm {m}x{K_DIM}x{QKV_WIDTH}",
        lambda a, b, o: torch.mm(a, b, out=o), (x, qkv_w, qkv_out),
        _gemm_flops(m, K_DIM, QKV_WIDTH),
        (m * K_DIM + K_DIM * QKV_WIDTH + m * QKV_WIDTH) * itemsize))

    # --- gated FFN: two library forms of llm_backbone_norm_gated_ffn -------
    gate_w = _bf16(K_DIM, FFN, device=device)
    up_w = _bf16(K_DIM, FFN, device=device)
    gate_out = _bf16(m, FFN, device=device)
    up_out = _bf16(m, FFN, device=device)
    hidden = _bf16(m, FFN, device=device)
    gu_flops = 2 * _gemm_flops(m, K_DIM, FFN)
    # The op's minimal traffic: A once, both weights once, the result once.
    # The two branch buffers this form materializes are the cost being priced.
    gu_bytes = (m * K_DIM + 2 * K_DIM * FFN + m * FFN) * itemsize

    def gu_two_gemms_fused_mul(a, w1, w2, g, u, h):
        torch.mm(a, w1, out=g)
        torch.mm(a, w2, out=u)
        h.copy_(gelu_tanh(g.float()) * u.float())
        return h

    cases.append((
        f"{target}/gated_ffn cublas 2xmm + gelu*up {m}x{K_DIM}x{FFN}",
        gu_two_gemms_fused_mul, (x, gate_w, up_w, gate_out, up_out, hidden),
        gu_flops, gu_bytes))

    # `torch._addmm_activation` carries a tanh-GELU epilogue in cuBLASLt; it
    # applies to the gate branch alone, so the multiply is still a third pass.
    gate_b = torch.zeros(FFN, dtype=torch.bfloat16, device=device)

    def gu_lt_gelu(a, w1, w2, b, g, u, h):
        torch._addmm_activation(b, a, w1, use_gelu=True, out=g)
        torch.mm(a, w2, out=u)
        h.copy_(g.float() * u.float())
        return h

    cases.append((
        f"{target}/gated_ffn cublasLt gelu + mm + mul {m}x{K_DIM}x{FFN}",
        gu_lt_gelu, (x, gate_w, up_w, gate_b, gate_out, up_out, hidden),
        gu_flops, gu_bytes))

    # --- down projection with the in-place residual (the Pi0 swap) ---------
    down_w = _bf16(FFN, K_DIM, device=device)
    residual = _bf16(m, K_DIM, device=device)
    cases.append((
        f"{target}/ffn_down_residual cublas addmm_ {m}x{FFN}x{K_DIM}",
        lambda o, a, b: o.addmm_(a, b), (residual, hidden, down_w),
        _gemm_flops(m, FFN, K_DIM),
        (m * FFN + FFN * K_DIM + 2 * m * K_DIM) * itemsize))

    # --- output projection with the in-place residual (the Pi0 swap) -------
    attn = _bf16(m, K_DIM, device=device)
    out_w = _bf16(K_DIM, K_DIM, device=device)
    residual2 = _bf16(m, K_DIM, device=device)
    cases.append((
        f"{target}/out_proj_residual cublas addmm_ {m}x{K_DIM}x{K_DIM}",
        lambda o, a, b: o.addmm_(a, b), (residual2, attn, out_w),
        _gemm_flops(m, K_DIM, K_DIM),
        (m * K_DIM + K_DIM * K_DIM + 2 * m * K_DIM) * itemsize))
    return cases


def run(targets: tuple[str, ...], timer: str, device: str = "cuda") -> list[KernelResult]:
    dev = torch.device(device)
    results: list[KernelResult] = []
    for target in targets:
        for label, fn, args, flops, nbytes in _cases(target, dev):
            samples = bench_gpu_time(fn, input_args=args,
                                     enable_cupti=(timer == "cupti"))
            results.append(KernelResult(label=label, samples=samples,
                                        flops=flops, bytes=nbytes))
            print(results[-1].perf_line(), flush=True)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", action="append", default=None,
                        choices=sorted(ROWS), help="default: both")
    parser.add_argument("--timer", choices=["cupti", "events"], default="cupti")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run this on a GPU node")
    results = run(tuple(args.target or sorted(ROWS)), args.timer)
    print()
    print(render_table(results))
    if args.csv:
        write_csv(args.csv, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
