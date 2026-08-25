"""Per-kernel roofline for the Pi0.5 decoder: MFU, MBU, and which bound is binding.

For every decoder call site this reports, at its real captured shape:

  intensity   FLOP per byte of HBM traffic -- which side of the H100 ridge point
              (989 TFLOP/s / 3.35 TB/s = 295 FLOP/byte) the kernel sits on
  MFU         achieved TFLOP/s over the 989 TFLOP/s bf16 tensor-core peak
  MBU         achieved TB/s over the 3.35 TB/s HBM3 peak
  sm_used     the fraction of the 132 SMs the grid occupies

The binding number is whichever roofline the kernel is under: MBU for a
memory-bound kernel, MFU for a compute-bound one. A low value on the *other*
axis is not a problem -- a memory-bound kernel is supposed to leave the tensor
cores idle.

Each kernel is timed in isolation, cold, cycling distinct weight buffers so the
reads miss L2 -- the same regime the sweeps and the graph use. Timing them
separately rather than reading the graph profile is deliberate: the profiler
aggregates by CUDA function name, so the two `tl_matmul_gated_res` shapes
(K=2048 and K=4096) collapse into one blended row, and a roofline needs them
apart. The isolated sum is cross-checked against the profiled decoder total.

The byte model is minimal HBM traffic: each input read once, each output
written once, weights being the cold term that dominates at M=50. It is a lower
bound on real traffic -- it ignores L2 spills and, for FlashDecoding, the Q
re-read per split, which is added explicitly. So a reported MBU near 1.0 means
the kernel is at the bandwidth wall; a low MBU on a memory-bound kernel means it
is leaving bandwidth on the table, usually to wave quantization.
"""
from __future__ import annotations

import argparse
import json

import torch

from flash_vla.hardware.nvidia.h100.spec import H100Spec
from flash_vla.models.pi05.spec import (
    DECODER_DIM,
    DECODER_FFN,
    DECODER_HEADS,
    HEAD_DIM,
    QKV_WIDTH,
)
from flash_vla.runtime.cuda import graph_time_cold

from .metrics import require_cuda

PEAK_FLOPS = H100Spec.TENSOR_CORE_DENSE_PEAK_FLOPS["bf16"]   # 989 TFLOP/s
PEAK_BW = H100Spec.HBM_BANDWIDTH_BYTES_PER_SECOND           # 3.35 TB/s
SM_COUNT = H100Spec.SM_COUNT                               # 132
RIDGE = PEAK_FLOPS / PEAK_BW                               # 295 FLOP/byte
CHUNK = 50
KEYS = 1018
BF16 = 2


def _rand(gen, *shape):
    return (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
            * 0.05).bfloat16()


def _cold_n(footprint_bytes: int) -> int:
    from flash_vla.tuning import cold_n_inner
    return min(cold_n_inner(footprint_bytes, H100Spec.L2_CACHE_SIZE_BYTES), 48)


def _sites(gen):
    """Yield (name, invoke(i), flop, bytes_, ctas) for every decoder call site."""
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers as w

    def gemm_ctas(cfg, n):
        return -(-n // cfg["BLOCK_N"]) * -(-CHUNK // cfg["BLOCK_M"])

    # --- qkv projection: M x K @ K x N, + rope ---
    n = _cold_n(DECODER_DIM * QKV_WIDTH * BF16)
    x = _rand(gen, CHUNK, DECODER_DIM)
    scale, bias, rope = _rand(gen, DECODER_DIM), _rand(gen, QKV_WIDTH), _rand(gen, CHUNK, HEAD_DIM)
    wq = [_rand(gen, DECODER_DIM, QKV_WIDTH) for _ in range(n)]
    Q = torch.empty((CHUNK * DECODER_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    K = torch.empty((CHUNK, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    V = torch.empty((CHUNK, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    nf = torch.empty((CHUNK,), dtype=torch.bfloat16, device="cuda")
    yield ("ada_qkv_gemm_rope",
           lambda i: w.decoder_norm_qkv_rope(x, scale, wq[i % n], bias, rope, Q, K, V, nf),
           2 * CHUNK * QKV_WIDTH * DECODER_DIM,
           (CHUNK * DECODER_DIM + DECODER_DIM * QKV_WIDTH + CHUNK * QKV_WIDTH) * BF16,
           gemm_ctas(w._DEC_QKV, QKV_WIDTH))

    # --- gated FFN: dual GEMM M x K @ K x N twice ---
    n = _cold_n(2 * DECODER_DIM * DECODER_FFN * BF16)
    gw = [_rand(gen, DECODER_DIM, DECODER_FFN) for _ in range(n)]
    uw = [_rand(gen, DECODER_DIM, DECODER_FFN) for _ in range(n)]
    gb, ub = _rand(gen, DECODER_FFN), _rand(gen, DECODER_FFN)
    out_ffn = torch.empty((CHUNK, DECODER_FFN), dtype=torch.bfloat16, device="cuda")
    yield ("ada_scaled_gate (dual GEMM)",
           lambda i: w.decoder_norm_gated_ffn(x, scale, gw[i % n], uw[i % n], gb, ub, out_ffn, nf),
           2 * (2 * CHUNK * DECODER_FFN * DECODER_DIM),
           (CHUNK * DECODER_DIM + 2 * DECODER_DIM * DECODER_FFN + CHUNK * DECODER_FFN) * BF16,
           gemm_ctas(w._DEC_GATE, DECODER_FFN))

    # --- gated residual, two K ---
    for label, k in (("o_proj", DECODER_HEADS * HEAD_DIM), ("ffn_down", DECODER_FFN)):
        n = _cold_n(k * DECODER_DIM * BF16)
        xin = _rand(gen, CHUNK, k)
        gate = _rand(gen, DECODER_DIM)
        res = _rand(gen, CHUNK, DECODER_DIM)
        wr = [_rand(gen, k, DECODER_DIM) for _ in range(n)]
        yield (f"matmul_gated_res ({label}, K={k})",
               lambda i, xin=xin, wr=wr, n=n, gate=gate, res=res:
                   w.decoder_out_proj_residual(xin, wr[i % n], gate, res),
               2 * CHUNK * DECODER_DIM * k,
               (CHUNK * k + k * DECODER_DIM + 2 * CHUNK * DECODER_DIM) * BF16,
               gemm_ctas(w._DEC_RESIDUAL, DECODER_DIM))

    # --- attention: FlashDecoding split + combine, Q re-read per split ---
    num_split, _ = w._num_splits(KEYS, w._FD_SPLIT["BLOCK_N"], w._FD_SPLIT["NUM_SPLIT"])
    m_flat = CHUNK * DECODER_HEADS
    n = _cold_n(2 * KEYS * HEAD_DIM * BF16)
    q = _rand(gen, m_flat, HEAD_DIM)
    kv = [(_rand(gen, KEYS, HEAD_DIM), _rand(gen, KEYS, HEAD_DIM)) for _ in range(n)]
    mask = torch.zeros((KEYS,), dtype=torch.bfloat16, device="cuda")
    mask[903:968] = -3.0e38
    o = torch.empty_like(q)
    yield ("fd_attention (split+combine)",
           lambda i: w.decoder_attention(q, kv[i % n][0], kv[i % n][1], mask, o),
           4 * m_flat * KEYS * HEAD_DIM,
           (num_split * m_flat * HEAD_DIM + 2 * KEYS * HEAD_DIM + m_flat * HEAD_DIM) * BF16,
           -(-m_flat // w._FD_SPLIT["BLOCK_M"]) * num_split)

    # --- action in/out projections, tiny ---
    n = _cold_n(DECODER_DIM * DECODER_DIM * BF16)
    ain = _rand(gen, CHUNK, 32)
    win = [_rand(gen, 32, DECODER_DIM) for _ in range(n)]
    bin_ = _rand(gen, DECODER_DIM)
    xd = _rand(gen, CHUNK, DECODER_DIM)
    wout = [_rand(gen, DECODER_DIM, 32) for _ in range(n)]
    bout = _rand(gen, 32)
    noise = torch.empty((CHUNK, 32), dtype=torch.bfloat16, device="cuda")
    yield ("action_in_proj",
           lambda i: w.decoder_action_in_proj(ain, win[i % n], bin_, xd[:, :DECODER_DIM]),
           2 * CHUNK * DECODER_DIM * 32,
           (CHUNK * 32 + 32 * DECODER_DIM + CHUNK * DECODER_DIM) * BF16,
           gemm_ctas(w._DEC_ACTION_IN, DECODER_DIM))
    yield ("action_out_proj",
           lambda i: w.decoder_action_out_proj(xd, wout[i % n], bout, noise, nf),
           2 * CHUNK * 32 * DECODER_DIM,
           (CHUNK * DECODER_DIM + DECODER_DIM * 32 + CHUNK * 32) * BF16,
           gemm_ctas(w._DEC_OUT_PROJ, 32))


def run(reps: int = 40, seed: int = 0) -> dict:
    """Time each decoder kernel cold and report its roofline row."""
    require_cuda()
    torch.cuda.init()
    from flash_vla.hardware.nvidia.h100.pi05.backends.tilelang import wrappers
    from flash_vla.runtime.cuda import ScratchPool

    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = []
    with wrappers.use_pool(ScratchPool()):
        for name, invoke, flop, bytes_, ctas in _sites(gen):
            for _ in range(4):
                invoke(0)
            torch.cuda.synchronize()
            us = graph_time_cold(invoke, n_inner=32, reps=reps)
            seconds = us * 1e-6
            achieved_flops = flop / seconds
            achieved_bw = bytes_ / seconds
            intensity = flop / bytes_
            rows.append({
                "kernel": name,
                "us": round(us, 3),
                "intensity": round(intensity, 1),
                "bound": "compute" if intensity > RIDGE else "memory",
                "MFU": round(achieved_flops / PEAK_FLOPS, 4),
                "MBU": round(achieved_bw / PEAK_BW, 4),
                "TFLOPs": round(achieved_flops / 1e12, 1),
                "TB_s": round(achieved_bw / 1e12, 2),
                "sm_used": round(min(ctas / SM_COUNT, 1.0), 3),
                "ctas": ctas,
            })

    report = {
        "device": torch.cuda.get_device_name(0),
        "peak": {"bf16_TFLOPs": PEAK_FLOPS / 1e12, "hbm_TB_s": PEAK_BW / 1e12,
                 "ridge_flop_per_byte": round(RIDGE, 1), "sms": SM_COUNT},
        "kernels": rows,
    }
    print(json.dumps(report, indent=2))
    print()
    print(f"{'kernel':32s} {'us':>7} {'intns':>7} {'bound':>7} "
          f"{'MFU':>6} {'MBU':>6} {'SM':>5}")
    for r in rows:
        binding = r["MBU"] if r["bound"] == "memory" else r["MFU"]
        print(f"{r['kernel']:32s} {r['us']:>7.2f} {r['intensity']:>7.1f} {r['bound']:>7} "
              f"{r['MFU']*100:>5.1f}% {r['MBU']*100:>5.1f}% {r['sm_used']:>5.2f}"
              f"   <- {binding*100:.0f}% of its wall")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run(args.reps, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
