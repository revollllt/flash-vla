"""vLLM 0.30.0 NVFP4 and 1D2D-FP8 CUTLASS GEMMs and their quantizers at VLA shapes.

vLLM has no MXFP8 GEMM of its own: on SM100+ its MXFP8 path quantizes with
FlashInfer and its GEMM is FlashInfer's, which bench_flashinfer.py measures.
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Literal

import torch
import vllm
from vllm import _custom_ops as vllm_ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import cosine, cycled, graph_time_us, weight_copies
from vla_shapes import SHAPES, floor_us, weight_bytes


def time_gemm(fmt: Literal["nvfp4", "fp8_1d2d"], m: int, k: int, n: int
              ) -> tuple[float, float, int, float]:
    """(µs per GEMM, cosine to BF16, weight copies, µs per activation quantize)."""
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    expected = x @ weight.t()
    copies = weight_copies(weight_bytes(fmt, k, n))
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    if fmt == "nvfp4":
        x_encode_scale = ((448 * 6) / x.float().abs().max()).reshape(1)
        weight_encode_scale = ((448 * 6) / weight.float().abs().max()).reshape(1)
        x_values, x_scales = vllm_ops.scaled_fp4_quant(x, x_encode_scale)
        quantized_weights = [vllm_ops.scaled_fp4_quant(weight, weight_encode_scale)
                             for _ in range(copies)]
        alpha = (1.0 / (x_encode_scale * weight_encode_scale)).float()
        calls = [lambda values=values, scales=scales: torch.ops._C.cutlass_scaled_fp4_mm(
                     out, x_values, values, x_scales, scales, alpha)
                 for values, scales in quantized_weights]
        quantize = lambda: vllm_ops.scaled_fp4_quant(x, x_encode_scale)
    else:
        x_values, x_scales = per_token_group_quant_fp8(x, 128, column_major_scales=True)
        # One FP32 scale per [128, 128] weight block (DeepGEMM-style 1D2D weights).
        weight_blocks = weight.float().reshape(n // 128, 128, k // 128, 128)
        block_scales = weight_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0
        weight_values = ((weight_blocks / block_scales[:, None, :, None]).clamp(-448, 448)
                         .to(torch.float8_e4m3fn).reshape(n, k))
        quantized_weights = [(weight_values.clone(), block_scales.clone()) for _ in range(copies)]
        calls = [lambda values=values, scales=scales: torch.ops._C.cutlass_scaled_mm(
                     out, x_values, values.t(), x_scales, scales.t(), None)
                 for values, scales in quantized_weights]
        quantize = lambda: per_token_group_quant_fp8(x, 128, column_major_scales=True)
    calls[0]()
    torch.cuda.synchronize()
    return (graph_time_us(cycled(calls, copies)), cosine(out, expected), copies,
            graph_time_us([quantize] * 64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = []
    for site in [site for site in SHAPES if args.only in site.name]:
        for fmt in ("nvfp4", "fp8_1d2d"):
            floor, bound = floor_us(fmt, site.m, site.k, site.n)
            row = dict(lib="vllm", version=vllm.__version__, op="gemm", fmt=fmt, backend="cutlass",
                       shape=site.name, m=site.m, k=site.k, n=site.n, count=site.count,
                       floor_us=round(floor, 3), floor_bound=bound)
            try:   # the CUTLASS kernels reject shapes they have no tile for
                gemm_us, similarity, copies, quantize_us = time_gemm(fmt, site.m, site.k, site.n)
                row.update(us=round(gemm_us, 3), cosine=round(similarity, 5),
                           weight_copies=copies, floor_ratio=round(gemm_us / floor, 3),
                           ok=similarity > (0.97 if fmt == "nvfp4" else 0.99),
                           quantize_us=round(quantize_us, 3))
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                        rows=rows), indent=1))


if __name__ == "__main__":
    main()
