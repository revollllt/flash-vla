"""FlashInfer 0.7.0 MXFP8 / NVFP4 GEMMs and quantizers at VLA shapes.

Each backend is timed as a CUDA graph of calls that cycle through enough
weight copies to exceed 2x the 96 MiB L2 (weights cold, as in the model) with
one warm activation, and checked against a BF16 matmul before it is timed.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
from typing import Literal

import flashinfer
from flashinfer import fp4_quantize, mm_fp4, mm_mxfp8, mxfp8_quantize
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import cosine, cycled, graph_time_us, weight_copies
from vla_shapes import SHAPES, floor_us, weight_bytes

BACKENDS = ("cutlass", "b12x", "cudnn", "cute-dsl")


def time_gemm(fmt: Literal["mxfp8", "nvfp4"], m: int, k: int, n: int, backend: str,
              autotune: bool) -> tuple[float, float, int]:
    """(µs per GEMM, cosine to BF16, weight copies) for one backend."""
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    expected = x @ weight.t()
    copies = weight_copies(weight_bytes(fmt, k, n))
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    if fmt == "mxfp8":
        x_values, x_scales = mxfp8_quantize(x, True)
        quantized_weights = [mxfp8_quantize(weight, True) for _ in range(copies)]
        calls = [lambda values=values, scales=scales: mm_mxfp8(
                     x_values, values.t(), x_scales, scales, out=out, out_dtype=torch.bfloat16,
                     backend=backend)
                 for values, scales in quantized_weights]
    else:
        x_encode_scale = (448 * 6) / x.float().abs().max()
        weight_encode_scale = (448 * 6) / weight.float().abs().max()
        x_values, x_scales = fp4_quantize(x, x_encode_scale, 16, False, True)
        quantized_weights = [fp4_quantize(weight, weight_encode_scale, 16, False, True)
                             for _ in range(copies)]
        alpha = (1.0 / (x_encode_scale * weight_encode_scale)).to(torch.float32).reshape(1)
        calls = [lambda values=values, scales=scales: mm_fp4(
                     x_values, values.t(), x_scales, scales.t(), alpha, torch.bfloat16, out,
                     backend=backend)
                 for values, scales in quantized_weights]
    with flashinfer.autotune(True) if autotune else contextlib.nullcontext():
        calls[0]()
    torch.cuda.synchronize()
    return graph_time_us(cycled(calls, copies)), cosine(out, expected), copies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--autotune", action="store_true",
                        help="profile each backend's tactics on the first call")
    parser.add_argument("--skip-quant", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = []
    for site in [site for site in SHAPES if args.only in site.name]:
        for fmt in ("mxfp8", "nvfp4"):
            for backend in BACKENDS:
                floor, bound = floor_us(fmt, site.m, site.k, site.n)
                row = dict(lib="flashinfer", version=flashinfer.__version__, op="gemm", fmt=fmt,
                           backend=backend, autotuned=args.autotune, shape=site.name, m=site.m,
                           k=site.k, n=site.n, count=site.count, floor_us=round(floor, 3),
                           floor_bound=bound)
                try:   # backends reject shapes and architectures they do not support
                    gemm_us, similarity, copies = time_gemm(fmt, site.m, site.k, site.n,
                                                            backend, args.autotune)
                    row.update(us=round(gemm_us, 3), cosine=round(similarity, 5),
                               weight_copies=copies, floor_ratio=round(gemm_us / floor, 3),
                               ok=similarity > (0.99 if fmt == "mxfp8" else 0.97))
                except Exception as error:
                    row.update(error=f"{type(error).__name__}: {str(error).splitlines()[0][:200]}")
                rows.append(row)
                print(json.dumps(row), flush=True)
        for fmt in () if args.skip_quant else ("mxfp8", "nvfp4"):
            x = torch.randn(site.m, site.k, device="cuda", dtype=torch.bfloat16)
            encode_scale = (448 * 6) / x.float().abs().max()
            quantize = ((lambda: mxfp8_quantize(x, True)) if fmt == "mxfp8"
                        else (lambda: fp4_quantize(x, encode_scale, 16, False, True)))
            row = dict(lib="flashinfer", version=flashinfer.__version__, op="quantize", fmt=fmt,
                       backend="cuda", shape=site.name, m=site.m, k=site.k,
                       us=round(graph_time_us([quantize] * 64), 3))
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                        rows=rows), indent=1))


if __name__ == "__main__":
    main()
