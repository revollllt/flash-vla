"""SGLang v0.5.20's KDA-generated NVFP4 SM120 GEMM at VLA shapes.

The kernel file is standalone (CuTe DSL and torch only), so it runs from the
vendored copy in the FlashInfer survey environment without installing SGLang.
Inputs are quantized with FlashInfer's fp4_quantize, whose 128x4 swizzled
scale layout is the one this kernel reads. Its tile is fixed at 16x64x512
for decode.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from flashinfer import fp4_quantize
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import cosine, cycled, graph_time_us, weight_copies
from vla_shapes import SHAPES, floor_us, weight_bytes

KDA_SOURCE = (Path(__file__).resolve().parents[2] / "third_party/quant-references/sglang/python/"
              "sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    module_spec = importlib.util.spec_from_file_location("kda_nvfp4", KDA_SOURCE)
    kda = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(kda)
    torch.manual_seed(0)
    rows = []
    for site in [site for site in SHAPES if args.only in site.name]:
        floor, bound = floor_us("nvfp4", site.m, site.k, site.n)
        row = dict(lib="sglang", version="0.5.20", op="gemm", fmt="nvfp4", backend="kda-cutedsl",
                   shape=site.name, m=site.m, k=site.k, n=site.n, count=site.count,
                   floor_us=round(floor, 3), floor_bound=bound)
        x = torch.randn(site.m, site.k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(site.n, site.k, device="cuda", dtype=torch.bfloat16) * 0.05
        x_encode_scale = (448 * 6) / x.float().abs().max()
        weight_encode_scale = (448 * 6) / weight.float().abs().max()
        x_values, x_scales = fp4_quantize(x, x_encode_scale, 16, False, True)
        copies = weight_copies(weight_bytes("nvfp4", site.k, site.n))
        quantized_weights = [fp4_quantize(weight, weight_encode_scale, 16, False, True)
                             for _ in range(copies)]
        alpha = (1.0 / (x_encode_scale * weight_encode_scale)).float().reshape(1)
        calls = [lambda values=values, scales=scales: kda._run_qwen3x_nvfp4_gemm(
                     x_values, values.T, x_scales, scales.T, alpha)
                 for values, scales in quantized_weights]
        try:   # the decode-specialised kernel rejects some shapes
            similarity = cosine(calls[0](), x @ weight.t())
            gemm_us = graph_time_us(cycled(calls, copies))
            row.update(us=round(gemm_us, 3), cosine=round(similarity, 5), weight_copies=copies,
                       floor_ratio=round(gemm_us / floor, 3), ok=similarity > 0.97)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                        rows=rows), indent=1))


if __name__ == "__main__":
    main()
