"""Does split-K close b12x's gap on VLA down projections?

b12x (FlashInfer 0.7.0's SM12x CuTe DSL block-scaled GEMM) supports
split_k_slices with atomic accumulation, but neither its default planner nor its
autotuner tactics use it. This forces S slices onto the default tile by
substituting the kernel class the runners import, zeroes the output before every
call (the atomics need it; the memset is inside the timed graph), and checks the
result. Run one S per process: the runners' compile cache key omits S.
"""
import argparse
import json
from pathlib import Path
import sys

import flashinfer
import flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x as b12x
from flashinfer import fp4_quantize, mm_fp4, mm_mxfp8, mxfp8_quantize
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import cosine, cycled, graph_time_us, weight_copies
from vla_shapes import SHAPES, floor_us, weight_bytes

TARGETS = ("groot/dit_down", "groot/llm_down", "groot/ref_down", "groot/vit_down",
           "groot/dit_o", "pi05/exp_down", "pi05/exp_o", "pi05/vit_down", "pi05/llm_down")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", choices=["mxfp8", "nvfp4"], required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = args.split

    class SplitKKernel(b12x.Sm120B12xBlockScaledDenseGemmKernel):
        def __init__(self, *kernel_args: object, **kernel_kwargs: object) -> None:
            split_kwargs = ({"split_k_slices": split, "split_k_atomic_bf16": True}
                            if split > 1 else {})
            super().__init__(*kernel_args, **{**kernel_kwargs, **split_kwargs})

    # The experiment itself: the runners construct the kernel through this name.
    b12x.Sm120B12xBlockScaledDenseGemmKernel = SplitKKernel
    torch.manual_seed(0)
    rows = []
    for site in [site for site in SHAPES if site.name in TARGETS]:
        floor, bound = floor_us(args.fmt, site.m, site.k, site.n)
        row = dict(lib="flashinfer", version=flashinfer.__version__, op="gemm", fmt=args.fmt,
                   backend="b12x", split_k=split, shape=site.name, m=site.m, k=site.k, n=site.n,
                   count=site.count, floor_us=round(floor, 3), floor_bound=bound)
        rows.append(row)
        if (site.k // 128) % split:
            row["skipped"] = f"{site.k // 128} K tiles not divisible by {split}"
            print(json.dumps(row), flush=True)
            continue
        x = torch.randn(site.m, site.k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(site.n, site.k, device="cuda", dtype=torch.bfloat16) * 0.05
        out = torch.empty(site.m, site.n, device="cuda", dtype=torch.bfloat16)
        copies = weight_copies(weight_bytes(args.fmt, site.k, site.n))
        if args.fmt == "mxfp8":
            x_values, x_scales = mxfp8_quantize(x, True)
            quantized_weights = [mxfp8_quantize(weight, True) for _ in range(copies)]
            gemm = lambda values, scales: mm_mxfp8(x_values, values.t(), x_scales, scales, out=out,
                                                   out_dtype=torch.bfloat16, backend="b12x")
        else:
            x_encode_scale = (448 * 6) / x.float().abs().max()
            weight_encode_scale = (448 * 6) / weight.float().abs().max()
            x_values, x_scales = fp4_quantize(x, x_encode_scale, 16, False, True)
            quantized_weights = [fp4_quantize(weight, weight_encode_scale, 16, False, True)
                                 for _ in range(copies)]
            alpha = (1.0 / (x_encode_scale * weight_encode_scale)).float().reshape(1)
            gemm = lambda values, scales: mm_fp4(x_values, values.t(), x_scales, scales.t(), alpha,
                                                 torch.bfloat16, out, backend="b12x")
        # Split-K accumulates atomically into `out`, so each call starts from zero;
        # S = 1 writes `out` and is timed without the memset.
        calls = [lambda values=values, scales=scales: (out.zero_() if split > 1 else out,
                                                       gemm(values, scales))
                 for values, scales in quantized_weights]
        try:   # the kernel rejects slice counts it cannot tile
            calls[0]()
            torch.cuda.synchronize()
            similarity = cosine(out, x @ weight.t())
            gemm_us = graph_time_us(cycled(calls, copies))
            row.update(us=round(gemm_us, 3), cosine=round(similarity, 5),
                       floor_ratio=round(gemm_us / floor, 3),
                       ok=similarity > (0.99 if args.fmt == "mxfp8" else 0.97))
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
        print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), rows=rows), indent=1))


if __name__ == "__main__":
    main()
