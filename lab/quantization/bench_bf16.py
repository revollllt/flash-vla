"""BF16 cuBLAS (torch.matmul) at the same shapes and conditions, the baseline a
quantized route has to beat."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import cycled, graph_time_us, weight_copies
from vla_shapes import SHAPES, floor_us, weight_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = []
    for site in SHAPES:
        x = torch.randn(site.m, site.k, device="cuda", dtype=torch.bfloat16)
        copies = weight_copies(weight_bytes("bf16", site.k, site.n))
        weights = [torch.randn(site.n, site.k, device="cuda", dtype=torch.bfloat16) * 0.05
                   for _ in range(copies)]
        out = torch.empty(site.m, site.n, device="cuda", dtype=torch.bfloat16)
        gemm_us = graph_time_us(cycled(
            [lambda weight=weight: torch.matmul(x, weight.t(), out=out) for weight in weights],
            copies))
        floor, bound = floor_us("bf16", site.m, site.k, site.n)
        row = dict(lib="torch-cublas", version=torch.__version__, op="gemm", fmt="bf16",
                   backend="cublas", shape=site.name, m=site.m, k=site.k, n=site.n,
                   count=site.count, floor_us=round(floor, 3), floor_bound=bound,
                   us=round(gemm_us, 3), floor_ratio=round(gemm_us / floor, 3),
                   weight_copies=copies)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), rows=rows), indent=1))


if __name__ == "__main__":
    main()
