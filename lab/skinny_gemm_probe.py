"""One eager launch of a single skinny-GEMM configuration, for ncu capture.

The benchmark harness times inside a CUDA graph, which ncu cannot select a
launch out of; this runs the same call site eagerly over rotating weights so
`--launch-skip` lands on a warm launch that still reads a cold matrix.

    python lab/skinny_gemm_probe.py --shape qkv --tile-n 64 --depth 8 --split 1
    python lab/skinny_gemm_probe.py --shape qkv --backend cublas
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_MODULE = (_REPO / "src" / "flash_vla" / "hardware" / "nvidia" / "h100"
           / "lingbot_vla" / "backends" / "cuda" / "skinny_gemm.py")

SHAPES = {
    "qkv": (51, 768, 2560, True),
    "o_proj": (51, 2048, 768, False),
    "gate_up": (51, 768, 5504, False),
    "down_proj": (51, 2752, 768, False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="qkv")
    parser.add_argument("--backend", default="kernel", choices=("kernel", "cublas"))
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--launches", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda")
    rows, k, n, has_bias = SHAPES[args.shape]
    torch.manual_seed(0)
    x = torch.randn(rows, k, device=device, dtype=torch.bfloat16) * 0.5
    weights = [torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
               for _ in range(args.launches)]
    bias = (torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1
            if has_bias else None)
    out = torch.empty(rows, n, device=device, dtype=torch.bfloat16)

    if args.backend == "cublas":
        for w in weights:
            if bias is None:
                torch.mm(x, w.t(), out=out)
            else:
                torch.addmm(bias, x, w.t(), out=out)
        torch.cuda.synchronize()
        return

    spec = importlib.util.spec_from_file_location("skinny_gemm", _MODULE)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    kernel.build()
    workspace = counters = None
    if args.split > 1:
        workspace, counters = kernel.make_workspace(rows, n, args.tile_n, device)
    for w in weights:
        kernel.linear(x, w, out, bias, tile_n=args.tile_n, depth=args.depth,
                      k_split=args.split, workspace=workspace, counters=counters)
    torch.cuda.synchronize()
    reference = F.linear(x, weights[-1], bias)
    print(f"cosine={F.cosine_similarity(out.float().flatten(), reference.float().flatten(), dim=0).item():.6f}")


if __name__ == "__main__":
    main()
