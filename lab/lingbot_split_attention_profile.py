"""Plain (ungraphed) launches of the split-key attention kernels, for Nsight.

    ncu --kernel-name regex:split_ --launch-count 6 --set full \
        python -m lab.lingbot_split_attention_profile --key-tile 40

Runs a handful of launches of each phase at the Target's shape so a capture has
something to select; correctness is the benchmark script's job, not this one's.
"""
from __future__ import annotations

import argparse

import torch

from flash_vla.hardware.nvidia.h100.lingbot_vla.backends.cuda import split_attention
from flash_vla.models.lingbot.spec import (
    HEAD_DIM, KV_HEADS, PREFIX_LEN, QUERY_HEADS, SUFFIX_LEN,
)

CACHE_LEN = PREFIX_LEN + SUFFIX_LEN
SCALE = HEAD_DIM ** -0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-tile", type=int, default=split_attention.DEFAULT_KEY_TILE)
    parser.add_argument("--launches", type=int, default=6)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    query = torch.randn(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, device=device)
    key = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    value = torch.randn(KV_HEADS, CACHE_LEN, HEAD_DIM, device=device)
    mask = torch.rand(SUFFIX_LEN, CACHE_LEN, device=device) > 0.25
    target = torch.zeros(SUFFIX_LEN, QUERY_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device)
    buffers = split_attention.workspace(QUERY_HEADS, SUFFIX_LEN, HEAD_DIM, CACHE_LEN, device,
                                        args.key_tile)
    for _ in range(args.launches):
        split_attention.fused_attention(query, key, value, mask, target, SCALE,
                                        buffers=buffers, key_tile=args.key_tile)
    torch.cuda.synchronize()
    print("clock_sm_mhz=", torch.cuda.clock_rate() if hasattr(torch.cuda, "clock_rate") else "n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
