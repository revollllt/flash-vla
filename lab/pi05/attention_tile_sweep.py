"""A: the tiles of our Pi0.5 attention kernels on RTX 5090, swept at each workload's geometry.

- split-KV expert attention (`split_kv_attention`): query rows per tile,
  keys per block and warps of `attend_block`, rows and warps of
  `merge_blocks`, at each workload's chunk and the far end of its valid range.
- vision attention (`triton_vision_attention`): query and key tiles, warps and
  stages at each workload's camera views.

Each tile that fits the shared memory is timed in one CUDA graph over one
input per layer, and reported with its error against a float32 softmax and the
registers and spills of its kernels.

    python -m lab.pi05.attention_tile_sweep --out results/pi05-rtx5090/replay-extent/a/attention-tile-sweep.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics
from typing import Callable

import torch
import triton
from triton.runtime.errors import OutOfResources

from eval.metrics import error_metrics
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import split_kv_attention, triton_vision_attention
from flash_vla.models.pi05.spec import (
    DECODER_HEADS, ENCODER_LAYERS, HEAD_DIM, VISION_HEAD_DIM, VISION_HEADS, VISION_LAYERS,
    VISION_TOKENS)
from flash_vla.runtime.cuda import graph_samples

REPS = 30
TOP = 6
#: Workload -> (camera views, physical prefix rows, chunk, valid prefix rows).
CASES = {"robodojo": (3, 968, 50, 869), "libero": (2, 712, 10, 534)}


def timed(launch: Callable[[int], object], layers: int) -> float:
    """Median microseconds per call over graph replays of one call per layer."""
    return statistics.median(graph_samples(launch, n_inner=layers, reps=REPS)) * 1e3


def expert(physical: int, chunk: int, valid: int) -> list[dict[str, object]]:
    generator = torch.Generator(device="cuda").manual_seed(1)
    queries = chunk * DECODER_HEADS
    cache = physical + chunk
    q = torch.randn(queries, HEAD_DIM, generator=generator, device="cuda").to(torch.bfloat16)
    k = [torch.randn(cache, HEAD_DIM, generator=generator, device="cuda").to(torch.bfloat16)
         for _ in range(ENCODER_LAYERS)]
    v = [torch.randn(cache, HEAD_DIM, generator=generator, device="cuda").to(torch.bfloat16)
         for _ in range(ENCODER_LAYERS)]
    keys = torch.cat((torch.arange(valid), torch.arange(physical, cache))).cuda()
    expected = torch.softmax(q.float() @ k[0][keys].float().T * HEAD_DIM ** -0.5, -1) @ v[0][keys].float()
    out = torch.empty_like(q)
    valid_rows = torch.tensor([valid], dtype=torch.int32, device="cuda")
    rows = []
    for block_m, block_n, warps, merge_rows, merge_warps in itertools.product(
            (16, 32, 64), (32, 64), (4, 8), (1, 2, 4), (2, 4)):
        blocks = triton.cdiv(cache, block_n)
        partial = torch.empty(blocks, queries, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        stats = torch.empty(blocks, queries, 2, dtype=torch.float32, device="cuda")

        def launch(i: int) -> tuple[triton.compiler.CompiledKernel, triton.compiler.CompiledKernel]:
            attend = split_kv_attention.attend_block[(triton.cdiv(queries, block_m), blocks)](
                q, k[i % ENCODER_LAYERS], v[i % ENCODER_LAYERS], valid_rows, partial, stats,
                HEAD_DIM ** -0.5, queries, physical, chunk, block_m, block_n, HEAD_DIM,
                num_warps=warps, num_stages=1)
            merge = split_kv_attention.merge_blocks[(triton.cdiv(queries, merge_rows),)](
                partial, stats, valid_rows, out, queries, chunk, triton.next_power_of_2(blocks),
                block_n, merge_rows, HEAD_DIM, num_warps=merge_warps)
            return attend, merge
        try:
            attend, merge = launch(0)
        except OutOfResources:           # a tile past the shared memory: not a candidate
            continue
        error = error_metrics(expected, out.float())["rel_rms"]
        rows.append({"block_m": block_m, "block_n": block_n, "warps": warps,
                     "merge_rows": merge_rows, "merge_warps": merge_warps,
                     "us": timed(launch, ENCODER_LAYERS), "rel_rms": error,
                     "spills": [attend.n_spills, merge.n_spills],
                     "registers": [attend.n_regs, merge.n_regs]})
    return sorted(rows, key=lambda row: row["us"])


def vision(views: int) -> list[dict[str, object]]:
    generator = torch.Generator(device="cuda").manual_seed(2)
    width = VISION_HEADS * VISION_HEAD_DIM
    qkv = [torch.randn(views, VISION_TOKENS, 3 * width, generator=generator,
                       device="cuda").to(torch.bfloat16) for _ in range(VISION_LAYERS)]
    heads = qkv[0].float().view(views, VISION_TOKENS, 3, VISION_HEADS, VISION_HEAD_DIM).permute(0, 2, 3, 1, 4)
    logits = heads[:, 0] @ heads[:, 1].transpose(-1, -2) * VISION_HEAD_DIM ** -0.5
    expected = (torch.softmax(logits, -1) @ heads[:, 2]).transpose(1, 2).reshape(views, VISION_TOKENS, width)
    out = torch.empty(views, VISION_TOKENS, width, dtype=torch.bfloat16, device="cuda")
    rows = []
    for block_m, block_n, warps, stages in itertools.product((32, 64, 128), (32, 64, 128), (4, 8),
                                                             (1, 2, 3)):
        def launch(i: int) -> triton.compiler.CompiledKernel:
            return triton_vision_attention.vision_attention[
                (VISION_TOKENS // block_m, views * VISION_HEADS)](
                qkv[i % VISION_LAYERS], out, VISION_HEAD_DIM ** -0.5, VISION_TOKENS, VISION_HEADS,
                VISION_HEAD_DIM, block_m, block_n, num_warps=warps, num_stages=stages)
        try:
            kernel = launch(0)
        except OutOfResources:
            continue
        error = error_metrics(expected, out.float())["rel_rms"]
        rows.append({"block_m": block_m, "block_n": block_n, "warps": warps, "stages": stages,
                     "us": timed(launch, VISION_LAYERS), "rel_rms": error,
                     "spills": kernel.n_spills, "registers": kernel.n_regs})
    return sorted(rows, key=lambda row: row["us"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {workload: {"expert": expert(physical, chunk, valid), "vision": vision(views)}
              for workload, (views, physical, chunk, valid) in CASES.items()}
    for workload, kernels in report.items():
        for kernel, rows in kernels.items():
            print(workload, kernel, [{key: (round(value, 2) if isinstance(value, float) else value)
                                      for key, value in row.items()} for row in rows[:TOP]],
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reps": REPS, "cases": CASES, "workloads": report}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
