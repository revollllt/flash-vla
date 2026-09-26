"""V0.4 and A: the Pi0.5 attentions on RTX 5090, the earlier kernels against FlashInfer and ours.

The backbone and expert attentions are multi-query (8 query heads, one KV head
of width 256) and bidirectional, masked on keys only:

- backbone: queries and keys are the prefix rows. The torch kernel
  (`models/pi05/attention.py`, compiled matmul-softmax-matmul) runs under the
  additive mask, at every physical row (before the replay buckets) or at the
  rows of the 64-row bucket the valid ones fall in (what the buckets replay);
  FlashInfer runs the valid rows only, planned per inference on a graph
  captured once (`BatchPrefillWithRaggedKVCacheWrapper` with `use_cuda_graph`,
  planned at the bucket's rows first, as the runner's preparation does), or as
  one call at that length (`single_prefill_with_kv_cache`).
- expert: the chunk's queries over the valid prefix keys and the chunk's own,
  which sit apart in the cache. The earlier kernels (`triton_qk_attention`)
  scan every cache row under the mask; FlashInfer reads the two ranges through
  a one-row page table (`BatchPrefillWithPagedKVCacheWrapper`), or one call over
  a contiguous copy of the keys it would read (`single_prefill`, a lower bound
  on what a gather-free contiguous layout would cost). `split_kv_attention`
  runs the valid keys, one block of them per CTA, and merges the blocks.
- vision: each view's 256 tokens over themselves, 16 heads of width 72, which
  FlashInfer does not build. The earlier kernel is the compiled
  `scaled_dot_product_attention`; `triton_vision_attention` reads each head as
  64 columns and a masked 16-column tail.

Each candidate is checked on the first layer's input against a float32 softmax
over the valid keys, then timed in one CUDA graph over one input per layer (18
backbone layers, 27 vision layers).

FlashInfer (`flashinfer-python`, whose JIT needs nvcc on PATH) is a lab
dependency only: no route runs it (its backbone gain over the bucket rows did
not survive end to end).

    python -m lab.pi05.attention_screen --out results/pi05-rtx5090/replay-extent/a/attention-screen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Callable

import flashinfer
import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import (
    split_kv_attention, triton_qk_attention, triton_vision_attention)
from flash_vla.models.pi05 import attention
from flash_vla.models.pi05.spec import (
    DECODER_HEADS, ENCODER_LAYERS, HEAD_DIM, MASK_NEG, VISION_HEAD_DIM, VISION_HEADS, VISION_LAYERS,
    VISION_TOKENS)
from flash_vla.runtime.cuda import graph_samples
from flash_vla.runtime.workspace import Scratch

REPS = 30
#: The replay bucket granularity (`hardware/nvidia/rtx5090/pi05/target.py`).
BUCKET_ROWS = 64
WORKSPACE_BYTES = 256 << 20
#: Workload -> (camera views, physical prefix rows, chunk, valid prefix rows at
#: the ends of its range).
CASES = {"robodojo": (3, 968, 50, (811, 869)), "libero": (2, 712, 10, (518, 534))}


def timed(launch: Callable[[int], object], layers: int = ENCODER_LAYERS) -> float:
    """Median microseconds per call over graph replays of one call per layer."""
    return statistics.median(graph_samples(launch, n_inner=layers, reps=REPS)) * 1e3


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Float32 softmax attention: q (rows, heads, 256) over k, v (keys, 256)."""
    logits = torch.einsum("rhd,kd->rhk", q.float(), k.float()) * HEAD_DIM ** -0.5
    return torch.einsum("rhk,kd->rhd", torch.softmax(logits, -1), v.float())


def within(expected: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    metrics = error_metrics(expected, observed.float())
    limit = tolerances()["shallow"]
    return {**metrics, "within": float(metrics["rel_rms"] <= limit["rel_rms_max"]
                                        and metrics["cosine_similarity"] >= limit["cosine_min"])}


def backbone(physical: int, valid: int, workspace: torch.Tensor) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(0)

    def tensor(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device="cuda").to(torch.bfloat16)

    q = [tensor(physical, DECODER_HEADS, HEAD_DIM) for _ in range(ENCODER_LAYERS)]
    k = [tensor(physical, HEAD_DIM) for _ in range(ENCODER_LAYERS)]
    v = [tensor(physical, HEAD_DIM) for _ in range(ENCODER_LAYERS)]
    out = torch.zeros(physical, DECODER_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    mask = torch.zeros(physical, dtype=torch.bfloat16, device="cuda")
    mask[valid:] = MASK_NEG
    expected = reference(q[0][:valid], k[0][:valid], v[0][:valid])
    rows: dict[str, dict[str, float]] = {}

    def physical_rows(i: int) -> None:
        attention.llm_backbone_attention(q[i % ENCODER_LAYERS].view(-1, HEAD_DIM), k[i % ENCODER_LAYERS],
                                         v[i % ENCODER_LAYERS], HEAD_DIM ** -0.5, mask,
                                         out.view(-1, HEAD_DIM))
    physical_rows(0)
    checked = within(expected, out[:valid])
    rows["torch (all physical rows)"] = {"us": timed(physical_rows), **checked}
    bucket = -(-valid // BUCKET_ROWS) * BUCKET_ROWS

    def bucketed(i: int) -> None:
        attention.llm_backbone_attention(q[i % ENCODER_LAYERS][:bucket].view(-1, HEAD_DIM),
                                         k[i % ENCODER_LAYERS][:bucket], v[i % ENCODER_LAYERS][:bucket],
                                         HEAD_DIM ** -0.5, mask[:bucket], out[:bucket].view(-1, HEAD_DIM))
    out.zero_()
    bucketed(0)
    checked = within(expected, out[:valid])
    rows["torch (bucket rows)"] = {"us": timed(bucketed), **checked}

    def single(i: int) -> torch.Tensor:
        return flashinfer.single_prefill_with_kv_cache(
            q[i % ENCODER_LAYERS][:valid], k[i % ENCODER_LAYERS][:valid, None],
            v[i % ENCODER_LAYERS][:valid, None], causal=False)
    rows["flashinfer single_prefill (valid rows)"] = {"us": timed(single), **within(expected, single(0))}

    indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
    ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", use_cuda_graph=True, qo_indptr_buf=indptr,
        kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device="cuda"), backend="fa2")
    for rows_planned in (bucket, valid):
        bounds = torch.tensor([0, rows_planned], dtype=torch.int32)
        ragged.plan(bounds, bounds, DECODER_HEADS, 1, HEAD_DIM, causal=False,
                    q_data_type=torch.bfloat16)

    def planned(i: int) -> None:
        ragged.run(q[i % ENCODER_LAYERS][:valid], k[i % ENCODER_LAYERS][:valid, None],
                   v[i % ENCODER_LAYERS][:valid, None], out=out[:valid])
    out.zero_()
    planned(0)
    checked = within(expected, out[:valid])
    rows["flashinfer ragged, planned per inference (valid rows)"] = {"us": timed(planned), **checked}
    return {"physical_rows": physical, "bucket_rows": bucket, "valid_rows": valid, "candidates": rows}


def expert(physical: int, chunk: int, valid: int, workspace: torch.Tensor) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(1)
    cache = physical + chunk

    def tensor(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device="cuda").to(torch.bfloat16)

    q = tensor(chunk, DECODER_HEADS, HEAD_DIM)
    k = [tensor(cache, HEAD_DIM) for _ in range(ENCODER_LAYERS)]
    v = [tensor(cache, HEAD_DIM) for _ in range(ENCODER_LAYERS)]
    mask = torch.zeros(cache, dtype=torch.bfloat16, device="cuda")
    mask[valid:physical] = MASK_NEG
    keys = torch.cat((torch.arange(valid), torch.arange(physical, cache))).cuda()
    expected = reference(q, k[0][keys], v[0][keys])
    out = torch.zeros(chunk * DECODER_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    scratch = Scratch(torch.device("cuda"))
    triton_qk_kernel = triton_qk_attention.make_wrappers(scratch)["action_expert_attention"]
    rows: dict[str, dict[str, float]] = {}

    def triton_qk(i: int) -> None:
        triton_qk_kernel(q.view(-1, HEAD_DIM), k[i % ENCODER_LAYERS], v[i % ENCODER_LAYERS], mask, out)
    triton_qk(0)
    checked = within(expected, out.view(chunk, DECODER_HEADS, HEAD_DIM))
    rows["triton-qk (every cache row)"] = {"us": timed(triton_qk), **checked}

    split_scratch = Scratch(torch.device("cuda"))
    split_kernel = split_kv_attention.make_wrappers(split_scratch)["action_expert_attention"]
    (select,) = split_scratch.replay_hooks
    select(valid, physical)

    def split(i: int) -> None:
        split_kernel(q.view(-1, HEAD_DIM), k[i % ENCODER_LAYERS], v[i % ENCODER_LAYERS], mask,
                     out, physical)
    out.zero_()
    split(0)
    checked = within(expected, out.view(chunk, DECODER_HEADS, HEAD_DIM))
    rows["split-kv (valid keys, one block per CTA, merged)"] = {"us": timed(split), **checked}

    paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    paged.plan(torch.tensor([0, chunk], dtype=torch.int32, device="cuda"),
               torch.tensor([0, len(keys)], dtype=torch.int32, device="cuda"),
               keys.int(), torch.tensor([1], dtype=torch.int32, device="cuda"),
               DECODER_HEADS, 1, HEAD_DIM, 1, causal=False, q_data_type=torch.bfloat16)

    def pages(i: int) -> torch.Tensor:
        return paged.run(q, (k[i % ENCODER_LAYERS][:, None, None], v[i % ENCODER_LAYERS][:, None, None]))
    rows["flashinfer paged, one-row pages over both ranges"] = {
        "us": timed(pages), **within(expected, pages(0))}

    gathered_k = [key[keys][:, None].contiguous() for key in k]
    gathered_v = [value[keys][:, None].contiguous() for value in v]

    def single(i: int) -> torch.Tensor:
        return flashinfer.single_prefill_with_kv_cache(
            q, gathered_k[i % ENCODER_LAYERS], gathered_v[i % ENCODER_LAYERS], causal=False)
    rows["flashinfer single_prefill (contiguous valid keys)"] = {
        "us": timed(single), **within(expected, single(0))}
    return {"physical_rows": physical, "chunk": chunk, "valid_rows": valid, "candidates": rows}


def vision(views: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(2)
    qkv = [torch.randn(views, VISION_TOKENS, 3 * VISION_HEADS * VISION_HEAD_DIM, generator=generator,
                       device="cuda").to(torch.bfloat16) for _ in range(VISION_LAYERS)]
    heads = qkv[0].float().view(views, VISION_TOKENS, 3, VISION_HEADS, VISION_HEAD_DIM).permute(0, 2, 3, 1, 4)
    logits = heads[:, 0] @ heads[:, 1].transpose(-1, -2) * VISION_HEAD_DIM ** -0.5
    expected = (torch.softmax(logits, -1) @ heads[:, 2]).transpose(1, 2).reshape(
        views, VISION_TOKENS, VISION_HEADS * VISION_HEAD_DIM)
    out = torch.zeros(views, VISION_TOKENS, VISION_HEADS * VISION_HEAD_DIM, dtype=torch.bfloat16,
                      device="cuda")
    rows: dict[str, dict[str, float]] = {}
    for name, kernel in (("torch (compiled scaled_dot_product_attention)",
                          attention.vision_encoder_attention),
                         ("triton-vision-attention (64 + masked 16 columns)",
                          triton_vision_attention.vision_encoder_attention)):
        out.zero_()
        kernel(qkv[0], out)
        checked = within(expected, out)
        rows[name] = {"us": timed(lambda i: kernel(qkv[i % VISION_LAYERS], out), VISION_LAYERS),
                      **checked}
    return {"views": views, "candidates": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    report = {workload: {"vision": [vision(views)],
                         "backbone": [backbone(physical, valid, workspace) for valid in valids],
                         "expert": [expert(physical, chunk, valid, workspace) for valid in valids]}
              for workload, (views, physical, chunk, valids) in CASES.items()}
    for workload, sites in report.items():
        for site, cases in sites.items():
            for case in cases:
                print(workload, site, case["views"] if site == "vision" else case["valid_rows"],
                      {name: (round(row["us"], 2), bool(row["within"]))
                       for name, row in case["candidates"].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reps": REPS, "flashinfer": flashinfer.__version__,
                                    "workloads": report}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
