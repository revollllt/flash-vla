"""V0.4: the Pi0.5 attentions on RTX 5090, the deployed kernels against FlashInfer.

Both attentions are multi-query (8 query heads, one KV head of width 256) and
bidirectional, masked on keys only:

- backbone: queries and keys are the prefix rows. The deployed kernel
  (`models/pi05/attention.py`, compiled matmul-softmax-matmul) runs every
  physical row under the additive mask; FlashInfer runs the valid rows only,
  planned per inference on a graph captured once (`BatchPrefillWithRaggedKVCacheWrapper`
  with `use_cuda_graph`), or as one call at that length (`single_prefill_with_kv_cache`).
- expert: the chunk's queries over the valid prefix keys and the chunk's own,
  which sit apart in the cache. The deployed kernels (`triton_qk_attention`)
  scan every cache row under the mask; FlashInfer reads the two ranges through
  a one-row page table (`BatchPrefillWithPagedKVCacheWrapper`), or one call over
  a contiguous copy of the keys it would read (`single_prefill`, a lower bound
  on what a gather-free contiguous layout would cost).

Each candidate is checked on the first layer's cache against a float32 softmax
over the valid keys, then timed in one CUDA graph over the backbone's 18
distinct layer caches.

    python -m lab.pi05.attention_screen --out results/pi05-rtx5090/replay-extent/v0/attention-screen.json
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
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import triton_qk_attention
from flash_vla.models.pi05 import attention
from flash_vla.models.pi05.spec import DECODER_HEADS, ENCODER_LAYERS, HEAD_DIM, MASK_NEG
from flash_vla.runtime.cuda import graph_samples
from flash_vla.runtime.workspace import Scratch

REPS = 30
WORKSPACE_BYTES = 256 << 20
#: Workload -> (physical prefix rows, chunk, valid prefix rows at the ends of its range).
CASES = {"robodojo": (968, 50, (811, 869)), "libero": (712, 10, (518, 534))}


def timed(launch: Callable[[int], object]) -> float:
    """Median microseconds per call over graph replays of 18 calls."""
    return statistics.median(graph_samples(launch, n_inner=ENCODER_LAYERS, reps=REPS)) * 1e3


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

    def deployed(i: int) -> None:
        attention.llm_backbone_attention(q[i % ENCODER_LAYERS].view(-1, HEAD_DIM), k[i % ENCODER_LAYERS],
                                         v[i % ENCODER_LAYERS], HEAD_DIM ** -0.5, mask,
                                         out.view(-1, HEAD_DIM))
    deployed(0)
    checked = within(expected, out[:valid])
    rows["deployed (all physical rows)"] = {"us": timed(deployed), **checked}

    def single(i: int) -> torch.Tensor:
        return flashinfer.single_prefill_with_kv_cache(
            q[i % ENCODER_LAYERS][:valid], k[i % ENCODER_LAYERS][:valid, None],
            v[i % ENCODER_LAYERS][:valid, None], causal=False)
    rows["flashinfer single_prefill (valid rows)"] = {"us": timed(single), **within(expected, single(0))}

    indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
    ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", use_cuda_graph=True, qo_indptr_buf=indptr,
        kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device="cuda"), backend="fa2")
    bounds = torch.tensor([0, valid], dtype=torch.int32)
    ragged.plan(bounds, bounds, DECODER_HEADS, 1, HEAD_DIM, causal=False, q_data_type=torch.bfloat16)

    def planned(i: int) -> None:
        ragged.run(q[i % ENCODER_LAYERS][:valid], k[i % ENCODER_LAYERS][:valid, None],
                   v[i % ENCODER_LAYERS][:valid, None], out=out[:valid])
    out.zero_()
    planned(0)
    checked = within(expected, out[:valid])
    rows["flashinfer ragged, planned per inference (valid rows)"] = {"us": timed(planned), **checked}
    return {"physical_rows": physical, "valid_rows": valid, "candidates": rows}


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
    deployed_kernel = triton_qk_attention.make_wrappers(scratch)["action_expert_attention"]
    rows: dict[str, dict[str, float]] = {}

    def deployed(i: int) -> None:
        deployed_kernel(q.view(-1, HEAD_DIM), k[i % ENCODER_LAYERS], v[i % ENCODER_LAYERS], mask, out)
    deployed(0)
    checked = within(expected, out.view(chunk, DECODER_HEADS, HEAD_DIM))
    rows["deployed (every cache row)"] = {"us": timed(deployed), **checked}

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    report = {workload: {"backbone": [backbone(physical, valid, workspace) for valid in valids],
                         "expert": [expert(physical, chunk, valid, workspace) for valid in valids]}
              for workload, (physical, chunk, valids) in CASES.items()}
    for workload, sites in report.items():
        for site, cases in sites.items():
            for case in cases:
                print(workload, site, case["valid_rows"],
                      {name: (round(row["us"], 2), bool(row["within"]))
                       for name, row in case["candidates"].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reps": REPS, "flashinfer": flashinfer.__version__,
                                    "workloads": report}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
