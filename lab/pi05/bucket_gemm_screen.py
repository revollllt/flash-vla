"""T1: the BF16 backbone GEMMs at every replay bucket's rows, deployed against the best.

Each workload's backbone replays at 64-row buckets of its prompt range
(`runtime/replay.py`): RoboDojo's 832, 896 and 968 rows, LIBERO's 576 and 712.
Each call site runs cuBLAS and every tile of the stream-K family at each of
them, over the backbone's distinct layer weights in one graph. The report
keeps every candidate's time and names the deployed one (the backends' tables
at the time of the run), so a table entry is adopted where the best beats the
deployed candidate by more than 5%. A candidate whose first-layer result misses
the shallow tolerance is dropped, not ranked.

    python -m lab.pi05.bucket_gemm_screen --out results/pi05-rtx5090/replay-extent/t/bucket-gemm-screen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

from eval.metrics import error_metrics
from eval.tolerances import tolerances

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import cutlass_gemm
from flash_vla.hardware.nvidia.rtx5090.pi05.backends import cutlass_backbone, fused_prefix_qkv
from flash_vla.models.pi05.spec import ENCODER_DIM, ENCODER_FFN, ENCODER_LAYERS, QKV_WIDTH
from flash_vla.tuning import sweep

REPS = 30
TILES = range(12)
#: Call site -> (K, N, whether C aliases D with beta 1).
SITES = {
    "llm_backbone_norm_qkv_rope": (ENCODER_DIM, QKV_WIDTH, False),
    "llm_backbone_out_proj_residual": (ENCODER_DIM, ENCODER_DIM, True),
    "llm_backbone_norm_gated_ffn (gate or up)": (ENCODER_DIM, ENCODER_FFN, False),
    "llm_backbone_ffn_down_residual": (ENCODER_FFN, ENCODER_DIM, True),
}
#: The bucket rows of both workloads (docs/workloads.md).
ROWS = (576, 712, 832, 896, 968)
#: A candidate that beats the deployed one by more than this is adopted.
MARGIN = 0.05


def screen(site: str, k: int, n: int, residual: bool, rows: int) -> dict[str, object]:
    """cuBLAS and every family tile at `rows`, over the backbone's layer weights."""
    generator = torch.Generator(device="cuda").manual_seed(0)

    def tensor(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator, device="cuda") * 0.05).to(torch.bfloat16)

    activation = tensor(rows, k)
    weights = [tensor(k, n) for _ in range(ENCODER_LAYERS)]
    output = tensor(rows, n)
    source = output.clone()
    expected = activation.float() @ weights[0].float() + (source.float() if residual else 0.0)
    limit = tolerances()["shallow"]

    def build(candidate: dict[str, int | str]) -> list[Callable[[], object]]:
        if candidate["config"] == "torch":
            return [lambda weight=weight: torch.addmm(output, activation, weight, out=output)
                    if residual else torch.mm(activation, weight, out=output)
                    for weight in weights]
        plans = [cutlass_gemm.plan(activation, weight, output, config=candidate["config"],
                                   c=output if residual else None, beta=1.0 if residual else 0.0)
                 for weight in weights]
        return [lambda planned=planned: cutlass_gemm.run(planned) for planned in plans]

    def correct(launches: list[Callable[[], object]]) -> bool:
        """The first layer's GEMM from a fresh output, within the shallow tolerance."""
        output.copy_(source)
        launches[0]()
        metrics = error_metrics(expected, output.float())
        return (metrics["rel_rms"] <= limit["rel_rms_max"]
                and metrics["cosine_similarity"] >= limit["cosine_min"])

    outcome = sweep([{"config": "torch"}, *({"config": config} for config in TILES)], build,
                    lambda launches, i: launches[i % ENCODER_LAYERS](), correct=correct,
                    label=f"M={rows} K={k} N={n}", n_inner=ENCODER_LAYERS, reps=REPS,
                    verbose=False)
    cutlass_gemm.release()
    # The candidate the backends run at this site and row count.
    geometry = (rows, k, n)
    if site == "llm_backbone_norm_qkv_rope":
        current = fused_prefix_qkv.CUTLASS_CONFIGS.get(geometry, "torch")
    elif geometry in cutlass_backbone.CUBLAS_GEOMETRIES:
        current = "torch"
    else:
        current = cutlass_backbone.CUTLASS_CONFIGS.get(geometry, 0)
    deployed_us = next(row["us"] for row in outcome.results if row["config"] == current)
    return {"rows": rows, "deployed": {"config": current, "us": deployed_us}, "best": outcome.best,
            "adopt": outcome.best["us"] < deployed_us * (1 - MARGIN), "candidates": outcome.results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cutlass_gemm.set_pdl(False)
    report = {site: [screen(site, k, n, residual, rows) for rows in ROWS]
              for site, (k, n, residual) in SITES.items()}
    for site, rows in report.items():
        print(site, [(row["rows"], row["deployed"]["config"], row["deployed"]["us"],
                      row["best"]["config"], row["best"]["us"], row["adopt"]) for row in rows],
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reps": REPS, "rows": ROWS, "margin": MARGIN,
                                    "sites": report}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
