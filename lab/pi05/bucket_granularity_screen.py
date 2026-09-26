"""V0.2: is a 64-row bucket worth its graph over a 128-row one, for the BF16 backbone GEMMs?

The attention runs at the exact valid length whatever the bucket (FlashInfer
re-plans per inference), and the MXFP8 GEMMs tile M by 128 rows (the scale
factors' TMA box), so only the BF16 backbone GEMMs can gain from a bucket finer
than 128 rows. Each call site runs cuBLAS and every tile of the stream-K family
at every bucket length a 64- or a 128-row granularity gives the workloads
(`--rows`), over the backbone's distinct layer weights in one graph, and the
fastest candidate is reported per length.

    python -m lab.pi05.bucket_granularity_screen --out results/pi05-rtx5090/replay-extent/v0/bucket-granularity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import cutlass_gemm
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
#: robodojo 811-869 valid rows: 832/896 (64) or 896 (128); libero 518-534: 576 or 640.
ROWS = (576, 640, 832, 896)


def fastest(k: int, n: int, residual: bool, rows: int) -> dict[str, object]:
    """cuBLAS and every family tile at `rows`, over the backbone's layer weights."""
    generator = torch.Generator(device="cuda").manual_seed(0)

    def tensor(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator, device="cuda") * 0.05).to(torch.bfloat16)

    activation = tensor(rows, k)
    weights = [tensor(k, n) for _ in range(ENCODER_LAYERS)]
    output = tensor(rows, n)

    def build(candidate: dict[str, int | str]) -> list[Callable[[], object]]:
        if candidate["config"] == "torch":
            return [lambda weight=weight: torch.addmm(output, activation, weight, out=output)
                    if residual else torch.mm(activation, weight, out=output)
                    for weight in weights]
        plans = [cutlass_gemm.plan(activation, weight, output, config=candidate["config"],
                                   c=output if residual else None, beta=1.0 if residual else 0.0)
                 for weight in weights]
        return [lambda planned=planned: cutlass_gemm.run(planned) for planned in plans]

    outcome = sweep([{"config": "torch"}, *({"config": config} for config in TILES)], build,
                    lambda launches, i: launches[i % ENCODER_LAYERS](),
                    label=f"M={rows} K={k} N={n}", n_inner=ENCODER_LAYERS, reps=REPS,
                    verbose=False)
    cutlass_gemm.release()
    return {"rows": rows, "best": outcome.best, "top": outcome.results[:3]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cutlass_gemm.set_pdl(False)
    report = {site: [fastest(k, n, residual, rows) for rows in ROWS]
              for site, (k, n, residual) in SITES.items()}
    for site, rows in report.items():
        print(site, [(row["rows"], row["best"]["us"], row["best"]["config"]) for row in rows],
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reps": REPS, "rows": ROWS, "sites": report}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
