"""G1: the tunable choices of the rtx5090/pi05 call sites G0 ranked, at each workload's geometry.

For every workload the Target builds, each GEMM call site runs cuBLAS
(`torch`) and every tile of the stream-K CUTLASS family
(`rtx5090/pi0/backends/cuda/cutlass_gemm.cu`, configs 0-11; `Site.deployed` is
what the Pi0.5 Target ran before G1, the expert's gated-residual GEMM being
config 9's tile with its own epilogue), and the vision attention runs every
SDPA backend. Each candidate is
timed by `flash_vla.tuning.sweep` in one CUDA graph over as many distinct
weights as the forward reads (one per layer, or the action output's one weight
once per denoising step), so the weight reads are as cold as in the engine,
and checked against torch within the shallow tolerances. Values are seeded
random: timing does not depend on them.

    python -m lab.pi05.geometry_screen --out results/pi05-rtx5090/workload-generalization/g1/geometry-screen.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from eval.metrics import error_metrics
from eval.tolerances import tolerances
from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import cutlass_gemm
from flash_vla.inference import declare, get_target
from flash_vla.models.pi05.spec import (ACTION_DIM, DECODER_DIM, DECODER_FFN, DECODER_HEADS,
                                        ENCODER_DIM, HEAD_DIM, QKV_WIDTH, VISION_DIM, VISION_FFN,
                                        VISION_HEAD_DIM, VISION_HEADS, VISION_TOKENS)
from flash_vla.tuning import sweep

TARGET = "rtx5090/pi05"
REPS = 30
#: The tiles of the family: configs 12-14 fuse GELU epilogues no call site here uses.
TILES = range(12)
SDPA_BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                 "cudnn": SDPBackend.CUDNN_ATTENTION}


@dataclass(frozen=True)
class Site:
    """One GEMM call site: rows follow the workload, the rest is the model's."""
    call_site: str
    stage: str
    k: int
    n: int
    #: `bias`: a broadcast bias; `residual`: C aliases D with beta 1; `none`: beta 0.
    source: str
    #: The family config, or `torch` for cuBLAS, the Target ran before G1.
    deployed: int | str
    #: The shape number counting its distinct weights, one per layer; `None`
    #: for one weight read once per denoising step.
    weights: str | None


SITES = (
    Site("vision_encoder_norm_qkv", "vision", VISION_DIM, 3 * VISION_DIM, "bias", 10,
         "vision_layers"),
    Site("vision_encoder_out_proj_residual", "vision", VISION_DIM, VISION_DIM, "bias", 10,
         "vision_layers"),
    Site("vision_encoder_norm_ffn_up", "vision", VISION_DIM, VISION_FFN, "bias", 10,
         "vision_layers"),
    Site("vision_encoder_ffn_down_residual", "vision", VISION_FFN, VISION_DIM, "bias", 10,
         "vision_layers"),
    Site("action_expert_out_proj_residual", "expert", DECODER_HEADS * HEAD_DIM, DECODER_DIM,
         "residual", 9, "layers"),
    Site("action_expert_ffn_down_residual", "expert", DECODER_FFN, DECODER_DIM, "residual", 9,
         "layers"),
    Site("action_expert_action_out_proj", "expert", DECODER_DIM, ACTION_DIM, "none", "torch",
         None),
    Site("llm_backbone_norm_qkv_rope", "backbone", ENCODER_DIM, QKV_WIDTH, "none", "torch",
         "layers"),
)


def gemm_sweep(site: Site, rows: int, layers: int, calls: int) -> dict[str, object]:
    """cuBLAS and every tile of the family on `layers` distinct weights of `site`
    at `rows`, `calls` launches per graph."""
    generator = torch.Generator(device="cuda").manual_seed(0)

    def tensor(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator, device="cuda") * 0.05).to(torch.bfloat16)

    activations = [tensor(rows, site.k) for _ in range(layers)]
    weights = [tensor(site.k, site.n) for _ in range(layers)]
    outputs = [torch.empty(rows, site.n, dtype=torch.bfloat16, device="cuda") for _ in range(layers)]
    # With no source, C is never read: the outputs stand in.
    sources = ([tensor(site.n) if site.source == "bias" else tensor(rows, site.n)
                for _ in range(layers)] if site.source != "none" else outputs)
    expected = [(a.float() @ b.float() + (0 if site.source == "none" else c.float())).to(torch.bfloat16)
                for a, b, c in zip(activations, weights, sources)]
    limit = tolerances()["shallow"]

    def build(candidate: dict[str, int | str]) -> list[Callable[[], object]]:
        launches: list[Callable[[], object]] = []
        for a, b, c, d in zip(activations, weights, sources, outputs):
            if candidate["config"] == "torch" and site.source == "none":
                launches.append(lambda a=a, b=b, d=d: torch.mm(a, b, out=d))
            elif candidate["config"] == "torch":
                launches.append(lambda a=a, b=b, c=c if site.source == "bias" else d, d=d:
                                torch.addmm(c, a, b, out=d))
            else:
                planned = cutlass_gemm.plan(
                    a, b, d, config=candidate["config"],
                    c=None if site.source == "none" else c if site.source == "bias" else d,
                    beta=0.0 if site.source == "none" else 1.0,
                    broadcast_c=site.source == "bias")
                launches.append(lambda planned=planned: cutlass_gemm.run(planned))
        return launches

    def correct(launches: list[Callable[[], object]]) -> bool:
        passed = True
        for index, launch in enumerate(launches):
            if site.source == "residual":
                outputs[index].copy_(sources[index])
            launch()
            metrics = error_metrics(expected[index], outputs[index])
            passed = passed and (metrics["rel_rms"] <= limit["rel_rms_max"]
                                 and metrics["cosine_similarity"] >= limit["cosine_min"])
        return passed

    outcome = sweep([{"config": "torch"}, *({"config": config} for config in TILES)], build,
                    lambda launches, i: launches[i % layers](),
                    correct=correct, label=f"{site.call_site} M={rows}", n_inner=calls,
                    reps=REPS)
    cutlass_gemm.release()
    return {"rows": rows, "k": site.k, "n": site.n, "layers": layers, "calls": calls,
            "deployed_config": site.deployed, "results": outcome.results,
            "failed": {category: count for category, count in outcome.counts.items() if count}}


def attention_sweep(views: int, layers: int) -> dict[str, object]:
    """Every SDPA backend on `layers` distinct (views, heads, 256, head_dim) Q, K, V."""
    shape = (views, VISION_HEADS, VISION_TOKENS, VISION_HEAD_DIM)
    qkv = [tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
           for _ in range(layers)]
    out = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(layers)]
    expected = [torch.nn.functional.scaled_dot_product_attention(*(t.float() for t in tensors))
                .to(torch.bfloat16) for tensors in qkv]
    limit = tolerances()["shallow"]

    def build(candidate: dict[str, str]) -> Callable[[int], None]:
        backend = SDPA_BACKENDS[candidate["backend"]]

        def invoke(i: int) -> None:
            with sdpa_kernel([backend]):
                out[i % layers].copy_(torch.nn.functional.scaled_dot_product_attention(
                    *qkv[i % layers]))
        return invoke

    def correct(invoke: Callable[[int], None]) -> bool:
        invoke(0)
        metrics = error_metrics(expected[0], out[0])
        return (metrics["rel_rms"] <= limit["rel_rms_max"]
                and metrics["cosine_similarity"] >= limit["cosine_min"])

    outcome = sweep([{"backend": name} for name in SDPA_BACKENDS], build,
                    lambda invoke, i: invoke(i), correct=correct,
                    label=f"vision_encoder_attention views={views}", n_inner=layers, reps=REPS)
    return {"views": views, "layers": layers, "deployed": "torch default", "results": outcome.results,
            "failed": {category: count for category, count in outcome.counts.items() if count},
            "errors": {category: repr(error) for category, error in outcome.errors.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cutlass_gemm.set_pdl(False)
    report: dict[str, object] = {"target": TARGET, "reps": REPS, "workloads": {}}
    for workload in get_target(TARGET).workloads:
        shape = declare(TARGET, workload=workload).shape
        rows = {"vision": shape["visual_tokens"], "expert": shape["expert_tokens"],
                "backbone": shape["prefix_len"]}
        report["workloads"][workload] = {
            "gemm": {site.call_site: gemm_sweep(
                site, rows[site.stage], 1 if site.weights is None else shape[site.weights],
                shape["steps"] if site.weights is None else shape[site.weights])
                     for site in SITES},
            "vision_encoder_attention": attention_sweep(shape["num_views"],
                                                        shape["vision_layers"]),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
