"""T2 parity of the SigLIP vision backends against their ABI mirrors.

    python -m lab.siglip.parity                    # every implemented call site
    python -m lab.siglip.parity --site vision_encoder_norm_qkv --seed 1

Kernel-level parity: each backend wrapper and the T2 mirror beside it are
handed the SAME tensors and the same `out` buffer, so any difference is an
implementation difference. Random weights are deliberate (`kernel-design`
parity contract); a checkpoint would only add coverage of weight conversion.

This is the structural gate -- a wrong layout, bias, epsilon or activation
shows here at full size, one layer deep, before any engine is built. It is not
the promotion gate: that is `eval.correctness` against the Target's reference
plan, which this script deliberately does not duplicate.

Gates on the registry's shallow tolerance for the runner's precision policy;
the other four metrics are reported.
"""
from __future__ import annotations

import argparse
import json

import torch

from eval.acceptance import tolerances
from eval.metrics import error_metrics
from flash_vla.hardware.nvidia.h100.siglip import geometry
from flash_vla.hardware.nvidia.h100.siglip.backends import cublas, cuda

#: Every (call site -> candidate wrapper, reference mirror) pair this script knows.
IMPLEMENTATIONS = {
    "siglip-cublas": (cublas.wrappers.ALL_WRAPPERS, cublas.norm_gemm_reference.REFERENCES),
    "siglip-cuda": (cuda.wrappers.ALL_WRAPPERS,
                    {**cuda.norm_gemm_reference.REFERENCES,
                     **cuda.attention_reference.REFERENCES}),
}


def _inputs(site: str, views: int, generator, device, dtype):
    """The op's arguments, positionally, plus a fresh `out` for each side.

    Scales are deliberately unit-normal: the LayerNorm reduction and the GELU
    saturation are where bf16 surprises hide, and both are exercised there.
    """
    def randn(*shape):
        return torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)

    x = randn(views, geometry.TOKENS, geometry.DIM)
    norm_w, norm_b = randn(geometry.DIM), randn(geometry.DIM)
    if site == "vision_encoder_norm_qkv":
        weight, bias, width = randn(geometry.DIM, geometry.QKV_DIM), randn(geometry.QKV_DIM), geometry.QKV_DIM
    elif site == "vision_encoder_norm_ffn_up":
        weight, bias, width = randn(geometry.DIM, geometry.FFN), randn(geometry.FFN), geometry.FFN
    elif site == "vision_encoder_attention":
        # The attention op takes the packed projection and nothing else.
        qkv = randn(views, geometry.TOKENS, geometry.QKV_DIM)
        out = torch.zeros((views, geometry.TOKENS, geometry.DIM), device=device, dtype=dtype)
        return (qkv,), out
    else:
        raise KeyError(f"no input builder for {site}")
    out = torch.zeros((views, geometry.TOKENS, width), device=device, dtype=dtype)
    return (x, norm_w, norm_b, weight, bias), out


def check_site(site: str, candidate, reference, *, views: int, seed: int, device: str) -> dict:
    """One call site: run both sides on identical inputs, return the six metrics."""
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)
    args, out_candidate = _inputs(site, views, generator, torch.device(device), dtype)
    out_reference = torch.zeros_like(out_candidate)

    candidate(*args, out_candidate)
    reference(*args, out_reference)
    torch.cuda.synchronize()

    gate = tolerances("bf16")["shallow"]
    metrics = error_metrics(out_reference, out_candidate)
    finite = bool(torch.isfinite(out_candidate).all().item())
    return {
        "call_site": site,
        "shape": tuple(out_candidate.shape),
        "metrics": metrics,
        "tolerance": dict(gate),
        "finite": finite,
        "passed": bool(finite
                       and metrics["rel_rms"] < gate["rel_rms_max"]
                       and metrics["cosine_similarity"] > gate["cosine_min"]),
    }


def run(backend: str, only_sites=None, *, views: int = 3, seed: int = 0,
        device: str = "cuda") -> dict:
    """Every implemented call site of `backend`, or the subset in `only_sites`."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command on an H100 GPU node")
    wrappers, references = IMPLEMENTATIONS[backend]
    sites = sorted(set(wrappers) & set(references))
    if only_sites:
        sites = [s for s in sites if s in set(only_sites)]
    if not sites:
        raise SystemExit(f"no call sites selected for {backend}")
    checks = [check_site(s, wrappers[s], references[s], views=views, seed=seed, device=device)
              for s in sites]
    report = {"backend": backend, "views": views, "seed": seed,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "checks": checks, "passed": all(c["passed"] for c in checks)}
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", default="siglip-cublas", choices=sorted(IMPLEMENTATIONS))
    parser.add_argument("--site", action="append", default=None, help="restrict to a call site")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return 0 if run(args.backend, args.site, views=args.views, seed=args.seed,
                    device=args.device)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
