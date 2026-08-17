"""Compare H100/Pi0 output against the official OpenPI PyTorch implementation."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from eval.baselines import openpi


def error_metrics(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    """Return the five numerical error metrics reported by this evaluation."""
    if reference.shape != output.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, output={output.shape}")

    reference = reference.float().flatten()
    output = output.float().flatten()
    absolute_error = (output - reference).abs()
    return {
        "max_abs": absolute_error.max().item(),
        "mean_abs": absolute_error.mean().item(),
        "rms_error": torch.sqrt(torch.mean((output - reference) ** 2)).item(),
        "p99_abs": torch.quantile(absolute_error, 0.99).item(),
        "cosine_similarity": F.cosine_similarity(reference, output, dim=0).item(),
    }


def run(checkpoint: str, seed: int = 0, device: str = "cuda") -> dict[str, float]:
    """Run both implementations with identical synthetic inputs and noise."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command on an H100 GPU node")

    from flash_vla.hardware.nvidia.h100.pi0 import Pi0Inference

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    images = torch.rand(
        (3, 224, 224, 3), generator=generator, device=torch_device, dtype=torch.float32
    ) * 2.0 - 1.0
    state = torch.randn((32,), generator=generator, device=torch_device, dtype=torch.float32)
    noise = torch.randn((50, 32), generator=generator, device=torch_device, dtype=torch.float32)

    baseline = openpi.load_model(checkpoint, torch_device)
    reference = openpi.sample_actions(baseline, images, state, noise).float().clone()
    target_weights = openpi.target_checkpoint(baseline)
    del baseline
    torch.cuda.empty_cache()

    engine = Pi0Inference(
        target_weights, num_views=3, chunk_size=50, steps=10, layers=18, fused=True, device=device
    )
    del target_weights
    torch.cuda.empty_cache()
    output = engine.forward(images, state, noise).clone()
    torch.cuda.synchronize()

    metrics = error_metrics(reference, output)
    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="OpenPI model.safetensors or its directory"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run(args.checkpoint, seed=args.seed, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
