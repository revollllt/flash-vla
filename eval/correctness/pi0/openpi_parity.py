"""Compare H100/Pi0 output against the official OpenPI PyTorch implementation."""

from __future__ import annotations

import argparse
import json

import torch

from eval.baselines import openpi
from eval.correctness.metrics import error_metrics


def run(checkpoint: str, seed: int = 0, device: str = "cuda") -> dict[str, float]:
    """Run both implementations with identical synthetic inputs and noise."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this command on an H100 GPU node")

    from flash_vla.hardware.nvidia.h100.pi0 import TARGET
    from flash_vla.runtime import ModelRunner

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

    engine = ModelRunner(TARGET, target_weights, device=device, num_views=3, chunk_size=50,
                         steps=10, layers=18)
    del target_weights
    torch.cuda.empty_cache()
    output = engine.forward(images=images, state=state, noise=noise).clone()
    torch.cuda.synchronize()

    metrics = {"identity": engine.identity.as_dict(), **error_metrics(reference, output)}
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
