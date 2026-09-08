"""Compare H100/Pi0 output against the official OpenPI PyTorch implementation.

    python -m eval.pi0.reference                       # the registry's checkpoint, reference route
    python -m eval.pi0.reference --plan shipped
    OPENPI_PI0_CHECKPOINT=/path/to/pi0 \
    OPENPI_PI0_MODEL_REVISION=immutable-id python -m eval.pi0.reference

The official-baseline tier of the acceptance registry: the Target's reference
route, built from the OpenPI checkpoint's weights, against OpenPI's own
forward on the same inputs and noise, judged at the registry's full-depth
tolerance. The promotion gate runs it with no arguments under the OpenPI
interpreter; the checkpoint comes from `eval.acceptance.OPENPI_PI0_CHECKPOINT`
and a missing one is reported as unavailable (exit 3), never as a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from eval.baselines import openpi
from eval.acceptance import OPENPI_PI0_CHECKPOINT, OPENPI_PI0_MODEL_REVISION, tolerances
from eval.metrics import error_metrics

#: The exit code and stderr marker the gate reads as "unavailable".
UNAVAILABLE = 3


def run(checkpoint: str, *, model_revision: str, seed: int = 0, device: str = "cuda",
        plan: str = "reference") -> dict[str, object]:
    """Run both implementations with identical synthetic inputs and noise.

    `plan` is the call-site plan the runner is built with; the reference route
    by default, so the tier judges the correctness oracle every candidate is
    compared against, as the Pi0.5 tier does.
    """
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

    engine = ModelRunner(TARGET, target_weights, model_revision=model_revision,
                         plan=plan, device=device, num_views=3, chunk_size=50,
                         steps=10, layers=18)
    del target_weights
    torch.cuda.empty_cache()
    output = engine.forward(images=images, state=state, noise=noise).clone()
    torch.cuda.synchronize()

    # A full forward on the checkpoint against the official implementation:
    # the registry's full-depth tolerance of the runner's precision policy.
    tolerance = tolerances(engine.identity.precision)["deepest"]
    metrics = error_metrics(reference, output)
    report = {"identity": engine.identity.as_dict(), **metrics,
              "threshold_key": "deepest", "tolerance": dict(tolerance),
              "passed": bool(torch.isfinite(output).all().item()
                             and metrics["cosine_similarity"] > tolerance["cosine_min"]
                             and metrics["rel_rms"] < tolerance["rel_rms_max"])}
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=None,
        help="OpenPI model.safetensors or its directory (default: the registry's "
             "OPENPI_PI0_CHECKPOINT)"
    )
    parser.add_argument(
        "--model-revision", default=None,
        help="immutable checkpoint ID (default: registered ID for the default checkpoint)"
    )
    parser.add_argument("--plan", default="reference",
                        help="the call-site plan to build the runner with (default: reference)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint or OPENPI_PI0_CHECKPOINT)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    if not checkpoint.is_file():
        print(f"baseline unavailable: OpenPI checkpoint not found at {checkpoint} "
              "(set OPENPI_PI0_CHECKPOINT)", file=sys.stderr)
        return UNAVAILABLE
    model_revision = (args.model_revision if args.checkpoint is not None
                      else args.model_revision or OPENPI_PI0_MODEL_REVISION)
    if not model_revision:
        print("baseline unavailable: an overridden OpenPI checkpoint needs "
              "OPENPI_PI0_MODEL_REVISION or --model-revision", file=sys.stderr)
        return UNAVAILABLE
    report = run(str(checkpoint), model_revision=model_revision, seed=args.seed,
                 device=args.device, plan=args.plan)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
