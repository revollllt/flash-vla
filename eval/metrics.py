"""The numerical error metrics every check reports, in one place.

Six metrics, one implementation: checks import `error_metrics` and never
re-declare it, so a threshold in the acceptance registry means the same thing
in every report. Two of them gate: `rel_rms` (root-mean-square error over the
reference's root-mean-square, a scale-free error that no single channel
dominates) and `cosine_similarity` (direction). `rms_error` is the reported
form of mean squared error; `max_abs` and `p99_abs` are always reported beside
the two gates because an action chunk mixes channels of different scale.

Both tensors are compared in float64 after flattening: in float32 the cosine
of a tensor with itself reads 0.99994 at a million elements (the PR1 vision
output), which is below the layer-0 tolerance, so float32 would gate on the
accumulator rather than on the kernel. A shape mismatch is an error, not a
broadcast.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

METRICS = ("max_abs", "mean_abs", "rms_error", "rel_rms", "p99_abs", "cosine_similarity")


def error_metrics(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    """The six error metrics of `output` against `reference`, same shape required."""
    if reference.shape != output.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, output={output.shape}")
    reference = reference.double().flatten()
    output = output.double().flatten()
    absolute_error = (output - reference).abs()
    rms_error = torch.sqrt(torch.mean(absolute_error ** 2)).item()
    reference_rms = torch.sqrt(torch.mean(reference ** 2)).item()
    if reference_rms > 0:
        rel_rms = rms_error / reference_rms
    else:
        rel_rms = 0.0 if rms_error == 0 else float("inf")
    return {
        "max_abs": absolute_error.max().item(),
        "mean_abs": absolute_error.mean().item(),
        "rms_error": rms_error,
        "rel_rms": rel_rms,
        "p99_abs": torch.quantile(absolute_error, 0.99).item(),
        "cosine_similarity": F.cosine_similarity(reference, output, dim=0).item(),
    }


__all__ = ["METRICS", "error_metrics"]
