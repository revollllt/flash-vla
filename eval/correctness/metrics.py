"""The numerical error metrics every parity script reports, in one place.

Five metrics, one implementation: parity scripts import `error_metrics` and
never re-declare it, so a threshold in the acceptance registry means the same
thing in every report. Root-mean-square error is the reported form of mean
squared error. Cosine similarity and RMS are dominated by the largest
channels, and an action chunk mixes channels of different scale, so `max_abs`
and `p99_abs` are always reported beside them.

Both tensors are compared in fp32 after flattening; a shape mismatch is an
error, not a broadcast.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

METRICS = ("max_abs", "mean_abs", "rms_error", "p99_abs", "cosine_similarity")


def error_metrics(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    """The five error metrics of `output` against `reference`, same shape required."""
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


__all__ = ["METRICS", "error_metrics"]
