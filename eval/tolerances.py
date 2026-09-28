"""Numerical tolerances by precision and evaluation depth.

Intermediate checks retain the prior reference-rounding calibration. Final
action cosine must exceed 0.999 for BF16 or 0.99 for quantized inference.
Numerical checks do not establish model task quality.
"""

from typing import TypedDict


class ToleranceSet(TypedDict):
    shallow: dict[str, float]
    layer0: dict[str, float]
    deepest: dict[str, float]
    end_to_end: dict[str, float]
    max_cosine_step: float


TOLERANCES: dict[str, ToleranceSet] = {
    "bf16": {
        "shallow": {"rel_rms_max": 6.6e-2, "cosine_min": 0.99978},
        "layer0": {"rel_rms_max": 6.6e-2, "cosine_min": 0.99978},
        "deepest": {"rel_rms_max": 3.4e-1, "cosine_min": 0.9943},
        "end_to_end": {"cosine_min": 0.999},
        "max_cosine_step": 0.005,
    },
}
# A quantization recipe's kernels against its fake-quant reference plan. They
# reproduce the reference's arithmetic up to FP32 summation order and every other
# call site keeps its BF16 route, so intermediate checks use BF16 limits.
TOLERANCES["mxfp8"] = {**TOLERANCES["bf16"], "end_to_end": {"cosine_min": 0.99}}


def tolerances(precision: str = "bf16") -> ToleranceSet:
    """Return the numerical requirements for the selected precision."""
    return TOLERANCES[precision].copy()
