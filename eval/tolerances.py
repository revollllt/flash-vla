"""Existing bf16 numerical tolerances, by evaluation depth.

These values retain the prior reference-rounding calibration; this cleanup does
not relax them. Multi-step synthetic-weight drift is reported separately from
model task quality.
"""

TOLERANCES = {
    "bf16": {
        "shallow": {"rel_rms_max": 6.6e-2, "cosine_min": 0.99978},
        "layer0": {"rel_rms_max": 6.6e-2, "cosine_min": 0.99978},
        "deepest": {"rel_rms_max": 3.4e-1, "cosine_min": 0.9943},
        "max_cosine_step": 0.005,
    },
}
# A quantization recipe's kernels against its fake-quant reference plan. They
# reproduce the reference's arithmetic up to FP32 summation order and every other
# call site keeps its BF16 route, so the BF16 requirements apply unchanged.
TOLERANCES["mxfp8"] = TOLERANCES["bf16"]


def tolerances(precision: str = "bf16"):
    """Return the existing numerical requirements for the selected precision."""
    return dict(TOLERANCES[precision])
