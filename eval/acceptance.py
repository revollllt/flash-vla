"""The acceptance registry: framework defaults and per-Target entries.

What the human defines lives here and nowhere else -- accuracy requirements,
the metric and framework conventions, and each Target's budget. The latency
objective is not a number in this file: it is derived from the Target's floor
model, and the optimization closes the gap to it. See
`ARCHITECTURE.md`.

Structure:

- `DEFAULTS` holds everything that is the same for every Target: the metric
  set, the statistics, repetition counts, the mandatory gates and their
  comparison structure, the noise policy, and the tolerance defaults keyed by
  precision policy (the tolerance for a `bf16` Target is a property of `bf16`,
  not of the model).
- `TARGETS` holds one entry per Target with only what differs: its budget,
  the model-specific scripts that implement the official-baseline tier, and
  any override of a default, each with a reason.

`for_target(name)` merges the two. No torch, no device: the plan registry's
precedent, so offline tools and the promotion gate read it without a GPU.
Parity scripts read their thresholds from `tolerances(precision)` rather than
declaring them, so a change here is a change everywhere.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

#: The latency metrics `benchmarks/latency.py` reports.
LATENCY_METRICS = ("chunk_latency", "device_latency", "host_time", "segment_latency",
                   "overhead")
#: Every latency metric reports all three.
STATISTICS = ("min", "median", "p99")
#: The five correctness metrics of `eval/metrics.py`.
CORRECTNESS_METRICS = ("max_abs", "mean_abs", "rms_error", "p99_abs", "cosine_similarity")

DEFAULTS: dict[str, Any] = {
    "latency": {
        "metrics": LATENCY_METRICS,
        "statistics": STATISTICS,
        "reps": 100,
        "warmup": 5,
        # Below this many repetitions the 99th percentile is the maximum; it is
        # then reported as insufficient rather than as a tail.
        "p99_min_reps": 100,
        # Deployment does not lock clocks, so neither does any measurement.
        "clocks": "unlocked",
        # A delta is claimable only from a same-process A/B/A whose control leg
        # reproduces, and only above the calibrated minimum detectable effect.
        "deltas": "same_process_aba",
        "objective": "gap_to_structural_floor",
        "candidate_rule": {
            "improve": ("chunk_latency", "min"),
            "no_regression": (("chunk_latency", "median"), ("chunk_latency", "p99")),
        },
    },
    "correctness": {
        "metrics": CORRECTNESS_METRICS,
        # Keyed by precision policy: a tolerance is a property of the rounding,
        # not of the model. bf16 values are the ones every shipped gate uses.
        "tolerances": {
            "bf16": {
                # Single step, single layer, against the in-engine reference
                # route: nothing has accumulated, so a wrong layout, mask, fold
                # or wiring shows at full size.
                "shallow_cosine": 0.999,
                # Layer 0 against the official baseline.
                "layer0_cosine": 0.9999,
                # Full depth against the official baseline on random weights:
                # rounding drift through every bf16 layer is expected.
                "deepest_cosine": 0.99,
                # A single layer losing more than this between consecutive
                # layers is a bug in that layer, not accumulation.
                "max_cosine_step": 0.005,
            },
        },
        # Ordered. `mode` is gate or report; `requires` names a Target
        # capability without which the check is skipped and recorded as such.
        "checks": (
            {"check": "replay_determinism", "mode": "gate"},
            {"check": "finiteness", "mode": "gate"},
            {"check": "in_engine_shallow", "oracle": "in_engine_reference",
             "config": {"steps": 1, "layers": 1}, "threshold": "shallow_cosine",
             "mode": "gate"},
            {"check": "in_engine_deep", "oracle": "in_engine_reference",
             "config": {"steps": 1, "layers": "full"}, "threshold": "shallow_cosine",
             "mode": "report"},
            {"check": "in_engine_multistep", "oracle": "in_engine_reference",
             "config": {"steps": "full", "layers": "full"}, "mode": "report"},
            {"check": "baseline_layer0", "oracle": "official_baseline",
             "threshold": "layer0_cosine", "mode": "gate", "requires": "baseline_adapter"},
            {"check": "baseline_depth", "oracle": "official_baseline",
             "threshold": "deepest_cosine", "step": "max_cosine_step", "mode": "report",
             "requires": "baseline_adapter"},
        ),
    },
    # Out of scope this phase; the slot exists so a precision policy other than
    # bf16 cannot be promoted while it is empty.
    "policy_quality": {"suite": None, "threshold": None},
}

#: Per-Target entries: budget, the scripts that implement the official-baseline
#: tier for this model, and overrides with reasons. Nothing that the model spec
#: or the engine already holds is restated here.
TARGETS: dict[str, dict[str, Any]] = {
    "hardware/nvidia/h100/pi05": {
        "budget": {"candidates": 6, "non_improving": 3, "jobs": 12},
        "scripts": {
            "in_engine_reference": "eval.correctness",
            "official_baseline": ("eval.pi05.reference",),
        },
        "capabilities": ("baseline_adapter",),
        "overrides": {},
    },
    "hardware/nvidia/h100/pi0": {
        "budget": {"candidates": 6, "non_improving": 3, "jobs": 12},
        "scripts": {
            "in_engine_reference": "eval.correctness",
            "official_baseline": ("eval.pi0.reference",),
        },
        "capabilities": ("baseline_adapter",),
        "overrides": {},
    },
}


def tolerances(precision: str = "bf16") -> Mapping[str, float]:
    """The correctness tolerances for one precision policy."""
    try:
        return dict(DEFAULTS["correctness"]["tolerances"][precision])
    except KeyError:
        raise KeyError(f"no tolerances defined for precision policy {precision!r}; "
                       f"defined: {sorted(DEFAULTS['correctness']['tolerances'])}") from None


def for_target(name: str) -> dict[str, Any]:
    """The framework defaults merged with `name`'s entry.

    A Target without an entry gets every default and no budget; it can run
    every check but cannot enter the promotion gate. An override replaces one
    dotted path in the defaults (`"latency.reps": 200`) and must carry a
    reason in the entry, which is kept beside the merged value.
    """
    merged = deepcopy(DEFAULTS)
    entry = TARGETS.get(name, {})
    merged["target"] = name
    merged["budget"] = deepcopy(entry.get("budget"))
    merged["scripts"] = deepcopy(entry.get("scripts", {}))
    merged["capabilities"] = tuple(entry.get("capabilities", ()))
    merged["overrides"] = {}
    for path, value in entry.get("overrides", {}).items():
        node = merged
        *head, leaf = path.split(".")
        for key in head:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"override {path!r} for {name!r} names no default")
        node[leaf] = value["value"] if isinstance(value, dict) and "value" in value else value
        merged["overrides"][path] = value
    return merged


__all__ = ["CORRECTNESS_METRICS", "DEFAULTS", "LATENCY_METRICS", "STATISTICS", "TARGETS",
           "for_target", "tolerances"]
