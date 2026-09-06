"""The acceptance registry: what the human defines, and the framework's conventions.

Acceptance means deployable. The human defines three things and one latency
number: the accuracy requirements (the precision policy, which oracle tiers
are required, any tolerance override), the deployment jitter bound that the
chunk latency's tail must satisfy, and each Target's budget. Everything else
here is a framework convention ratified once for every Target. No latency
objective is written here: the floor model (`benchmarks/floor.py`) is
guidance for where to look, never a target to reach.

Structure:

- `DEFAULTS` holds everything that is the same for every Target: the metric
  set, the statistics, repetition counts, the promotion bar, the run-validity
  limit on the control spread, the candidate rule and its two modes, the
  deployment jitter bound, the stop condition, the mandatory correctness
  checks, and the tolerance defaults keyed by precision policy (the tolerance
  for a `bf16` Target is a property of `bf16`, not of the model).
- `TARGETS` holds one entry per Target with only what differs: its budget,
  the scripts that implement the official-baseline tier and the interpreter
  they need, and any override of a default, each with a reason.

`for_target(name)` merges the two. No torch, no device: offline tools and the
gate read it without a GPU. Checks read their thresholds through
`tolerances(precision)[key]` rather than declaring numbers, so a change here
is a change everywhere.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Mapping

#: The latency metrics `benchmarks/latency.py` reports.
LATENCY_METRICS = ("chunk_latency", "device_latency", "host_time", "segment_latency",
                   "overhead")
#: Every latency metric reports all three.
STATISTICS = ("min", "median", "p99")
#: The correctness metrics of `eval/metrics.py`.
CORRECTNESS_METRICS = ("max_abs", "mean_abs", "rms_error", "rel_rms", "p99_abs",
                       "cosine_similarity")

#: The interpreter the official-baseline scripts run under: the OpenPI
#: environment, which carries the reference implementation. Override with
#: `OPENPI_PYTHON`; the gate records a baseline check as unavailable when the
#: interpreter does not exist.
OPENPI_PYTHON = os.environ.get(
    "OPENPI_PYTHON", "/data/user/jzou521/codes/cuda/openpi-official/.venv/bin/python")

#: The OpenPI checkpoint the Pi0 official-baseline tier compares against. Pi0's
#: adapter loads real weights (Pi0.5's builds random ones), so its script needs
#: a path; override with `OPENPI_PI0_CHECKPOINT`. The script reports the tier
#: as unavailable when the path does not exist, and the gate records it so.
OPENPI_PI0_CHECKPOINT = os.environ.get(
    "OPENPI_PI0_CHECKPOINT",
    "/data/user/jzou521/models/openpi/openpi-assets/checkpoints/pi0_libero_pytorch")

DEFAULTS: dict[str, Any] = {
    "latency": {
        "metrics": LATENCY_METRICS,
        "statistics": STATISTICS,
        "reps": 100,
        "warmup": 5,
        # Seconds of forwards before each leg's warmup, so clocks and
        # temperature settle before anything is read: the first run of a job
        # spread its control legs by 0.17 ms while the later runs of the same
        # job read 0.004 to 0.05 (job 598975).
        "soak_s": 10,
        # Below this many repetitions the 99th percentile is the maximum; it is
        # then reported as insufficient rather than as a tail.
        "p99_min_reps": 100,
        # Deployment does not lock clocks, so neither does any measurement.
        "clocks": "unlocked",
        # Deltas come only from a same-process A/B/A; the spread between the
        # two control legs is the run's minimum detectable effect.
        "deltas": "same_process_aba",
        # A performance candidate must improve the chunk `min` by at least
        # this much, in absolute terms, on top of being distinguishable from
        # the control spread. The number is the bar the Pi0.5 optimization
        # loop used for every promotion it recorded.
        "promotion_bar_ms": 0.10,
        # A run whose two control legs differ by more than this on the chunk
        # `min` cannot decide a bar-sized effect: the verdict is `blocked`,
        # rerun. Equal to the bar. On this cluster the spread of an unlocked,
        # shared node read 0.007 to 0.12 ms across five runs (jobs 598904 and
        # 598948); a limit at half the bar blocked three of them.
        "control_spread_max_ms": 0.10,
        # Two modes. `improve` is for a performance candidate: improve the
        # first statistic by more than max(bar, spread) and regress none of the
        # others by more than max(bar, that statistic's own control spread).
        # `no_regression` is for a refactor or a correctness fix: regress no
        # statistic by more than that. A plan against itself regressed its
        # median by 0.065 ms over a 0.050 ms `min` spread (job 598975): the
        # `min` spread is not the noise of the median, and a regression
        # smaller than the bar is not one the bar would have promoted.
        "candidate_rule": {
            "improve": ("chunk_latency", "min"),
            "no_regression": (("chunk_latency", "min"), ("chunk_latency", "median"),
                              ("chunk_latency", "p99")),
            "modes": ("improve", "no_regression"),
            "default_mode": "improve",
        },
    },
    # The one latency number the human sets: how far the tail may sit above
    # the floor of the same run. A closed-loop controller misses a tick on a
    # late chunk, so this is a deployment property, read on the candidate leg
    # under deployment conditions (clocks unlocked, shared node). When the
    # reference leg violates it in the same run the node is the cause and the
    # verdict is `blocked`, not `fail`.
    "deployment": {
        "metric": "chunk_latency",
        "jitter_ms": 0.5,
    },
    # When a Target's optimization stops: the deployment bound holds and no
    # candidate is left to build, or the budget is spent, or every call site
    # measures within this much of its measured ceiling in the floor report.
    "stop": {"headroom_pct": 10},
    "correctness": {
        "metrics": CORRECTNESS_METRICS,
        # Keyed by precision policy: a tolerance is a property of the rounding,
        # not of the model. bf16 values are the ones every shipped gate uses.
        # Two gates per depth: `rel_rms_max` (root-mean-square error over the
        # reference's root-mean-square, scale-free and not dominated by one
        # channel) and `cosine_min` (direction). Both must hold. The numbers
        # are calibrated, not chosen: `eval/calibrate.py` reads the natural
        # dispersion between the two promoted routes of each Target (bf16
        # reduction order is the only difference) and the gate sits at ten
        # times the worst of it; CALIBRATION_SOURCE names the run.
        "tolerances": {
            "bf16": {
                # Single step, single layer, against the in-engine reference
                # route: nothing has accumulated, so a wrong layout, mask, fold
                # or wiring shows at full size.
                # Worst shallow dispersion: Pi0 rel_rms 6.6e-3, cosine 0.999978.
                "shallow": {"rel_rms_max": 6.6e-2, "cosine_min": 0.99978},
                # Layer 0 against the official baseline: one layer, nothing
                # accumulated, so the shallow pair applies until the baseline
                # tier has calibration reports of its own.
                "layer0": {"rel_rms_max": 6.6e-2, "cosine_min": 0.99978},
                # Full depth against the official baseline on random weights:
                # rounding drift through every bf16 layer is expected.
                # Worst single-pass full-depth dispersion: Pi0.5 prefix cache
                # rel_rms 3.4e-2, cosine 0.99943.
                "deepest": {"rel_rms_max": 3.4e-1, "cosine_min": 0.9943},
                # A single layer losing more than this between consecutive
                # layers is a bug in that layer, not accumulation.
                "max_cosine_step": 0.005,
            },
        },
        # Ordered. `mode` is gate or report; `threshold` names a key of the
        # precision policy's tolerances; `requires` names a Target capability
        # without which the check is skipped and recorded as such.
        "checks": (
            {"check": "replay_determinism", "mode": "gate"},
            {"check": "finiteness", "mode": "gate"},
            {"check": "in_engine_shallow", "oracle": "in_engine_reference",
             "config": {"steps": 1, "layers": 1}, "threshold": "shallow",
             "mode": "gate"},
            {"check": "in_engine_deep", "oracle": "in_engine_reference",
             "config": {"steps": 1, "layers": "full"}, "threshold": "shallow",
             "mode": "report"},
            {"check": "in_engine_multistep", "oracle": "in_engine_reference",
             "config": {"steps": "full", "layers": "full"}, "mode": "report"},
            {"check": "baseline_layer0", "oracle": "official_baseline",
             "threshold": "layer0", "mode": "gate", "requires": "baseline_adapter"},
            {"check": "baseline_depth", "oracle": "official_baseline",
             "threshold": "deepest", "step": "max_cosine_step", "mode": "report",
             "requires": "baseline_adapter"},
        ),
    },
    # Out of scope this phase; the slot exists so a precision policy other than
    # bf16 cannot be promoted while it is empty.
    "policy_quality": {"suite": None, "threshold": None},
}

#: Per-Target entries: budget, the official-baseline scripts and their
#: interpreter, and overrides with reasons. Nothing that the model spec or the
#: runner already holds is restated here.
#:
#: `budget` is counted per kernel-design task (one optimization lane on this
#: Target: `candidates` built, `non_improving` in a row, GPU `jobs`), not per
#: Target: every task on the Target starts with it, and a task's contract may
#: narrow it, never widen it.
TARGETS: dict[str, dict[str, Any]] = {
    "hardware/nvidia/h100/pi05": {
        "budget": {"candidates": 6, "non_improving": 3, "jobs": 12},
        "scripts": {
            "in_engine_reference": "eval.correctness",
            "official_baseline": ("eval.pi05.reference",),
        },
        "baseline_python": OPENPI_PYTHON,
        "capabilities": ("baseline_adapter",),
        "overrides": {},
    },
    "hardware/nvidia/h100/pi0": {
        "budget": {"candidates": 6, "non_improving": 3, "jobs": 12},
        "scripts": {
            "in_engine_reference": "eval.correctness",
            "official_baseline": ("eval.pi0.reference",),
        },
        "baseline_python": OPENPI_PYTHON,
        "capabilities": ("baseline_adapter",),
        "overrides": {},
    },
}


#: The run whose reports calibrated the tolerances above (`eval/calibrate.py`).
CALIBRATION_SOURCE = ("job 599011 (ACD1-50, tree e7cd680): eval.correctness shipped vs "
                      "reference on both Targets, seeds 0 and 1 at 1 step x 1 layer, "
                      "seed 0 at 1 x 18; factor 10")


def tolerances(precision: str = "bf16") -> Mapping[str, Any]:
    """The correctness tolerances for one precision policy: per depth key a
    `{"rel_rms_max", "cosine_min"}` pair, plus `max_cosine_step`."""
    try:
        return dict(DEFAULTS["correctness"]["tolerances"][precision])
    except KeyError:
        raise KeyError(f"no tolerances defined for precision policy {precision!r}; "
                       f"defined: {sorted(DEFAULTS['correctness']['tolerances'])}") from None


def for_target(name: str) -> dict[str, Any]:
    """The framework defaults merged with `name`'s entry.

    A Target without an entry gets every default and no budget; it can run
    every check but cannot enter the gate. An override replaces one dotted
    path in the defaults (`"deployment.jitter_ms": {"value": 1.0, "reason":
    ...}`) and must carry a reason in the entry, which is kept beside the
    merged value.
    """
    merged = deepcopy(DEFAULTS)
    entry = TARGETS.get(name, {})
    merged["target"] = name
    merged["budget"] = deepcopy(entry.get("budget"))
    merged["scripts"] = deepcopy(entry.get("scripts", {}))
    merged["baseline_python"] = entry.get("baseline_python")
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


__all__ = ["CALIBRATION_SOURCE", "CORRECTNESS_METRICS", "DEFAULTS", "LATENCY_METRICS",
           "OPENPI_PI0_CHECKPOINT", "OPENPI_PYTHON", "STATISTICS", "TARGETS", "for_target",
           "tolerances"]
