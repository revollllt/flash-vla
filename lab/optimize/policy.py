"""Historical Campaign qualification policy, used only by lab.optimize.

Numerical tolerances are owned by eval.tolerances. Daily evaluation and timing
never import this budget, promotion or publication policy.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from benchmarks.config import LATENCY_DEFAULTS, LATENCY_METRICS, STATISTICS
from eval.tolerances import TOLERANCES, tolerances

#: The correctness metrics of `eval/metrics.py`.
CORRECTNESS_METRICS = ("max_abs", "mean_abs", "rms_error", "rel_rms", "p99_abs",
                       "cosine_similarity")

#: Official runtimes are machine configuration, never repository-local defaults.
#: Unset or missing interpreters make the official tier unavailable.
OPENPI_PYTHON = os.environ.get("OPENPI_PYTHON")
LINGBOT_PYTHON = os.environ.get("LINGBOT_PYTHON")

#: Pi0 official weights require both a location and an explicit immutable ID.
#: The legacy environment variable name denotes checkpoint provenance, not
#: architecture model_revision. Neither value is inferred from a machine path.
OPENPI_PI0_CHECKPOINT = os.environ.get("OPENPI_PI0_CHECKPOINT")
OPENPI_PI0_MODEL_REVISION = os.environ.get("OPENPI_PI0_MODEL_REVISION")

DEFAULTS: dict[str, Any] = {
    "latency": {
        **LATENCY_DEFAULTS,
        # The explicit legacy gate requires the chunk `min` to improve by at least
        # this much, in absolute terms, on top of being distinguishable from
        # the control spread. The number is the bar the Pi0.5 optimization
        # loop used for every promotion it recorded.
        "promotion_bar_ms": 0.10,
        # Maximum control spread for explicit historical qualification.
        "control_spread_max_ms": 0.10,
        # Two modes. `improve` is for a performance candidate: improve the
        # first statistic by more than max(bar, spread) and regress none of the
        # others by more than max(bar, that statistic's own control spread).
        # `no_regression` is for a refactor or a correctness fix: regress no
        # statistic by more than that.
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
        "tolerances": TOLERANCES,
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
    "hardware/nvidia/h100/lingbot_vla": {
        "budget": {"candidates": 6, "non_improving": 3, "jobs": 12},
        "scripts": {
            "in_engine_reference": "eval.correctness",
            "official_baseline": ("eval.lingbot.parity",),
        },
        "baseline_python": LINGBOT_PYTHON,
        "capabilities": ("baseline_adapter",),
        "overrides": {},
    },
    "hardware/nvidia/h100/pi05": {
        "budget": {"candidates": 6, "non_improving": 3, "jobs": 12},
        "scripts": {
            "in_engine_reference": "eval.correctness",
            "official_baseline": ("eval.pi05.reference",),
        },
        # The expert wiring check uses one flow step; its identity must say so.
        "baseline_report_shapes": {
            "eval.pi05.reference": {"llm_backbone": {}, "action_expert": {"steps": 1}},
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
    merged["baseline_report_shapes"] = deepcopy(entry.get("baseline_report_shapes", {}))
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


def validate_baseline_workloads(script, identities, expected, stages=()) -> None:
    """Match every registered adapter stage, allowing only its declared depth."""
    from dataclasses import replace
    from flash_vla.runtime.identity import Identity

    shapes = for_target(expected.target)["baseline_report_shapes"].get(script)
    if not identities:
        raise ValueError("official adapter emitted no workload identities")
    if shapes is not None:
        if (len(stages) != len(identities) or len(stages) != len(shapes)
                or set(stages) != set(shapes)):
            raise ValueError("official adapter stage coverage differs from acceptance")
        expected_reports = [replace(expected, shape={**expected.shape, **shapes[stage]})
                            for stage in stages]
    else:
        expected_reports = [expected] * len(identities)
    for value, wanted in zip(identities, expected_reports):
        if not wanted.same_workload(Identity.from_dict(value)):
            raise ValueError("official correctness adapter checked another workload")


__all__ = ["CORRECTNESS_METRICS", "DEFAULTS", "LATENCY_METRICS",
           "OPENPI_PI0_CHECKPOINT", "OPENPI_PYTHON", "STATISTICS", "TARGETS", "for_target",
           "tolerances"]
