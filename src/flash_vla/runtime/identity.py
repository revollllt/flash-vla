"""The identity every engine exposes and every report is stamped with.

Two measurements are comparable only when their identities match. The four
axes of a Target -- hardware, model revision, shape profile, precision policy
-- plus the resolved call-site plan are what this carries; environment facts
(device name, driver, framework versions, node, job) are stamped by the
harness beside it, and the git revision is read here so no harness restates
it. No torch dependency: offline tools read identities without a device.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: The precision policies defined so far. A policy names the dtype of every
#: tensor class; `bf16` is bf16 weights, activations and KV cache with fp32
#: accumulation. A policy other than `bf16` cannot be promoted without a
#: policy-quality gate.
PRECISION_POLICIES = ("bf16",)


def git_revision(start: Path | str | None = None) -> str | None:
    """The short HEAD revision of the checkout containing `start`, if any."""
    root = Path(start or __file__).resolve()
    try:
        out = subprocess.run(["git", "-C", str(root.parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass(frozen=True)
class Identity:
    """Target identity: the four axes, the shape numbers and the plan."""
    target: str                                  # e.g. "hardware/nvidia/h100/pi05"
    hardware: str                                # e.g. "h100-sxm5-80gb"
    model: str                                   # e.g. "pi05"
    shape: Mapping[str, int]                     # the shape profile's numbers
    plan: Mapping[str, str]                      # call site -> backend, resolved
    precision: str = "bf16"
    #: Target-local table options that change what runs without changing the
    #: route (Pi0's fused overlay flag). Part of comparability.
    options: Mapping[str, Any] = field(default_factory=dict)
    revision: str | None = field(default_factory=git_revision)

    def __post_init__(self) -> None:
        if self.precision not in PRECISION_POLICIES:
            raise ValueError(f"unknown precision policy {self.precision!r}; "
                             f"defined: {PRECISION_POLICIES}")

    @property
    def shape_profile(self) -> str:
        """A name derived from the shape numbers, stable across runs."""
        return "-".join(f"{k}{v}" for k, v in sorted(self.shape.items()))

    def as_dict(self) -> dict[str, Any]:
        return {"target": self.target, "hardware": self.hardware, "model": self.model,
                "shape_profile": self.shape_profile, "shape": dict(self.shape),
                "precision": self.precision, "plan": dict(self.plan),
                "options": dict(self.options), "revision": self.revision}

    def comparable(self, other: "Identity") -> bool:
        """Whether a measurement under `self` may be compared with one under `other`."""
        return (self.target == other.target and self.hardware == other.hardware
                and self.model == other.model and dict(self.shape) == dict(other.shape)
                and self.precision == other.precision and dict(self.plan) == dict(other.plan)
                and dict(self.options) == dict(other.options))
