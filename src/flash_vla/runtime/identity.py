"""The identity every runner exposes and every report is stamped with.

Two measurements are comparable only when their identities match. The four
axes of a Target -- hardware, model revision, shape profile, precision policy
-- plus the resolved call-site plan are what this carries; environment facts
(device name, driver, framework versions, node, job) are stamped by the
harness beside it. The engine revision identifies the flash-vla source that
produced the report but is not a Target axis. No torch dependency: offline
tools read identities without a device.
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
IDENTITY_SCHEMA_VERSION = 2


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
    """Target identity, resolved plan, and the source revision that ran it."""
    target: str                                  # e.g. "hardware/nvidia/h100/pi05"
    hardware: str                                # e.g. "h100-sxm5-80gb"
    model: str                                   # e.g. "pi05"
    model_revision: str | None                   # None only when reading a v1 report
    shape: Mapping[str, int]                     # the shape profile's numbers
    plan: Mapping[str, str]                      # call site -> backend, resolved
    precision: str = "bf16"
    engine_revision: str | None = field(default_factory=git_revision)

    def __post_init__(self) -> None:
        if self.precision not in PRECISION_POLICIES:
            raise ValueError(f"unknown precision policy {self.precision!r}; "
                             f"defined: {PRECISION_POLICIES}")
        if self.model_revision in {"latest", "main", "current", "unknown"}:
            raise ValueError(f"model_revision must be immutable, got {self.model_revision!r}")

    @property
    def shape_profile(self) -> str:
        """A name derived from the shape numbers, stable across runs."""
        return "-".join(f"{k}{v}" for k, v in sorted(self.shape.items()))

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": IDENTITY_SCHEMA_VERSION,
                "target": self.target, "hardware": self.hardware, "model": self.model,
                "model_revision": self.model_revision,
                "shape_profile": self.shape_profile, "shape": dict(self.shape),
                "precision": self.precision, "plan": dict(self.plan),
                "engine_revision": self.engine_revision}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Identity":
        """Read the current schema or a v1 report without inventing its model revision."""
        version = value.get("schema_version", 1)
        if version not in (1, IDENTITY_SCHEMA_VERSION):
            raise ValueError(f"unsupported Identity schema version {version!r}")
        return cls(target=value["target"], hardware=value["hardware"], model=value["model"],
                   model_revision=value.get("model_revision"), shape=value["shape"],
                   plan=value["plan"], precision=value.get("precision", "bf16"),
                   engine_revision=(value.get("revision") if version == 1
                                    else value.get("engine_revision")))

    def same_workload(self, other: "Identity") -> bool:
        """Same TargetKey axes; plan and engine revision may differ.

        This is the relation between the legs of an A/B/A: a candidate plan
        against the reference plan on one workload.
        """
        return (self.model_revision is not None and other.model_revision is not None
                and self.hardware == other.hardware and self.model == other.model
                and self.model_revision == other.model_revision
                and dict(self.shape) == dict(other.shape)
                and self.precision == other.precision)

    def comparable(self, other: "Identity") -> bool:
        """Whether a measurement under `self` may be compared with one under `other`."""
        return self.same_workload(other) and dict(self.plan) == dict(other.plan)
