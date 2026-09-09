"""Architecture identity, execution policy and measurement provenance.

No GPU imports: report readers and Campaign discovery use these types offline.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

PRECISION_POLICIES = ("bf16",)
IDENTITY_SCHEMA_VERSION = 3


def canonical_digest(value: Any) -> str:
    """Digest of small semantic metadata, never source files or weight values."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def inference_signature(*, architecture: Mapping, parameter_shapes: Mapping,
                        weight_layout: Any, io_contract: Mapping,
                        control_flow: Mapping) -> str:
    """Compatibility signature of the model ABI before candidate transformations."""
    return canonical_digest(dict(architecture=architecture, parameter_shapes=parameter_shapes,
                                 weight_layout=weight_layout, io_contract=io_contract,
                                 control_flow=control_flow))


def validate_weight_schema(actual: Mapping, expected: Mapping) -> None:
    """Check names and shapes before allocating or loading any runtime weight."""
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = {name: (tuple(actual[name]), tuple(expected[name]))
                  for name in set(actual) & set(expected)
                  if tuple(actual[name]) != tuple(expected[name])}
    if missing or extra or mismatched:
        raise ValueError("inference signature mismatch: weight ABI differs; "
                         f"missing={missing}, extra={extra}, shapes={mismatched}; "
                         "resolve a compatible Target/model revision")


def git_revision(start: Path | str | None = None) -> str | None:
    """The full HEAD revision of a clean checkout containing start, if any."""
    root = Path(start or __file__).resolve()
    try:
        status = subprocess.run(["git", "-C", str(root.parent), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=5)
        if status.returncode != 0 or status.stdout:
            return None
        out = subprocess.run(["git", "-C", str(root.parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


@dataclass(frozen=True)
class ExecutionVariant:
    """High-level quality/performance choices; kernel plans are independent."""
    quantization: Mapping[str, Any] = field(default_factory=lambda: {"mode": "bf16"})
    cache: Mapping[str, Any] = field(default_factory=lambda: {"mode": "none"})

    def __post_init__(self) -> None:
        for name in ("quantization", "cache"):
            value = dict(getattr(self, name))
            if not isinstance(value["mode"], str) or not value["mode"]:
                raise ValueError(f"{name} needs a nonempty mode")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict:
        return {"quantization": dict(self.quantization), "cache": dict(self.cache)}

    @classmethod
    def from_dict(cls, value: Mapping) -> ExecutionVariant:
        return cls(**value)


@dataclass(frozen=True)
class MeasurementContext:
    """Checkpoint/fixture identity and the environment of an observation."""
    weights: Mapping[str, Any]
    fixture: Mapping[str, Any]
    environment: Mapping[str, Any]
    hostname: str | None = None
    slurm_job_id: str | None = None
    timestamp: float | None = None
    reference_provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def context_id(self) -> str:
        # Locations and observation provenance do not identify immutable assets.
        return canonical_digest({
            "weights": {key: self.weights[key] for key in ("checkpoint_id", "checkpoint_digest")},
            "fixture": {key: self.fixture[key] for key in ("id", "digest")},
        })

    @property
    def segment_key(self) -> dict:
        return {"context_id": self.context_id, "environment": dict(self.environment)}

    @property
    def fingerprint(self) -> str:
        return canonical_digest(self.segment_key)

    def as_dict(self) -> dict:
        return dict(weights=dict(self.weights), fixture=dict(self.fixture),
                    environment=dict(self.environment), hostname=self.hostname,
                    slurm_job_id=self.slurm_job_id, timestamp=self.timestamp,
                    reference_provenance=dict(self.reference_provenance))

    @classmethod
    def from_dict(cls, value: Mapping) -> MeasurementContext:
        return cls(**value)


@dataclass(frozen=True)
class Identity:
    """Target architecture and shape, execution strategy, and candidate provenance.

    A legacy value remains legacy until explicitly migrated; no architecture is
    inferred from its checkpoint-based model_revision.
    """
    target: str
    hardware: str
    model: str
    model_revision: str | None
    shape: Mapping[str, int]
    plan: Mapping[str, str]
    precision: str = "bf16"
    engine_revision: str | None = field(default_factory=git_revision)
    inference_signature: str | None = None
    execution_variant: ExecutionVariant | Mapping | None = None
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        version = self.schema_version
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported Identity schema version {version!r}")
        object.__setattr__(self, "schema_version", version)
        variant = self.execution_variant
        if variant is None:
            variant = ExecutionVariant(quantization={"mode": self.precision})
        elif not isinstance(variant, ExecutionVariant):
            variant = ExecutionVariant.from_dict(variant)
        object.__setattr__(self, "execution_variant", variant)
        object.__setattr__(self, "precision", variant.quantization["mode"])
        if version == 3 and (not self.inference_signature or not self.model_revision):
            raise ValueError("Identity v3 requires architecture revision and inference signature")
        revision = self.model_revision
        if revision is not None and (revision != revision.strip() or not revision or
                                     revision in {"latest", "main", "current", "unknown"}):
            raise ValueError(f"model_revision must be explicit, got {revision!r}")

    @property
    def shape_profile(self) -> str:
        return "-".join(f"{key}{value}" for key, value in sorted(self.shape.items()))

    def as_dict(self) -> dict[str, Any]:
        value = dict(schema_version=self.schema_version, target=self.target,
                     hardware=self.hardware, model=self.model, shape_profile=self.shape_profile,
                     shape=dict(self.shape), plan=dict(self.plan))
        if self.schema_version == 1:
            return dict(value, precision=self.precision, revision=self.engine_revision)
        value.update(model_revision=self.model_revision, engine_revision=self.engine_revision)
        if self.schema_version == 2:
            return dict(value, precision=self.precision)
        return dict(value, inference_signature=self.inference_signature,
                    execution_variant=self.execution_variant.as_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Identity:
        version = value.get("schema_version", 1)
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported Identity schema version {version!r}")
        return cls(target=value["target"], hardware=value["hardware"], model=value["model"],
                   model_revision=None if version == 1 else value["model_revision"],
                   shape=value["shape"], plan=value["plan"],
                   precision=value.get("precision", "bf16"),
                   engine_revision=value.get("revision") if version == 1 else value.get("engine_revision"),
                   inference_signature=value["inference_signature"] if version == 3 else None,
                   execution_variant=value["execution_variant"] if version == 3 else None,
                   schema_version=version)

    def same_workload(self, other: Identity) -> bool:
        return (self.model_revision is not None and other.model_revision is not None
                and self.schema_version == other.schema_version
                and self.hardware == other.hardware and self.model == other.model
                and self.model_revision == other.model_revision
                and self.inference_signature == other.inference_signature
                and dict(self.shape) == dict(other.shape)
                and self.execution_variant == other.execution_variant)

    def comparable(self, other: Identity) -> bool:
        return self.same_workload(other) and dict(self.plan) == dict(other.plan)
