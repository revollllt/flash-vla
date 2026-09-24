"""What an observation was measured under, beside the architecture it measured.

`MeasurementContext` names the weights and fixture an engine ran
(`ModelRunner.measurement_context`) and the environment it ran in. Two
observations compare only when their `segment_key`s match: the same immutable
assets on the same environment. Hostname, job and timestamp are provenance
and never part of the comparison. No GPU imports: report readers use this
offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from flash_vla.provenance import canonical_digest


@dataclass(frozen=True)
class MeasurementContext:
    """Checkpoint/fixture identity and the environment of an observation.

    `weights` and `fixture` are the report forms of `WeightsProvenance` and
    `FixtureProvenance`; saved reports may carry a `None` digest.
    """
    weights: Mapping[str, str | None]
    fixture: Mapping[str, str | None]
    environment: Mapping[str, object]
    hostname: str | None = None
    slurm_job_id: str | None = None
    timestamp: float | None = None
    reference_provenance: Mapping[str, object] = field(default_factory=dict)

    @property
    def context_id(self) -> str:
        # Locations and observation provenance do not identify immutable assets.
        return canonical_digest({
            "weights": {key: self.weights[key] for key in ("checkpoint_id", "checkpoint_digest")},
            "fixture": {key: self.fixture[key] for key in ("id", "digest")},
        })

    @property
    def segment_key(self) -> dict[str, object]:
        return {"context_id": self.context_id, "environment": dict(self.environment)}

    @property
    def fingerprint(self) -> str:
        return canonical_digest(self.segment_key)

    def as_dict(self) -> dict[str, object]:
        return dict(weights=dict(self.weights), fixture=dict(self.fixture),
                    environment=dict(self.environment), hostname=self.hostname,
                    slurm_job_id=self.slurm_job_id, timestamp=self.timestamp,
                    reference_provenance=dict(self.reference_provenance))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MeasurementContext:
        return cls(**value)


__all__ = ["MeasurementContext"]
