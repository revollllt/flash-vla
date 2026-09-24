"""Provenance values that name what ran, computed without a device.

`WeightsProvenance`, `FixtureProvenance` and `ImplementationProvenance` name the
weights a runner holds, the inputs a measurement feeds it and the checkout its
implementation came from; a runner receives them at construction and never
changes them. `canonical_digest` names small semantic metadata -- a fixture
description, an architecture contract -- and never source files or weight
values. `git_revision` names the clean source checkout an engine was built
from. The runtime never computes any of these on its own: an entry point that
builds an engine passes them in, so constructing a runner does not depend on a
checkout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess


@dataclass(frozen=True, kw_only=True)
class WeightsProvenance:
    """The weights a runner holds: a logical checkpoint ID and the digest of its
    immutable manifest (for synthetic weights, the seeded revision twice)."""
    checkpoint_id: str
    checkpoint_digest: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class FixtureProvenance:
    """The inputs a measurement feeds a runner: a logical fixture ID and its digest."""
    id: str
    digest: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ImplementationProvenance:
    """The clean checkout a Target's backends were loaded from, when it is not
    this one (`measurement.source_checkout`), and the controller checkout that loaded them."""
    revision: str
    checkout: str
    controller_revision: str
    target_package: str
    module: str
    scope: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def canonical_digest(value: object) -> str:
    """`sha256:` digest of JSON-serializable metadata, independent of key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def git_revision(start: Path | str | None = None) -> str | None:
    """The full HEAD revision of the clean checkout containing `start`, if any.

    `None` when `start` is not in a git checkout, git is unavailable, or the
    checkout has uncommitted changes: a dirty tree names no revision.
    """
    directory = Path(start or __file__).resolve().parent
    try:
        status = subprocess.run(["git", "-C", str(directory), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=5)
        if status.returncode != 0 or status.stdout:
            return None
        head = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    revision = head.stdout.strip()
    return revision if head.returncode == 0 and revision else None


__all__ = ["FixtureProvenance", "ImplementationProvenance", "WeightsProvenance", "canonical_digest",
           "git_revision"]
