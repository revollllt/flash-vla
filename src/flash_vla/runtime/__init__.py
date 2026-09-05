"""Runtime: the mechanisms invariant across Targets.

Static addresses (`cuda.StaticArena`, `cuda.ScratchPool`), graph segments and
their lifecycle (`cuda.Program`), the identity every report carries
(`Identity`), plan validation against backend-declared route constraints
(`binding`), and the engine protocol generic harnesses consume (`Engine`).
Nothing here knows a model, a backend or a device.
"""
from __future__ import annotations

from .identity import PRECISION_POLICIES, Identity, git_revision

__all__ = ["PRECISION_POLICIES", "Identity", "git_revision"]
