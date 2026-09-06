"""SigLIP vision encoder kernels and backends for H100.

A device-level component package
([device component packages](../../../../../../.agents/notes/implemented/architecture/2026-09-07-device-component-packages.md)):
it owns the kernels, reference mirrors and backend factories of the vision
tower that Pi0.5 and Pi0 run at identical shapes, written once. It imports
`models/`, `runtime/` and the shared tile library, and never a Target; each
Target registers a backend from here under a name of its own and keeps every
routing decision.

`geometry` is the single source of the tower's dimensions for everything here.
"""
from __future__ import annotations

from . import geometry

__all__ = ["geometry"]
