"""Runtime: the mechanisms invariant across Targets.

The op vocabulary (`ops`), the explicit computation graph and its builder
(`graph`), the VLA template a Target subclasses (`vla`), the backend registry
and plan binding (`registry`, `binding`), the one engine (`runner.ModelRunner`),
static addresses and graph capture (`cuda`), the identity every report carries
(`Identity`) and the engine protocol harnesses consume (`engine`). Nothing
here knows a model, a backend or a device.
"""
from __future__ import annotations

from .graph import BufRef, Graph, WeightRef
from .identity import PRECISION_POLICIES, Identity, git_revision
from .ops import OpSpec, Vocabulary
from .registry import Registry
from .runner import ModelRunner, Scratch
from .vla import DTYPES, STAGES, STAGE_OUTPUTS, VLA, Input

__all__ = ["BufRef", "DTYPES", "Graph", "Identity", "Input", "ModelRunner", "OpSpec",
           "PRECISION_POLICIES", "Registry", "STAGES", "STAGE_OUTPUTS", "Scratch", "VLA",
           "Vocabulary", "WeightRef", "git_revision"]
