"""Runtime: the mechanisms invariant across Targets.

The op vocabulary (`ops`), the explicit computation graph and its builder
(`graph`), the VLA template a Target subclasses (`vla`), the backend registry
and plan binding (`registry`, `binding`), the workspace allocator backends
receive (`workspace`), the one engine (`runner.ModelRunner`),
static addresses and graph capture (`cuda`), the identity every report carries
(`Identity`) and the engine protocol harnesses consume (`engine`). Nothing
here knows a model, a backend or a device.
"""
from __future__ import annotations

from .graph import BufRef, Graph, WeightRef
from .identity import IDENTITY_SCHEMA_VERSION, PRECISION_POLICIES, Identity
from .ops import OpSpec, Vocabulary
from .registry import Backend, GraphContract, Registry
from .runner import ModelRunner
from .vla import DTYPES, STAGES, STAGE_OUTPUTS, VLA, Input, QuantizationRecipe
from .workspace import Scratch

__all__ = ["Backend", "BufRef", "DTYPES", "Graph", "GraphContract", "IDENTITY_SCHEMA_VERSION",
           "Identity", "Input", "ModelRunner", "OpSpec", "PRECISION_POLICIES",
           "QuantizationRecipe", "Registry", "STAGES", "STAGE_OUTPUTS", "Scratch", "VLA",
           "Vocabulary", "WeightRef"]
