"""GR00T N1.7 LIBERO on the RTX 5090, BF16: the PyTorch reference stages.

The model is `flash_vla.models.groot_n17`; both plans route every stage to the
plain-torch backend until kernels for this machine exist.
"""
from __future__ import annotations

from flash_vla.models.groot_n17.definition import GrootModel
from flash_vla.models.groot_n17.weights import CHECKPOINT_ID, FIXTURE_ID
from flash_vla.runtime.vla import Target

from .backends import REGISTRY

TARGET = Target(
    name="hardware/nvidia/rtx5090/groot_n17",
    hardware="rtx5090-32gb",
    model=GrootModel(),
    registry=REGISTRY,
    plan={},
    reference_plan={},
    assets={"checkpoint": CHECKPOINT_ID, "fixture": FIXTURE_ID},
)

__all__ = ["TARGET"]
