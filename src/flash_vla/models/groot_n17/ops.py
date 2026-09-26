"""The GR00T graph's call sites: vision, backbone and action head as one op each.

Each stage takes every weight of its module as a numbered parameter
(`PARAMS`), bound to the checkpoint names of `WEIGHTS`.
"""
from __future__ import annotations

from flash_vla.runtime.ops import OpSpec

from .reference import PREFIXES
from .weights import weight_shapes

#: The three monolithic call sites of the GR00T graph.
CALL_SITES = ("groot_vision", "groot_backbone", "groot_action")
#: Checkpoint weight names of each stage, in the order its op takes them.
WEIGHTS = {part: tuple(name for name in weight_shapes() if name.startswith(prefix))
           for part, prefix in PREFIXES.items()}
#: The positional weight parameter names of each stage's op.
PARAMS = {part: tuple(f"w{i}" for i in range(len(names))) for part, names in WEIGHTS.items()}


OPS = (
    OpSpec("groot_vision", ("pixels", "out", "deepstack") + PARAMS["vision"],
           outputs=("out", "deepstack"), weights=PARAMS["vision"]),
    OpSpec("groot_backbone", ("input_ids", "attention_mask", "position_ids", "image_indices",
           "vision", "deepstack", "out") + PARAMS["backbone"], outputs=("out",),
           weights=PARAMS["backbone"]),
    OpSpec("groot_action", ("backbone", "state", "noise", "embodiment", "input_ids",
           "attention_mask", "out", "velocity") + PARAMS["action"], outputs=("out", "velocity"),
           weights=PARAMS["action"]),
)

__all__ = ["CALL_SITES", "OPS", "PARAMS", "WEIGHTS"]
