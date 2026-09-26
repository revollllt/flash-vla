"""Which GR00T call site each op of the reference's forward belongs to (`measurement.work`).

GR00T's graph runs each stage as one monolithic call site, so the rules follow
the reference's top-level parts: the Qwen3-VL vision tower, its text model, and
the DiT action head. Ops the reference runs outside a part belong to the call
site before them.

`reference_run` builds the reference on the meta device, in upstream's
inference dtypes, at a workload's shape.
"""
from __future__ import annotations

from typing import Mapping

import torch

from flash_vla.runtime.work import CallSiteRule, ReferenceRun

from ..official import official_schema
from .reference import PREFIXES, load, make_reference
from .spec import ACTION_DIM, CHUNK, PATCH_WIDTH, PATCHES_PER_VIEW, STATE_DIM, VIEW_GRID

RULES = (
    CallSiteRule(r"^vision(\.|$)", None, "groot_vision"),
    CallSiteRule(r"^backbone(\.|$)", None, "groot_backbone"),
    CallSiteRule(r"^action(\.|$)", None, "groot_action"),
)


def reference_run(shape: Mapping[str, int], prompt_tokens: int | None) -> ReferenceRun:
    """The reference at `shape`. GR00T's text sequence has no padding slots (its
    length is the workload's), so `prompt_tokens` selects nothing."""
    meta = torch.device("meta")
    schema = official_schema(make_reference().parts(), prefixes=PREFIXES)
    model = load({name: torch.empty(dims, dtype=torch.bfloat16, device=meta)
                  for name, dims in schema.items()})
    views, length = shape["views"], shape["sequence_length"]
    pixels = torch.empty(views * PATCHES_PER_VIEW, PATCH_WIDTH, dtype=torch.bfloat16, device=meta)
    input_ids = torch.zeros(1, length, dtype=torch.long, device=meta)
    attention_mask = torch.ones(1, length, dtype=torch.long, device=meta)
    position_ids = torch.zeros(3, 1, length, dtype=torch.long, device=meta)
    image_indices = torch.zeros(shape["visual_tokens"], dtype=torch.long, device=meta)
    state = torch.empty(1, 1, STATE_DIM, dtype=torch.bfloat16, device=meta)
    embodiment = torch.zeros(1, dtype=torch.long, device=meta)
    noise = torch.empty(1, CHUNK, ACTION_DIM, dtype=torch.bfloat16, device=meta)

    def forward() -> tuple[torch.Tensor, ...]:
        outputs = model(pixels, grid=(VIEW_GRID,) * views, input_ids=input_ids,
                        attention_mask=attention_mask, position_ids=position_ids,
                        image_indices=image_indices, state=state, embodiment_id=embodiment,
                        noise=noise, steps=shape["steps"])
        return (outputs.actions,)

    return ReferenceRun(model, forward, (pixels, input_ids, attention_mask, position_ids,
                                         image_indices, state, embodiment, noise))


__all__ = ["RULES", "reference_run"]
