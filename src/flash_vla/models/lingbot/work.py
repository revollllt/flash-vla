"""Which LingBot call site each op of the reference's forward belongs to (`measurement.work`).

LingBot's graph runs each stage as one monolithic call site, so the rules
follow the reference's top-level parts: the vision tower, the language model
(with its token embedding), and the expert with the head around it. Ops the
reference runs outside a part belong to the call site before them.

`reference_run` builds the reference on the meta device, in upstream's
inference dtypes, at the workload's shape.
"""
from __future__ import annotations

from typing import Mapping

import torch

from flash_vla.runtime.work import CallSiteRule, ReferenceRun

from ..official import official_schema
from .reference import PREFIXES, load, make_reference
from .spec import ACTION_DIM, GRID, LANGUAGE_SLOTS, PATCH_ROWS_PER_VIEW, PATCH_WIDTH, STATE_DIM

RULES = (
    CallSiteRule(r"^vision(\.|$)", None, "lingbot_vision"),
    CallSiteRule(r"^backbone(\.|$)", None, "lingbot_prefix"),
    CallSiteRule(r"^(expert|head)(\.|$)", None, "lingbot_action"),
)


def reference_run(shape: Mapping[str, int], prompt_tokens: int | None) -> ReferenceRun:
    """The reference at `shape`. The engine computes every one of the 72 language
    slots, masked or not, so the trace does too and `prompt_tokens` selects nothing."""
    meta = torch.device("meta")
    schema = official_schema(make_reference().parts(), prefixes=PREFIXES)
    model = load({name: torch.empty(dims, dtype=torch.bfloat16, device=meta)
                  for name, dims in schema.items()})
    views = shape["views"]
    pixels = torch.empty(views, PATCH_ROWS_PER_VIEW, PATCH_WIDTH, dtype=torch.bfloat16, device=meta)
    image_masks = torch.ones(views, dtype=torch.bool, device=meta)
    language_tokens = torch.zeros(1, LANGUAGE_SLOTS, dtype=torch.long, device=meta)
    language_masks = torch.ones(1, LANGUAGE_SLOTS, dtype=torch.bool, device=meta)
    state = torch.empty(1, STATE_DIM, dtype=torch.bfloat16, device=meta)
    noise = torch.empty(1, shape["chunk"], ACTION_DIM, dtype=torch.bfloat16, device=meta)

    def forward() -> tuple[torch.Tensor, ...]:
        outputs = model(pixels, grid=GRID, image_masks=image_masks, language_tokens=language_tokens,
                        language_masks=language_masks, state=state, noise=noise,
                        steps=shape["steps"], depth=shape["layers"])
        return (outputs.actions,)

    return ReferenceRun(model, forward,
                        (pixels, image_masks, language_tokens, language_masks, state, noise))


__all__ = ["RULES", "reference_run"]
