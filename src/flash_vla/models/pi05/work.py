"""Which Pi0.5 call site each op of the reference's forward belongs to (`measurement.work`).

The shared PaliGemma-with-expert rules (`models/paligemma/work.py`) and Pi0.5's
head: the chunk's input projection. The timestep MLP and every adaptive-norm
modulation depend only on the flow schedule, so they are constant and fold
away, as the engine folds them into its weights (`weights.fold`). A Target
that renames call sites maps them through its `call_site_aliases`.

`reference_run` builds the reference on the meta device, in upstream's
inference dtypes, at a workload's shape.
"""
from __future__ import annotations

from typing import Mapping

import torch

from flash_vla.runtime.work import CallSiteRule, ReferenceRun

from ..official import official_schema
from ..paligemma.work import RULES as PALIGEMMA_RULES
from .reference import PREFIXES, load, make_reference
from .spec import ACTION_DIM, IMAGE_CHANNELS, IMAGE_SIZE

RULES = (
    *PALIGEMMA_RULES,
    CallSiteRule(r"^head\.action_in_proj$", None, "action_expert_action_in_proj"),
)


def reference_run(shape: Mapping[str, int], extent: int | None) -> ReferenceRun:
    """The reference at `shape`, prompted with `extent` valid tokens (the replay
    axis, `prompt_tokens`), or with every one of the `prompt_len` slots (the
    physical layout) when `None`.
    The prompt is an inference input: it carries the state."""
    meta = torch.device("meta")
    schema = official_schema(make_reference().parts(), prefixes=PREFIXES)
    model = load({name: torch.empty(dims, dtype=torch.bfloat16, device=meta)
                  for name, dims in schema.items()})
    prompt = shape["prompt_len"] if extent is None else extent
    views = shape["num_views"]
    images = torch.empty(views, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS, device=meta)
    image_masks = torch.ones(views, dtype=torch.bool, device=meta)
    prompt_ids = torch.zeros(prompt, dtype=torch.long, device=meta)
    prompt_mask = torch.ones(prompt, dtype=torch.bool, device=meta)
    noise = torch.empty(shape["chunk"], ACTION_DIM, device=meta)

    def forward() -> tuple[torch.Tensor, ...]:
        outputs = model(images, image_masks=image_masks, prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask, noise=noise, steps=shape["steps"],
                        depth=shape["layers"])
        return (outputs.actions,)

    return ReferenceRun(model, forward, (images, image_masks, prompt_ids, prompt_mask, noise))


__all__ = ["RULES", "reference_run"]
