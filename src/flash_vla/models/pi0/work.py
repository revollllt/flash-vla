"""Which Pi0 call site each op of the reference's forward belongs to (`measurement.work`).

The shared PaliGemma-with-expert rules (`models/paligemma/work.py`) and Pi0's
head, as the engine fuses it (`graph.build`): the state token; the chunk's
input projection together with the first action-time MLP layer, whose
timestep half is a per-step bias (`action_expert_action_in_proj`); and the
second MLP layer (`action_expert_action_mlp`). The prompt is fixed when the weights load
(the engine bakes it into `language_embeds`), so it is not an inference input
and its embedding folds away; the state token is computed once, however often
the reference recomputes it.

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
from .spec import ACTION_DIM, IMAGE_CHANNELS, IMAGE_SIZE, STATE_DIM

RULES = (
    *PALIGEMMA_RULES,
    CallSiteRule(r"^head\.state_proj$", None, "action_expert_state_proj"),
    CallSiteRule(r"^head\.(action_in_proj|action_time_mlp_in)$", None,
                 "action_expert_action_in_proj"),
    CallSiteRule(r"^head\.action_time_mlp_out$", None, "action_expert_action_mlp"),
)


def reference_run(shape: Mapping[str, int], prompt_tokens: int | None) -> ReferenceRun:
    """The reference at `shape`. Pi0's prompt has no padding slots
    (`prompt_len` are all valid), so `prompt_tokens` selects nothing."""
    meta = torch.device("meta")
    schema = official_schema(make_reference().parts(), prefixes=PREFIXES)
    model = load({name: torch.empty(dims, dtype=torch.bfloat16, device=meta)
                  for name, dims in schema.items()})
    views = shape["num_views"]
    images = torch.empty(views, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS, device=meta)
    image_masks = torch.ones(views, dtype=torch.bool, device=meta)
    prompt_ids = torch.zeros(shape["prompt_len"], dtype=torch.long, device=meta)
    prompt_mask = torch.ones(shape["prompt_len"], dtype=torch.bool, device=meta)
    state = torch.empty(STATE_DIM, device=meta)
    noise = torch.empty(shape["chunk"], ACTION_DIM, device=meta)

    def forward() -> tuple[torch.Tensor, ...]:
        outputs = model(images, image_masks=image_masks, prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask, state=state, noise=noise, steps=shape["steps"],
                        depth=shape["layers"])
        return (outputs.actions,)

    return ReferenceRun(model, forward, (images, image_masks, state, noise))


__all__ = ["RULES", "reference_run"]
