"""Device-independent GR00T reference stages, on the runner's buffers and weights.

Each op's weights are the runner's, bound to the reference's part without a
copy on the op's first call (`official.bind_parts`). The vision tables, which
the reference builds from host-side lists, are built once, during warmup,
into runner workspace, so the captured vision stage only reads them.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from flash_vla.models.groot_n17.ops import CALL_SITES as NAMES, WEIGHTS
from flash_vla.models.groot_n17.reference import PREFIXES, VisionTables, make_reference
from flash_vla.models.official import bind_parts
from flash_vla.models.groot_n17.spec import STEPS
from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch


def make_wrappers(scratch: Scratch, selected_names: frozenset[str] | None = None) -> dict[str, Wrapper]:
    reference = make_reference()
    bound: set[str] = set()
    #: Per view count and patch grid, the tables staged into workspace.
    tables: dict[tuple[int, int, int], VisionTables] = {}

    def bind_once(part: str, tensors: tuple[torch.Tensor, ...]) -> None:
        if part in bound:
            return
        bind_parts({part: reference.parts()[part]}, prefixes=PREFIXES,
                   weights=dict(zip(WEIGHTS[part], tensors)), precision="bfloat16")
        bound.add(part)

    def staged_tables(views: int, rows: int, columns: int) -> VisionTables:
        key = (views, rows, columns)
        if key in tables:
            return tables[key]
        built = reference.vision.tables(((1, rows, columns),) * views)
        tables[key] = replace(built, **{
            role: scratch(f"groot_vision_{role}", table.shape, table.dtype, table.device).copy_(table)
            for role, table in (("position", built.position), ("cos", built.cos), ("sin", built.sin))})
        return tables[key]

    @torch.no_grad()
    def vision(pixels: torch.Tensor, out: torch.Tensor, deepstack: torch.Tensor,
               grid_rows: int, grid_columns: int,
               *weights: torch.Tensor) -> None:
        bind_once("vision", weights)
        merged, features = reference.vision(
            pixels, tables=staged_tables(pixels.shape[0] // (grid_rows * grid_columns),
                                        grid_rows, grid_columns))
        out.copy_(merged)
        for slot, feature in enumerate(features):
            deepstack[slot].copy_(feature)

    @torch.no_grad()
    def backbone(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                 image_indices: torch.Tensor, vision: torch.Tensor, deepstack: torch.Tensor,
                 out: torch.Tensor, *weights: torch.Tensor) -> None:
        bind_once("backbone", weights)
        out.copy_(reference.backbone(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                     image_indices=image_indices, vision=vision,
                                     deepstack=deepstack.unbind(0)))

    @torch.no_grad()
    def action(backbone_features: torch.Tensor, state: torch.Tensor, noise: torch.Tensor,
               embodiment: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               out: torch.Tensor, velocity: torch.Tensor, *weights: torch.Tensor) -> None:
        bind_once("action", weights)
        actions, first_velocity = reference.action(backbone_features, input_ids=input_ids,
                                                   attention_mask=attention_mask, state=state,
                                                   embodiment_id=embodiment, noise=noise, steps=STEPS)
        out.copy_(actions)
        velocity.copy_(first_velocity)

    wrappers = dict(zip(NAMES, (vision, backbone, action)))
    return {name: wrappers[name] for name in (selected_names or NAMES)}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)
