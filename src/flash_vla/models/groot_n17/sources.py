"""Where a GR00T N1.7 runner's weights and fixture come from, and the provenance each carries.

The checkpoint and the prepared observation fixture are frozen assets named by
the Target's logical IDs (`Target.assets`), resolved through the machine's
asset configuration (`flash_vla.assets`); an explicit local path overrides one
and must name its own ID. Only the noise is drawn, from the seed, so the
fixture's provenance names the seed beside the observation
(`flash_vla.inference.build_runner`).
"""
from __future__ import annotations

import torch

from flash_vla.assets import locate_assets
from flash_vla.provenance import FixtureProvenance, WeightsProvenance
from flash_vla.runtime.identity import validate_weight_schema
from flash_vla.runtime.runner import RunnerSource
from flash_vla.runtime.vla import ConfigValue, Target

from .definition import GrootConfig
from .weights import CHECKPOINT_ID, Checkpoint, weight_shapes


def runner_source(target: Target[GrootConfig, None], *, device: str, declare: bool,
                  seed: int = 0, steps: int = 4, layers: int = 16, sequence_length: int = 156,
                  checkpoint: str | None = None, fixture: str | None = None,
                  checkpoint_id: str | None = None, fixture_id: str | None = None,
                  asset_config: str | None = None) -> RunnerSource:
    """The frozen weights, seeded fixture and configuration of one GR00T
    construction; `declare` reads no asset configuration and no weights."""
    overrides = {"checkpoint": checkpoint, "fixture": fixture}
    ids = {"checkpoint": checkpoint_id, "fixture": fixture_id}
    for role, path in overrides.items():
        if path is not None and ids[role] is None:
            raise ValueError(f"{role} override requires {role}_id")
    identifiers = {role: ids[role] or value for role, value in target.assets.items()}
    fixture_context = f"{identifiers['fixture']}/noise-seed-{seed}"
    named_weights = WeightsProvenance(checkpoint_id=identifiers["checkpoint"],
                                      checkpoint_digest=identifiers["checkpoint"])
    named_fixture = FixtureProvenance(id=fixture_context, digest=fixture_context)
    config = dict(sequence_length=sequence_length, steps=steps, layers=layers)
    if declare:
        return RunnerSource(checkpoint=None, weights_provenance=named_weights,
                            fixture_provenance=named_fixture, assets={}, config=config)
    assets = locate_assets(identifiers, overrides=overrides, config=asset_config)
    return RunnerSource(checkpoint=Checkpoint(assets["checkpoint"]),
                        weights_provenance=named_weights, fixture_provenance=named_fixture,
                        assets=assets, config=config)


def official_weights(*, device: str, seed: int = 0, checkpoint: str | None = None,
                     checkpoint_id: str | None = None, asset_config: str | None = None,
                     **construction: ConfigValue) -> dict[str, torch.Tensor]:
    """The official checkpoint's tensors a construction with these options
    runs, for the reference (`reference.load`), located as `runner_source`
    locates them; the checkpoint stores bfloat16. GR00T has no seeded weights,
    so `seed` selects nothing here, nor do the other construction options."""
    assets = locate_assets({"checkpoint": checkpoint_id or CHECKPOINT_ID},
                           overrides={"checkpoint": checkpoint}, config=asset_config)
    source = Checkpoint(assets["checkpoint"])
    validate_weight_schema(source.shapes, weight_shapes())
    weights = {name: torch.empty(shape, dtype=torch.bfloat16, device=device)
               for name, shape in weight_shapes().items()}
    source.copy_into(weights)
    return weights


__all__ = ["official_weights", "runner_source"]
