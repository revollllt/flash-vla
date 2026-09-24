"""Where a LingBot-VLA runner's weights and fixture come from, and the provenance each carries.

By default both are frozen assets named by the Target's logical IDs
(`Target.assets`), resolved to local paths through the machine's asset
configuration (`flash_vla.assets`). An explicit local checkpoint or fixture
overrides the configured one and must name its own ID and digest
(`flash_vla.inference.build_runner`). A `synthetic` construction needs no
asset: seeded random weights in the official layout
(`reference.random_weights`) and inputs drawn from the same seed
(`definition.LingBotModel.sample_inputs`).
"""
from __future__ import annotations

import torch

from flash_vla.assets import locate_assets
from flash_vla.provenance import FixtureProvenance, WeightsProvenance, canonical_digest
from flash_vla.runtime.runner import RunnerSource
from flash_vla.runtime.vla import ConfigValue, Target

from . import reference, weights
from .definition import FIXTURE_SEED, LingBotConfig
from .spec import CHECKPOINT_REVISION, FLOW_STEPS, LAYERS, WEIGHT_SHAPES, random_checkpoint_revision

#: The producer of the synthetic inputs, part of their ID and digest.
SYNTHETIC_FIXTURE = "flash-vla/lingbot-inputs-v1"


def runner_source(target: Target[LingBotConfig, None], *, device: str, declare: bool,
                  seed: int = FIXTURE_SEED, steps: int = FLOW_STEPS, layers: int = LAYERS,
                  synthetic: bool = False, checkpoint: str | None = None,
                  fixture: str | None = None, checkpoint_id: str | None = None,
                  checkpoint_digest: str | None = None, fixture_id: str | None = None,
                  fixture_digest: str | None = None,
                  asset_config: str | None = None) -> RunnerSource:
    """The weights, fixture and configuration of one LingBot construction.

    The recorded fixture provides only `FIXTURE_SEED`; a `synthetic`
    construction draws its weights and inputs from `seed`. `declare` reads
    no asset configuration and draws no weights.
    """
    config = dict(steps=steps, layers=layers, synthetic=synthetic)
    assets_named = (checkpoint, fixture, checkpoint_id, checkpoint_digest, fixture_id, fixture_digest,
                    asset_config)
    if synthetic and any(value is not None for value in assets_named):
        raise ValueError("a synthetic construction takes no checkpoint, fixture or asset configuration")
    if synthetic:
        revision = random_checkpoint_revision(seed)
        return RunnerSource(
            checkpoint=None if declare else reference.random_weights(seed, device=device),
            weights_provenance=WeightsProvenance(checkpoint_id=revision, checkpoint_digest=revision),
            fixture_provenance=FixtureProvenance(
                id=f"{SYNTHETIC_FIXTURE}/seed-{seed}",
                digest=canonical_digest({"producer": SYNTHETIC_FIXTURE, "seed": seed})),
            assets={}, config=config)
    if checkpoint_id is None and (checkpoint is not None or checkpoint_digest is not None):
        raise ValueError("checkpoint override needs checkpoint_id and checkpoint_digest")
    if checkpoint_id is not None and not checkpoint_digest:
        raise ValueError("checkpoint_digest must identify the immutable weights manifest")
    if fixture_id is None and (fixture is not None or fixture_digest is not None):
        raise ValueError("fixture override needs fixture_id and fixture_digest")
    if fixture_id is not None and not fixture_digest:
        raise ValueError("fixture_digest must identify the immutable fixture")
    # The Target's frozen assets name themselves: their logical ID is their digest.
    named_weights = (WeightsProvenance(checkpoint_id=target.assets["checkpoint"],
                                       checkpoint_digest=target.assets["checkpoint"])
                     if checkpoint_id is None
                     else WeightsProvenance(checkpoint_id=checkpoint_id,
                                            checkpoint_digest=checkpoint_digest))
    named_fixture = (FixtureProvenance(id=target.assets["fixture"], digest=target.assets["fixture"])
                     if fixture_id is None
                     else FixtureProvenance(id=fixture_id, digest=fixture_digest))
    if declare:
        return RunnerSource(checkpoint=None, weights_provenance=named_weights,
                            fixture_provenance=named_fixture, assets={}, config=config)
    assets = locate_assets(dict(target.assets, checkpoint=named_weights.checkpoint_id,
                                fixture=named_fixture.id),
                           overrides={"checkpoint": checkpoint, "fixture": fixture},
                           config=asset_config)
    return RunnerSource(checkpoint=weights.load_checkpoint(assets["checkpoint"]),
                        weights_provenance=named_weights, fixture_provenance=named_fixture,
                        assets=assets, config=config)


def official_weights(*, device: str, seed: int, synthetic: bool = False,
                     checkpoint: str | None = None, checkpoint_id: str | None = None,
                     asset_config: str | None = None,
                     **construction: ConfigValue) -> dict[str, torch.Tensor]:
    """The official-layout weights a construction with these options runs,
    for the reference (`reference.load`): the seeded random ones, or the
    checkpoint's, located as `runner_source` locates them and read into the
    runner's bfloat16. The layout is the engine's own, so nothing is
    converted. The other construction options select nothing here."""
    if synthetic:
        return reference.random_weights(seed, device=device)
    assets = locate_assets({"checkpoint": checkpoint_id or CHECKPOINT_REVISION},
                           overrides={"checkpoint": checkpoint}, config=asset_config)
    tensors = {name: torch.empty(shape, dtype=torch.bfloat16, device=device)
               for name, shape in WEIGHT_SHAPES.items()}
    weights.load_checkpoint(assets["checkpoint"]).copy_into(tensors)
    return tensors


__all__ = ["SYNTHETIC_FIXTURE", "official_weights", "runner_source"]
