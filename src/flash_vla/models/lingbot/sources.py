"""Where a LingBot-VLA runner's weights and fixture come from, and the provenance each carries.

Both are frozen assets named by the Target's logical IDs (`Target.assets`),
resolved to local paths through the machine's asset configuration
(`flash_vla.assets`). An explicit local checkpoint or fixture overrides the
configured one and must name its own ID and digest
(`flash_vla.inference.build_runner`).
"""
from __future__ import annotations

from flash_vla.assets import locate_assets
from flash_vla.provenance import FixtureProvenance, WeightsProvenance
from flash_vla.runtime.runner import RunnerSource
from flash_vla.runtime.vla import Target

from . import weights
from .definition import FIXTURE_SEED, LingBotConfig


def runner_source(target: Target[LingBotConfig, None], *, device: str, declare: bool,
                  seed: int = FIXTURE_SEED, steps: int = 10, layers: int = 36,
                  checkpoint: str | None = None, fixture: str | None = None,
                  checkpoint_id: str | None = None, checkpoint_digest: str | None = None,
                  fixture_id: str | None = None, fixture_digest: str | None = None,
                  asset_config: str | None = None) -> RunnerSource:
    """The frozen weights, fixture and configuration of one LingBot construction.

    The fixture is recorded at `FIXTURE_SEED`, so `seed` selects nothing.
    `declare` reads no asset configuration and no weights.
    """
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
    config = dict(steps=steps, layers=layers)
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

__all__ = ["runner_source"]
