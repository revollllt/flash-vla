"""Where a Pi0 runner's weights and fixture come from, and the provenance each carries.

Pi0's weights and inputs are both seeded synthetic ones: the seed names the
checkpoint (its ID is its digest) and the fixture. Neither carries hardware,
so every Pi0 Target shares this module (`flash_vla.inference.build_runner`).
"""
from __future__ import annotations

from flash_vla.provenance import FixtureProvenance, WeightsProvenance, canonical_digest
from flash_vla.runtime.runner import RunnerSource
from flash_vla.runtime.vla import Target

from .definition import Pi0Config
from .spec import random_checkpoint_revision
from .weights import random_checkpoint

#: The producer of the seeded input fixture, part of its ID and digest.
FIXTURE_PRODUCER = "flash-vla/pi0-inputs-v1"


def runner_source(target: Target[Pi0Config, None], *, device: str, declare: bool,
                  seed: int = 0, num_views: int = 3, chunk_size: int = 50, steps: int = 10,
                  layers: int = 18, prompt_len: int = 0) -> RunnerSource:
    """The seeded weights, fixture and configuration of one Pi0 construction;
    `declare` generates no weight values."""
    revision = random_checkpoint_revision(seed)
    named_fixture = FixtureProvenance(
        id=f"{FIXTURE_PRODUCER}/seed-{seed}",
        digest=canonical_digest({"producer": FIXTURE_PRODUCER, "seed": seed}))
    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers,
                  prompt_len=prompt_len)
    checkpoint = None if declare else random_checkpoint(
        num_views=num_views, chunk_size=chunk_size, prompt_len=prompt_len, seed=seed, device=device)
    return RunnerSource(checkpoint=checkpoint,
                        weights_provenance=WeightsProvenance(checkpoint_id=revision,
                                                             checkpoint_digest=revision),
                        fixture_provenance=named_fixture, assets={}, config=config)


__all__ = ["runner_source"]
