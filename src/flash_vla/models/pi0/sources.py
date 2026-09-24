"""Where a Pi0 runner's weights and fixture come from, and the provenance each carries.

Weights are seeded synthetic ones -- the seed names the checkpoint, and its ID
is its digest -- or an OpenPI PyTorch checkpoint (`converted_checkpoint`),
whose immutable ID the caller states. Either is converted as a real
checkpoint is (`openpi.target_checkpoint`), with the fixture's prompt baked
in: Pi0 fixes its prompt at load. The fixture is the seeded one
`sample_inputs` draws plus `prompt_len` prompt token ids drawn from the same
seed (`fixture_prompt`); the default is an empty prompt, as the official
baseline check runs (`eval.pi0.official.sample_actions`). Neither carries
hardware, so every Pi0 Target shares this module
(`flash_vla.inference.build_runner`).
"""
from __future__ import annotations

import torch

from flash_vla.models.official import official_schema
from flash_vla.provenance import FixtureProvenance, WeightsProvenance, canonical_digest
from flash_vla.runtime.runner import RunnerSource
from flash_vla.runtime.vla import ConfigValue, Target

from . import openpi, reference, weights
from ..paligemma.reference import VOCABULARY
from .definition import Pi0Config
from .spec import random_checkpoint_revision

#: The producer of the seeded input fixture, part of its ID and digest. v2:
#: the fixture includes `prompt_len` prompt token ids drawn from the seed.
FIXTURE_PRODUCER = "flash-vla/pi0-inputs-v2"


def fixture_prompt(seed: int, prompt_len: int) -> torch.Tensor:
    """The fixture's prompt: `prompt_len` token ids drawn from `seed`, on the CPU."""
    return torch.randint(0, VOCABULARY, (prompt_len,), generator=torch.Generator().manual_seed(seed))


def runner_source(target: Target[Pi0Config, None], *, device: str, declare: bool,
                  seed: int = 0, num_views: int = 3, chunk_size: int = 50, steps: int = 10,
                  layers: int = 18, prompt_len: int = 0, converted_checkpoint: str | None = None,
                  checkpoint_id: str | None = None,
                  checkpoint_digest: str | None = None) -> RunnerSource:
    """The weights, fixture and configuration of one Pi0 construction;
    `declare` reads and generates no weight values."""
    if converted_checkpoint is None and (checkpoint_id is not None or checkpoint_digest is not None):
        raise ValueError("checkpoint provenance requires a real checkpoint path")
    if converted_checkpoint is not None and not (checkpoint_id and checkpoint_digest):
        raise ValueError("a converted checkpoint requires checkpoint_id and checkpoint_digest")
    named_fixture = FixtureProvenance(
        id=f"{FIXTURE_PRODUCER}/seed-{seed}",
        digest=canonical_digest({"producer": FIXTURE_PRODUCER, "seed": seed,
                                 "prompt_len": prompt_len}))
    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers,
                  prompt_len=prompt_len)
    if converted_checkpoint is None:
        revision = random_checkpoint_revision(seed)
        named_weights = WeightsProvenance(checkpoint_id=revision, checkpoint_digest=revision)
    else:
        named_weights = WeightsProvenance(checkpoint_id=checkpoint_id,
                                          checkpoint_digest=checkpoint_digest)
    if declare:
        return RunnerSource(checkpoint=None, weights_provenance=named_weights,
                            fixture_provenance=named_fixture, assets={}, config=config)
    prompt_ids = fixture_prompt(seed, prompt_len)
    checkpoint = (weights.random_checkpoint(seed, prompt_ids=prompt_ids, device=device)
                  if converted_checkpoint is None
                  else openpi.target_checkpoint(openpi.read_checkpoint(converted_checkpoint),
                                                prompt_ids=prompt_ids))
    return RunnerSource(checkpoint=checkpoint, weights_provenance=named_weights,
                        fixture_provenance=named_fixture, assets={}, config=config)


def official_weights(*, device: str, seed: int = 0, converted_checkpoint: str | None = None,
                     **construction: ConfigValue) -> dict[str, torch.Tensor]:
    """The official-layout weights a construction with these options runs, for
    the reference (`reference.load`): the seeded random ones, or the OpenPI
    PyTorch checkpoint's. The other construction options select nothing here."""
    if converted_checkpoint is None:
        return reference.random_weights(seed, device=device)
    return openpi.read_official(converted_checkpoint, device=device,
                                names=official_schema(reference.make_reference().parts(),
                                                      prefixes=reference.PREFIXES))


__all__ = ["fixture_prompt", "official_weights", "runner_source"]
