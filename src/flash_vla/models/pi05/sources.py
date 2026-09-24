"""Where a Pi0.5 runner's weights and fixture come from, and the provenance each carries.

Weights are seeded synthetic ones, an OpenPI checkpoint (loaded through
OpenPI, which validates the named upstream config and supplies the action
horizon), or an already converted checkpoint (read without OpenPI: it resolves
no config, so the caller supplies the shape profile and the immutable ID).
Every construction folds the weights for its own step schedule. The fixture is
the seeded one `sample_inputs` draws, prompted with the construction's task.
Neither carries hardware, so every Pi0.5 Target shares this module
(`flash_vla.inference.build_runner`).
"""
from __future__ import annotations

import os
from pathlib import Path

from flash_vla.provenance import FixtureProvenance, WeightsProvenance, canonical_digest
from flash_vla.runtime.runner import RunnerSource
from flash_vla.runtime.vla import Target

from . import openpi, weights
from .definition import Pi05Config
from .prompt import PrefixInputs
from .spec import MAX_TOKEN_LEN, random_checkpoint_revision

#: The task a construction prompts with unless it names one.
DEFAULT_PROMPT = "pick up the plate and put it in the sink"
#: The producer of the seeded input fixture, part of its ID and digest.
FIXTURE_PRODUCER = "flash-vla/pi05-inputs-v1"


def runner_source(target: Target[Pi05Config, PrefixInputs], *, device: str, declare: bool,
                  seed: int = 0, num_views: int = 3, chunk_size: int | None = None,
                  steps: int = 10, layers: int = 18, prompt_len: int | None = None,
                  prompt: str = DEFAULT_PROMPT, tokenizer_path: str | None = None,
                  checkpoint: str | None = None, converted_checkpoint: str | None = None,
                  checkpoint_id: str | None = None, checkpoint_digest: str | None = None,
                  openpi_config: str | None = None) -> RunnerSource:
    """The weights, fixture and configuration of one Pi0.5 construction.

    `declare` reads no weight values and no assets; an OpenPI checkpoint's
    config is still resolved, since it fixes the shape.
    """
    if checkpoint is not None and converted_checkpoint is not None:
        raise ValueError("pass checkpoint or converted_checkpoint, not both")
    # The chunk is the Target's own default unless an upstream config names one.
    default_chunk = target.model.configure().chunk_size
    if checkpoint is None and converted_checkpoint is None:
        if any(value is not None for value in (checkpoint_id, checkpoint_digest, openpi_config)):
            raise ValueError("checkpoint provenance/config requires a real checkpoint path")
        checkpoint_id = checkpoint_digest = random_checkpoint_revision(seed)
        chunk_size = default_chunk if chunk_size is None else chunk_size
    elif converted_checkpoint is not None:
        if not (checkpoint_id and checkpoint_digest):
            raise ValueError("a converted checkpoint requires checkpoint_id and checkpoint_digest")
        if openpi_config is not None:
            raise ValueError("a converted checkpoint resolves no OpenPI config; "
                             "pass `checkpoint` to have OpenPI resolve and validate one")
        chunk_size = default_chunk if chunk_size is None else chunk_size
    else:
        if not (checkpoint_id and checkpoint_digest and openpi_config):
            raise ValueError("real checkpoint requires checkpoint_id, checkpoint_digest and openpi_config")
        reference_config = openpi.resolve_config(checkpoint, openpi_config)
        if chunk_size is not None and chunk_size != reference_config.action_horizon:
            raise ValueError("chunk_size differs from the checkpoint's explicit OpenPI config")
        if prompt_len is not None and prompt_len != reference_config.max_token_len:
            raise ValueError("prompt_len differs from the checkpoint's explicit OpenPI config")
        chunk_size = reference_config.action_horizon

    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers,
                  prompt_len=prompt_len or MAX_TOKEN_LEN, prompt=prompt)
    named_weights = WeightsProvenance(checkpoint_id=checkpoint_id,
                                      checkpoint_digest=checkpoint_digest)
    # The seeded inputs of `sample_inputs(seed)`, prompted with the construction's task.
    named_fixture = FixtureProvenance(
        id=f"{FIXTURE_PRODUCER}/seed-{seed}",
        digest=canonical_digest({"producer": FIXTURE_PRODUCER, "seed": seed, "prompt": prompt}))
    if declare:
        return RunnerSource(checkpoint=None, weights_provenance=named_weights,
                            fixture_provenance=named_fixture, assets={}, config=config)
    if converted_checkpoint is not None:
        unfolded = openpi.converted_checkpoint(converted_checkpoint)
    elif checkpoint is None:
        unfolded = weights.random_checkpoint(seed=seed, device=device)
    else:
        model = openpi.build_model(checkpoint, device, seed=seed, config=reference_config)
        unfolded = openpi.target_checkpoint(model)
        del model
    tokenizer = Path(tokenizer_path or os.environ["PALIGEMMA_TOKENIZER"])
    return RunnerSource(checkpoint=weights.fold(unfolded, steps=steps),
                        weights_provenance=named_weights, fixture_provenance=named_fixture,
                        assets={"tokenizer": tokenizer}, config=config)


__all__ = ["runner_source"]
