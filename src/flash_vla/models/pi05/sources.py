"""Where a Pi0.5 runner's weights and fixture come from, and the provenance each carries.

The workload fixes the shape (cameras, chunk, prompt slots, what the prompt
carries; `definition.Pi05Model.workloads`). Weights are seeded synthetic ones,
an OpenPI checkpoint (loaded through OpenPI, which validates the named upstream
config; its action horizon and prompt length must be the workload's), or an
already converted checkpoint (read without OpenPI: it resolves no config, so
the caller names the immutable ID).
Every construction folds the weights for its own step schedule. The fixture is
the seeded one `sample_inputs` draws, prompted with the construction's task.
Neither carries hardware, so every Pi0.5 Target shares this module
(`flash_vla.inference.build_runner`).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from flash_vla.models.official import official_schema
from flash_vla.models.pi0.openpi import read_official
from flash_vla.provenance import FixtureProvenance, WeightsProvenance, canonical_digest
from flash_vla.runtime.runner import RunnerSource
from flash_vla.runtime.vla import ConfigValue, Target

from . import openpi, reference, weights
from .definition import Pi05Config
from .prompt import PrefixInputs
from .spec import STATE_DIM, random_checkpoint_revision

#: The task a construction prompts with unless it names one.
DEFAULT_PROMPT = "pick up the plate and put it in the sink"
#: The producer of the seeded input fixture, part of its ID and digest. v2: the
#: prompt's content (with or without the state, and how many state values) is
#: part of the fixture.
FIXTURE_PRODUCER = "flash-vla/pi05-inputs-v2"


def runner_source(target: Target[Pi05Config, PrefixInputs], *, device: str, declare: bool,
                  num_views: int, chunk_size: int, prompt_len: int, discrete_state: bool,
                  robot_state_dim: int = STATE_DIM, prompt_tokens: int | None = None,
                  seed: int = 0, steps: int = 10,
                  layers: int = 18, prompt: str = DEFAULT_PROMPT, tokenizer_path: str | None = None,
                  checkpoint: str | None = None, converted_checkpoint: str | None = None,
                  checkpoint_id: str | None = None, checkpoint_digest: str | None = None,
                  openpi_config: str | None = None) -> RunnerSource:
    """The weights, fixture and configuration of one Pi0.5 construction.

    `declare` reads no weight values and no assets; an OpenPI checkpoint's
    config is still resolved, since it fixes the shape. `prompt_tokens`
    replaces the tokenized prompt by that many seeded tokens (`Pi05Config`).
    """
    if checkpoint is not None and converted_checkpoint is not None:
        raise ValueError("pass checkpoint or converted_checkpoint, not both")
    if checkpoint is None and converted_checkpoint is None:
        if any(value is not None for value in (checkpoint_id, checkpoint_digest, openpi_config)):
            raise ValueError("checkpoint provenance/config requires a real checkpoint path")
        checkpoint_id = checkpoint_digest = random_checkpoint_revision(seed)
    elif converted_checkpoint is not None:
        if not (checkpoint_id and checkpoint_digest):
            raise ValueError("a converted checkpoint requires checkpoint_id and checkpoint_digest")
        if openpi_config is not None:
            raise ValueError("a converted checkpoint resolves no OpenPI config; "
                             "pass `checkpoint` to have OpenPI resolve and validate one")
    else:
        if not (checkpoint_id and checkpoint_digest and openpi_config):
            raise ValueError("real checkpoint requires checkpoint_id, checkpoint_digest and openpi_config")
        reference_config = openpi.resolve_config(checkpoint, openpi_config)
        if (chunk_size, prompt_len) != (reference_config.action_horizon, reference_config.max_token_len):
            raise ValueError(f"the checkpoint's OpenPI config (action_horizon "
                             f"{reference_config.action_horizon}, max_token_len "
                             f"{reference_config.max_token_len}) differs from the workload's "
                             f"chunk_size {chunk_size} and prompt_len {prompt_len}")

    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers,
                  prompt_len=prompt_len, prompt=prompt, discrete_state=discrete_state,
                  robot_state_dim=robot_state_dim, prompt_tokens=prompt_tokens)
    named_weights = WeightsProvenance(checkpoint_id=checkpoint_id,
                                      checkpoint_digest=checkpoint_digest)
    # The seeded inputs of `sample_inputs(seed)`, prompted with the construction's
    # task and as much of the state as the prompt carries.
    named_fixture = FixtureProvenance(
        id=f"{FIXTURE_PRODUCER}/seed-{seed}",
        digest=canonical_digest({"producer": FIXTURE_PRODUCER, "seed": seed, "prompt": prompt,
                                 "discrete_state": discrete_state,
                                 "robot_state_dim": robot_state_dim,
                                 **({} if prompt_tokens is None
                                    else {"prompt_tokens": prompt_tokens})}))
    if declare:
        return RunnerSource(checkpoint=None, weights_provenance=named_weights,
                            fixture_provenance=named_fixture, assets={}, config=config)
    if converted_checkpoint is not None:
        unfolded = openpi.converted_checkpoint(converted_checkpoint)
    elif checkpoint is None:
        unfolded = weights.random_checkpoint(seed=seed, device=device)
    else:
        model = openpi.build_model(checkpoint, device, seed=seed, config=reference_config)
        unfolded = openpi.target_checkpoint(model.state_dict())
        del model
    tokenizer = Path(tokenizer_path or os.environ["PALIGEMMA_TOKENIZER"])
    return RunnerSource(checkpoint=weights.fold(unfolded, steps=steps),
                        weights_provenance=named_weights, fixture_provenance=named_fixture,
                        assets={"tokenizer": tokenizer}, config=config)


def official_weights(*, device: str, seed: int, checkpoint: str | None = None,
                     converted_checkpoint: str | None = None,
                     **construction: ConfigValue) -> dict[str, torch.Tensor]:
    """The official-layout weights a construction with these options runs, for
    the reference (`reference.load`): the seeded random ones, or the OpenPI
    PyTorch checkpoint's -- both `checkpoint` and `converted_checkpoint` name
    one. The other construction options select nothing here."""
    path = checkpoint or converted_checkpoint
    if path is None:
        return reference.random_weights(seed, device=device)
    return read_official(path, device=device,
                         names=official_schema(reference.make_reference().parts(),
                                               prefixes=reference.PREFIXES))


__all__ = ["official_weights", "runner_source"]
