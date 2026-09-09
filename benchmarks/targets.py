"""Runner factories by Target name, for the generic harnesses.

The one place a harness learns how to construct a Target: random weights at a
stated seed, the tokenizer the model needs, a default task. Everything after
construction goes through the engine protocol (`flash_vla.runtime.engine`),
so the latency, profile, kernel, floor and correctness runners contain no
model names; this registry is where they enter.

Weights are seeded so two runners built with the same seed share them
bit-for-bit, which is what makes an in-engine comparison an implementation
difference and nothing else.

A plan is `"shipped"` (the Target's one deployed plan, the default),
`"reference"` (its correctness oracle route), a JSON object, or a path to a
JSON file (the candidate plans of the optimization workspace, `lab/plans/`).
"""
from __future__ import annotations

import os
from typing import Any, Callable

from flash_vla.runtime import ModelRunner
from flash_vla.runtime.identity import canonical_digest

DEFAULT_LINGBOT_CHECKPOINT = "/data/user/jzou521/models/lingbot-vla-4b-posttrain-robotwin-fb71a2c"
DEFAULT_LINGBOT_FIXTURE = (
    "/data/user/jzou521/codes/cuda/flash-vla/artifacts/onboarding/"
    "lingbot-vla-4b-h100-bf16/official/fixture.safetensors"
)

DEFAULT_PROMPT = "pick up the plate and put it in the sink"


def _pi05(plan: Any = "shipped", *, seed: int = 0, num_views: int = 3, chunk_size: int = 50,
          steps: int = 10, layers: int = 18, prompt_len: int | None = None,
          device: str = "cuda", prompt: str = DEFAULT_PROMPT, tokenizer_path: str | None = None,
          declare: bool = False):
    from flash_vla.hardware.nvidia.h100.pi05 import TARGET
    from flash_vla.models.pi05.spec import MAX_TOKEN_LEN
    from flash_vla.models.pi05.spec import random_checkpoint_revision
    from flash_vla.models.pi05.tokenize import Pi05Tokenizer
    from flash_vla.models.pi05.weights import fold, random_checkpoint

    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers,
                  prompt_len=prompt_len or MAX_TOKEN_LEN, prompt=prompt)
    checkpoint_id = random_checkpoint_revision(seed)
    if declare:
        runner = ModelRunner(TARGET, None, checkpoint_id=checkpoint_id,
                             checkpoint_digest=checkpoint_id, plan=plan,
                           device=device, capture=False, **config)
    else:
        checkpoint = fold(random_checkpoint(seed=seed, device=device), steps=steps)
        runner = ModelRunner(TARGET, checkpoint, checkpoint_id=checkpoint_id,
                             checkpoint_digest=checkpoint_id, plan=plan, device=device,
                           tokenizer=Pi05Tokenizer(tokenizer_path), **config)
    fixture = {"producer": "flash-vla/pi05-inputs-v1", "seed": seed, "prompt": prompt}
    runner.measurement_context["fixture"] = {
        "id": fixture["producer"] + "/seed-" + str(seed), "digest": canonical_digest(fixture),
    }
    return runner


def _pi0(plan: Any = "shipped", *, seed: int = 0, num_views: int = 3, chunk_size: int = 50,
         steps: int = 10, layers: int = 18, prompt_len: int = 0, device: str = "cuda",
         declare: bool = False):
    from flash_vla.hardware.nvidia.h100.pi0 import TARGET
    from flash_vla.models.pi0 import random_checkpoint
    from flash_vla.models.pi0.spec import random_checkpoint_revision

    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers)
    checkpoint_id = random_checkpoint_revision(seed)
    if declare:
        runner = ModelRunner(TARGET, None, checkpoint_id=checkpoint_id,
                             checkpoint_digest=checkpoint_id, plan=plan,
                           device=device, capture=False,
                           prompt_len=prompt_len, **config)
    else:
        checkpoint = random_checkpoint(num_views=num_views, chunk_size=chunk_size,
                                       prompt_len=prompt_len, seed=seed, device=device)
        runner = ModelRunner(TARGET, checkpoint, checkpoint_id=checkpoint_id,
                             checkpoint_digest=checkpoint_id, plan=plan,
                           device=device, **config)
    fixture = {"producer": "flash-vla/pi0-inputs-v1", "seed": seed}
    runner.measurement_context["fixture"] = {
        "id": fixture["producer"] + "/seed-" + str(seed), "digest": canonical_digest(fixture),
    }
    return runner


def _lingbot(plan: Any = "shipped", *, seed: int = 42, steps: int = 10, layers: int = 36,
             device: str = "cuda", declare: bool = False,
             checkpoint: str = DEFAULT_LINGBOT_CHECKPOINT,
             fixture: str = DEFAULT_LINGBOT_FIXTURE,
             checkpoint_id: str | None = None, checkpoint_digest: str | None = None,
             fixture_id: str | None = None, fixture_digest: str | None = None):
    from flash_vla.hardware.nvidia.h100.lingbot_vla import TARGET
    from flash_vla.models.lingbot import CHECKPOINT_REVISION, load_checkpoint

    if checkpoint_id is None:
        if checkpoint != DEFAULT_LINGBOT_CHECKPOINT:
            raise ValueError("checkpoint override needs checkpoint_id and checkpoint_digest")
        checkpoint_id = CHECKPOINT_REVISION
        checkpoint_digest = CHECKPOINT_REVISION
    if not checkpoint_digest:
        raise ValueError("checkpoint_digest must identify the immutable weights manifest")
    if fixture_id is None:
        if fixture != DEFAULT_LINGBOT_FIXTURE:
            raise ValueError("fixture override needs fixture_id and fixture_digest")
        fixture_id = "lingbot-robotwin-canonical-v1/seed-42"
        fixture_digest = fixture_id
    if not fixture_digest:
        raise ValueError("fixture_digest must identify the immutable fixture")
    config = dict(steps=steps, layers=layers)
    if not declare:
        os.environ["LINGBOT_CHECKPOINT"] = checkpoint
        os.environ["LINGBOT_FIXTURE"] = fixture
    runner = ModelRunner(TARGET, None if declare else load_checkpoint(checkpoint),
                         checkpoint_id=checkpoint_id, checkpoint_digest=checkpoint_digest,
                         plan=plan, device="cpu" if declare else device,
                         capture=not declare, **config)
    runner.measurement_context["fixture"] = {"id": fixture_id, "digest": fixture_digest}
    return runner



#: Target name -> factory. Short aliases resolve through `resolve`.
TARGETS: dict[str, Callable[..., Any]] = {
    "hardware/nvidia/h100/lingbot_vla": _lingbot,
    "hardware/nvidia/h100/pi05": _pi05,
    "hardware/nvidia/h100/pi0": _pi0,
}
_ALIASES = {"h100/pi05": "hardware/nvidia/h100/pi05", "pi05": "hardware/nvidia/h100/pi05",
            "h100/pi0": "hardware/nvidia/h100/pi0", "pi0": "hardware/nvidia/h100/pi0",
            "h100/lingbot_vla": "hardware/nvidia/h100/lingbot_vla",
            "lingbot_vla": "hardware/nvidia/h100/lingbot_vla"}

#: The two plan names every Target understands.
PLAN_NAMES = ("shipped", "reference")


def resolve(name: str) -> str:
    """The full Target name for `name` or one of its aliases."""
    full = _ALIASES.get(name, name)
    if full not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {sorted(TARGETS)} "
                       f"and aliases {sorted(_ALIASES)}")
    return full


def build(name: str, plan: Any = "shipped", **overrides):
    """Construct the runner of Target `name` on `plan`."""
    return TARGETS[resolve(name)](plan, **overrides)


def declare(name: str, plan: Any = "shipped", **overrides):
    """Construct the runner of Target `name` up to its graph: no weights, no device."""
    return TARGETS[resolve(name)](plan, declare=True, device="cpu", **overrides)


__all__ = ["DEFAULT_PROMPT", "PLAN_NAMES", "TARGETS", "build", "declare", "resolve"]
