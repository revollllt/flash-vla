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

from typing import Any, Callable

from flash_vla.runtime import ModelRunner

DEFAULT_PROMPT = "pick up the plate and put it in the sink"


def _pi05(plan: Any = "shipped", *, seed: int = 0, num_views: int = 3, chunk_size: int = 50,
          steps: int = 10, layers: int = 18, prompt_len: int | None = None,
          device: str = "cuda", prompt: str = DEFAULT_PROMPT, tokenizer_path: str | None = None,
          declare: bool = False):
    from flash_vla.hardware.nvidia.h100.pi05 import TARGET
    from flash_vla.models.pi05.spec import MAX_TOKEN_LEN
    from flash_vla.models.pi05.tokenize import Pi05Tokenizer
    from flash_vla.models.pi05.weights import fold, random_checkpoint

    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers,
                  prompt_len=prompt_len or MAX_TOKEN_LEN, prompt=prompt)
    if declare:
        return ModelRunner(TARGET, None, plan=plan, device=device, capture=False, **config)
    checkpoint = fold(random_checkpoint(seed=seed, device=device), steps=steps)
    return ModelRunner(TARGET, checkpoint, plan=plan, device=device,
                       tokenizer=Pi05Tokenizer(tokenizer_path), **config)


def _pi0(plan: Any = "shipped", *, seed: int = 0, num_views: int = 3, chunk_size: int = 50,
         steps: int = 10, layers: int = 18, prompt_len: int = 0, device: str = "cuda",
         declare: bool = False):
    from flash_vla.hardware.nvidia.h100.pi0 import TARGET
    from flash_vla.models.pi0 import random_checkpoint

    config = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers)
    if declare:
        return ModelRunner(TARGET, None, plan=plan, device=device, capture=False,
                           prompt_len=prompt_len, **config)
    checkpoint = random_checkpoint(num_views=num_views, chunk_size=chunk_size,
                                   prompt_len=prompt_len, seed=seed, device=device)
    return ModelRunner(TARGET, checkpoint, plan=plan, device=device, **config)


#: Target name -> factory. Short aliases resolve through `resolve`.
TARGETS: dict[str, Callable[..., Any]] = {
    "hardware/nvidia/h100/pi05": _pi05,
    "hardware/nvidia/h100/pi0": _pi0,
}
_ALIASES = {"h100/pi05": "hardware/nvidia/h100/pi05", "pi05": "hardware/nvidia/h100/pi05",
            "h100/pi0": "hardware/nvidia/h100/pi0", "pi0": "hardware/nvidia/h100/pi0"}

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
