"""Engine factories by Target name, for the generic harnesses.

The one place a harness learns how to construct a Target: random weights at a
stated seed, the tokenizer the model needs, a default task. Everything after
construction goes through the engine protocol (`flash_vla.runtime.engine`),
so the latency and correctness runners contain no model names; this registry
is where they enter.

Weights are seeded so two engines built with the same seed share them
bit-for-bit, which is what makes an in-engine comparison an implementation
difference and nothing else.
"""
from __future__ import annotations

from typing import Any, Callable

from .plans import parse_plan

DEFAULT_PROMPT = "pick up the plate and put it in the sink"


def _pi05(plan: dict[str, str] | None = None, *, seed: int = 0, num_views: int = 3,
          chunk_size: int = 50, steps: int = 10, layers: int = 18,
          prompt_len: int | None = None, device: str = "cuda",
          prompt: str = DEFAULT_PROMPT, tokenizer_path: str | None = None):
    from flash_vla.hardware.nvidia.h100.pi05 import Pi05Inference
    from flash_vla.models.pi05.spec import MAX_TOKEN_LEN
    from flash_vla.models.pi05.tokenize import Pi05Tokenizer
    from flash_vla.models.pi05.weights import fold, random_checkpoint

    tokenizer = Pi05Tokenizer(tokenizer_path)
    checkpoint = fold(random_checkpoint(seed=seed, device=device), steps=steps)
    engine = Pi05Inference(checkpoint, tokenizer, num_views=num_views, chunk_size=chunk_size,
                           steps=steps, layers=layers, prompt_len=prompt_len or MAX_TOKEN_LEN,
                           device=device, plan=plan)
    engine.set_task(prompt)
    return engine


def _pi0(plan: dict[str, str] | None = None, *, seed: int = 0, num_views: int = 3,
         chunk_size: int = 50, steps: int = 10, layers: int = 18, prompt_len: int = 0,
         device: str = "cuda", fused: bool = True):
    from flash_vla.hardware.nvidia.h100.pi0 import Pi0Inference
    from flash_vla.models.pi0 import random_checkpoint

    if plan:
        raise ValueError("the Pi0 Target has one backend; select its fused overlay with "
                         f"fused=True/False, not a plan (got {plan})")
    checkpoint = random_checkpoint(num_views=num_views, chunk_size=chunk_size,
                                   prompt_len=prompt_len, seed=seed, device=device)
    return Pi0Inference(checkpoint, num_views=num_views, chunk_size=chunk_size, steps=steps,
                        layers=layers, fused=fused, device=device)


#: Target name -> factory. Short aliases resolve through `resolve`.
TARGETS: dict[str, Callable[..., Any]] = {
    "hardware/nvidia/h100/pi05": _pi05,
    "hardware/nvidia/h100/pi0": _pi0,
}
_ALIASES = {"h100/pi05": "hardware/nvidia/h100/pi05", "pi05": "hardware/nvidia/h100/pi05",
            "h100/pi0": "hardware/nvidia/h100/pi0", "pi0": "hardware/nvidia/h100/pi0"}


def resolve(name: str) -> str:
    """The full Target name for `name` or one of its aliases."""
    full = _ALIASES.get(name, name)
    if full not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {sorted(TARGETS)} "
                       f"and aliases {sorted(_ALIASES)}")
    return full


def build(name: str, plan: str | dict[str, str] | None = None, **overrides):
    """Construct the engine of Target `name` on `plan` (a plan name, JSON or dict)."""
    routes = parse_plan(plan) if isinstance(plan, str) else plan
    return TARGETS[resolve(name)](routes, **overrides)


__all__ = ["DEFAULT_PROMPT", "TARGETS", "build", "resolve"]
