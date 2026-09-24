"""Pi0 checkpoint construction helpers."""

from __future__ import annotations

import torch

from .openpi import target_checkpoint
from .reference import random_weights


def random_checkpoint(seed: int = 0, *, prompt_ids: torch.Tensor,
                      device: str = "cuda") -> dict[str, torch.Tensor]:
    """A synthetic Pi0 checkpoint in the engine's layout.

    It is the seeded official-layout weights the reference runs
    (`reference.random_weights`), converted as a real checkpoint is
    (`openpi.target_checkpoint`), with `prompt_ids` as the fixed prompt, so the
    engine and the reference hold the same model.
    """
    return target_checkpoint(random_weights(seed, device=device), prompt_ids=prompt_ids.to(device))


__all__ = ["random_checkpoint"]
