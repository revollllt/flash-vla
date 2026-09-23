"""Pi0.5 operations on a constructed runner: change the task, run the prefix alone.

They act on any device's Pi0.5 runner; its host state is the model's
`prompt.PrefixInputs`, which exists once the runner has loaded a checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch

from flash_vla.runtime.runner import ModelRunner

from .definition import Pi05Config
from .prompt import PrefixInputs

#: A runner built from a Pi0.5 Target.
Pi05Runner = ModelRunner[Pi05Config, PrefixInputs]


def prefix_inputs(runner: Pi05Runner) -> PrefixInputs:
    """The prompt host state of a runner that has loaded its checkpoint."""
    if runner.host_state is None:
        raise RuntimeError("this runner declared its graph without a checkpoint; it has no host state")
    return runner.host_state


def set_task(runner: Pi05Runner, prompt: str) -> None:
    """Install the task string. Call whenever it changes; see `Pi05Tokenizer.set_task`."""
    prefix_inputs(runner).tokenizer.set_task(prompt)


def forward_prefix(runner: Pi05Runner, images: torch.Tensor, state: torch.Tensor | np.ndarray) -> int:
    """Run the vision encoder, the host slot and the backbone; return the valid prefix rows.

    The KV cache is left in `buffers["prefix_k"]` / `["prefix_v"]`, of which
    the first `n_valid` rows carry data and the rest are masked padding.
    """
    runner.buffers["images"].copy_(images)
    runner.replay("vision_encoder")
    runner.host("prompt", state=state)
    runner.replay("llm_backbone")
    return prefix_inputs(runner).n_valid


__all__ = ["Pi05Runner", "forward_prefix", "prefix_inputs", "set_task"]
