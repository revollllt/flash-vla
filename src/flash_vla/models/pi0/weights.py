"""Pi0 checkpoint construction helpers."""

from __future__ import annotations

import torch

from .spec import weight_shapes


def random_checkpoint(num_views: int = 3, chunk_size: int = 50, prompt_len: int = 256,
                      wscale: float = 0.05, seed: int = 0, device: str = "cuda") -> dict:
    """Create a synthetic checkpoint for benchmarking.

    `num_views` and `chunk_size` remain in the public signature for compatibility
    with the engine helpers; Pi0 weight shapes do not depend on either value.
    The small default scale keeps deep random residual streams numerically useful.
    """
    del num_views, chunk_size
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        name: (torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * wscale
               ).to(torch.bfloat16)
        for name, shape in weight_shapes(prompt_len).items()
    }
