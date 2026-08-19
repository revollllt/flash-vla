"""Per-inference prefix inputs, computed on the host into pinned staging.

Pi0.5's prompt contains the discretized state, so four things change on every
call and none of them can be baked into the graph: the prompt token ids, which
rows of the prompt are valid, the attention mask that follows from that, and the
decoder's RoPE offset.

All four are cheap -- 16 us of tokenization and 27 KB of copies -- and all four
are *host* work, which is why they are here rather than in a kernel. The engine
overlaps them with the vision tower: vision depends only on the images, this
depends only on the state, and both inputs arrive together. See PLAN.md §3.2.
The device-side alternative is real but buys nothing once the host work is
hidden, and it would put the token ids out of reach of the parity gate.

Staging is pinned and reused, so `build` allocates nothing and the copies can
be issued `non_blocking=True` -- a pageable copy would synchronize and undo the
overlap.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from flash_vla.models.pi05.spec import ENCODER_DIM, HEAD_DIM, ROPE_THETA, VISION_TOKENS

from .buffers import MASK_NEG


class PrefixInputs:
    """Pinned host staging for one inference's prompt-dependent inputs.

    `build(state)` fills the buffers and returns the number of valid prefix
    rows. `copy_into(buffers, stream_ordered=True)` issues the four copies.
    """

    def __init__(self, tokenizer, num_views: int, chunk_size: int):
        self.tokenizer = tokenizer
        self.prompt_len = tokenizer.max_token_len
        self.image_tokens = num_views * VISION_TOKENS
        self.encoder_seq_len = self.image_tokens + self.prompt_len
        self.cache_len = self.encoder_seq_len + chunk_size
        self.chunk_size = chunk_size

        def pinned(*shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, pin_memory=True)

        self.token_ids = pinned(self.prompt_len, dtype=torch.int32)
        self.embed_scale = pinned(self.prompt_len, 1)
        self.mask_bias = pinned(self.cache_len)
        self.decoder_rope = pinned(chunk_size, HEAD_DIM)

        # The embedder scales by sqrt(width) (`models/gemma.py:150`, and its
        # PyTorch mirror). Folding it into the same vector that zeroes padding
        # keeps the gather to one multiply.
        self._embed_scale = math.sqrt(ENCODER_DIM)
        # Suffix keys are never masked; only the prefix half is rewritten.
        self.mask_bias[self.encoder_seq_len:] = 0.0

        self._inv_freq = 1.0 / (ROPE_THETA ** (
            torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        self._offsets = torch.arange(chunk_size, dtype=torch.float32)

        self.n_valid = 0

    def build(self, state) -> int:
        """Tokenize `state` into the staging buffers; return the valid prefix length."""
        if isinstance(state, torch.Tensor):
            state = state.detach().to("cpu", torch.float32).numpy()
        tokens, mask = self.tokenizer.encode(np.asarray(state))
        n_tokens = int(mask.sum())
        n_valid = self.image_tokens + n_tokens

        self.token_ids.copy_(torch.from_numpy(tokens.astype(np.int32)))
        scale = torch.from_numpy(mask.astype(np.float32) * self._embed_scale)
        self.embed_scale.copy_(scale[:, None])

        self.mask_bias[:n_valid] = 0.0
        self.mask_bias[n_valid:self.encoder_seq_len] = MASK_NEG

        # Suffix positions are n_valid + 0..chunk-1 (`models/pi0.py:259`).
        phase = self._inv_freq[None, :] * (self._offsets + n_valid)[:, None]
        self.decoder_rope.copy_(
            torch.stack([torch.cos(phase), torch.sin(phase)], dim=2).view(-1, HEAD_DIM))

        self.n_valid = n_valid
        return n_valid

    @torch.no_grad()
    def copy_into(self, buffers: dict[str, torch.Tensor], non_blocking: bool = True) -> None:
        """Issue the staged copies onto the current stream."""
        buffers["prompt_token_ids"].copy_(self.token_ids, non_blocking=non_blocking)
        buffers["prompt_embed_scale"].copy_(self.embed_scale, non_blocking=non_blocking)
        buffers["prefix_mask_bias"].copy_(self.mask_bias, non_blocking=non_blocking)
        buffers["decoder_rope_weights"].copy_(self.decoder_rope, non_blocking=non_blocking)

    @property
    def nbytes(self) -> int:
        """Total bytes copied per inference, for the record."""
        return sum(t.numel() * t.element_size() for t in
                   (self.token_ids, self.embed_scale, self.mask_bias, self.decoder_rope))
