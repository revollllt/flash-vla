"""Per-inference prefix inputs, computed on the host into pinned staging.

Pi0.5's prompt contains the discretized state, so four things change on every
call and none of them can be baked into the graph: the prompt token ids, which
rows of the prompt are valid, the attention mask that follows from that, and the
decoder's RoPE offset.

All four are *host* work, which is why they are here rather than in a kernel.
The program overlaps them with the vision tower: the host slot sits between the
vision_encoder and llm_backbone stages, vision depends only on the images, this
depends only on the state, and both inputs arrive together. The device-side
alternative is real but buys nothing once the host work is hidden, and it would
put the token ids out of reach of the reference check.

Staging is pinned and reused, so `build` allocates nothing and the copies can
be issued `non_blocking=True` -- a pageable copy would synchronize and undo the
overlap.

Three of the four vectors are *tabulated*, not computed per call. The tokenizer
pads to `prompt_len` and marks a prefix of `n_tokens` valid, so the embedding
scale, the attention mask and the decoder's rotary table are functions of
`n_tokens` alone, which ranges over `[0, prompt_len]`. Computing them per call
put torch's intra-op thread pool on the GPU's critical path, and the wait at its
barrier is unbounded: the slot's wall time was bimodal, 0.34 ms in 96.5% of
forwards and 8 to 16 ms in the rest, which is the whole of Pi0.5's chunk-latency
tail (jobs 599781, 599786). The tables are built by running the same expression
once per reachable count at construction, so what the slot copies is what the
per-call arithmetic produced, bit for bit.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from flash_vla.models.pi05.spec import (
    ENCODER_DIM,
    HEAD_DIM,
    MASK_NEG,
    ROPE_THETA,
    VISION_TOKENS,
)


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
        #: A numpy view of the token buffer, so writing it is one numpy store
        #: rather than a tensor construction and a dispatched copy.
        self._token_view = self.token_ids.numpy()

        # The embedder scales by sqrt(width) (`models/gemma.py:150`, and its
        # PyTorch mirror). Folding it into the same vector that zeroes padding
        # keeps the gather to one multiply.
        self._embed_scale = math.sqrt(ENCODER_DIM)
        self._inv_freq = 1.0 / (ROPE_THETA ** (
            torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        self._offsets = torch.arange(chunk_size, dtype=torch.float32)
        self._scale_table, self._mask_table, self._rope_table = self._tabulate()

        self.n_valid = 0

    # -- construction -------------------------------------------------------

    def _tabulate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three token-count-dependent vectors, for every reachable count.

        `Pi05Tokenizer.encode` pads to `prompt_len` and returns a mask that is
        `n_tokens` ones followed by zeros, so each vector has `prompt_len + 1`
        possible values and nothing else about the state reaches them. Each row
        is produced by the expression the slot used to evaluate per call, so the
        tables are bit-identical to it by construction rather than by argument.

        At this Target's shape the three together are about 5.6 MB.
        """
        counts = range(self.prompt_len + 1)
        scale = torch.empty(len(counts), self.prompt_len, 1, dtype=torch.bfloat16)
        mask = torch.zeros(len(counts), self.cache_len, dtype=torch.bfloat16)
        rope = torch.empty(len(counts), self.chunk_size, HEAD_DIM, dtype=torch.bfloat16)
        valid_mask = np.zeros(self.prompt_len, dtype=bool)
        for n_tokens in counts:
            valid_mask[:n_tokens] = True
            valid_mask[n_tokens:] = False
            scale[n_tokens].copy_(
                torch.from_numpy(valid_mask.astype(np.float32) * self._embed_scale)[:, None])
            n_valid = self.image_tokens + n_tokens
            # Suffix keys are never masked; only the prefix half is written.
            mask[n_tokens, n_valid:self.encoder_seq_len] = MASK_NEG
            # Suffix positions are n_valid + 0..chunk-1 (`models/pi0.py:259`).
            phase = self._inv_freq[None, :] * (self._offsets + n_valid)[:, None]
            rope[n_tokens].copy_(
                torch.stack([torch.cos(phase), torch.sin(phase)], dim=2).view(-1, HEAD_DIM))
        return scale, mask, rope

    # -- per inference ------------------------------------------------------

    def build(self, state) -> int:
        """Tokenize `state` into the staging buffers; return the valid prefix length.

        Tokenization is host arithmetic over 32 table lookups; everything after
        it is a selection from `_tabulate`'s rows and a contiguous copy, so the
        slot performs no elementwise work whose size could reach torch's
        intra-op thread pool.
        """
        if isinstance(state, torch.Tensor):
            state = state.detach().to("cpu", torch.float32).numpy()
        tokens, valid = self.tokenizer.encode(np.asarray(state))
        n_tokens = int(valid.sum())

        self._token_view[:] = tokens
        self.embed_scale.copy_(self._scale_table[n_tokens])
        self.mask_bias.copy_(self._mask_table[n_tokens])
        self.decoder_rope.copy_(self._rope_table[n_tokens])

        self.n_valid = self.image_tokens + n_tokens
        return self.n_valid

    @torch.no_grad()
    def copy_into(self, buffers: dict[str, torch.Tensor], non_blocking: bool = True) -> None:
        """Issue the staged copies onto the current stream."""
        buffers["prompt_ids"].copy_(self.token_ids, non_blocking=non_blocking)
        buffers["prompt_scale"].copy_(self.embed_scale, non_blocking=non_blocking)
        buffers["mask_bias"].copy_(self.mask_bias, non_blocking=non_blocking)
        buffers["action_expert_rope"].copy_(self.decoder_rope, non_blocking=non_blocking)

    @property
    def nbytes(self) -> int:
        """Total bytes copied per inference, for the record."""
        return sum(t.numel() * t.element_size() for t in
                   (self.token_ids, self.embed_scale, self.mask_bias, self.decoder_rope))
