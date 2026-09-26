"""Pi0.5 expert attention over the valid keys only, split along the keys.

The action expert's chunk (`[chunk * 8, 256]` queries, a token's 8 heads
adjacent) attends multi-query over one KV head: the prefix's first `n_valid`
rows and the chunk's own rows, which sit past the whole prefix in the cache
(`[prefix_len, cache_len)`); the additive mask is `MASK_NEG` on the rows between.
Only a few hundred query rows meet under a thousand keys, so a kernel that
walks the keys per query tile leaves most of the GPU idle. The work is split as
FlashInfer's split-KV prefill splits it, down to one block of keys per CTA:

- `attend_block`: a tile of queries against one block of `BLOCK_N` valid keys,
  read through the gap in the cache; it writes the block's softmax-normalized
  output (bf16) and its row maximum and sum.
- `merge_blocks`: every block's partial for a few query rows, loaded at once
  and weighted by `exp(block max - largest block max) * block sum`.

The valid length is a device scalar the replay hook writes before each replay
(`Scratch.on_replay`), so a graph captured once runs exactly the keys of the
inference it replays; blocks past them exit, and the mask is not read.

Tiles are the fastest of `lab/pi05/attention_tile_sweep.py` at each workload's
chunk: 16 query rows at LIBERO's 80 queries, 32 at RoboDojo's 400, keys by 32
and 4 warps at both. Only those two query counts were swept; `WIDE_QUERIES`
splits them.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flash_vla.models.pi05.spec import HEAD_DIM
from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch

NAMES = frozenset({"action_expert_attention"})
BLOCK_N = 32
WARPS = 4
MERGE_ROWS = 2
MERGE_WARPS = 2
#: The device scalar holding this replay's valid prefix rows.
VALID_ROWS = "pi05_expert_valid_rows"
#: Query rows past which the 32-row tile is used: between the two swept counts.
WIDE_QUERIES = 128


@triton.jit
def attend_block(Q, K, V, Valid, Partial, Stats, scale, QUERIES: tl.constexpr,
                 PREFIX: tl.constexpr, CHUNK: tl.constexpr, BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr, HEAD: tl.constexpr):
    """Query tile `program_id(0)` over valid key block `program_id(1)`: the
    normalized bf16 output, and the rows' maximum and sum of exponentials."""
    block = tl.program_id(1)
    valid = tl.load(Valid)
    start = block * BLOCK_N
    if start >= valid + CHUNK:
        return
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    inside = rows < QUERIES
    dims = tl.arange(0, HEAD)
    logical = start + tl.arange(0, BLOCK_N)
    keep = logical < valid + CHUNK
    # Logical keys [0, valid) are the prefix's, [valid, valid + CHUNK) the chunk's.
    physical = tl.where(logical < valid, logical, logical - valid + PREFIX)
    q = tl.load(Q + rows[:, None] * HEAD + dims[None, :], mask=inside[:, None], other=0.0)
    k = tl.load(K + physical[:, None] * HEAD + dims[None, :], mask=keep[:, None], other=0.0)
    v = tl.load(V + physical[:, None] * HEAD + dims[None, :], mask=keep[:, None], other=0.0)
    logits = tl.where(keep[None, :], tl.dot(q, tl.trans(k)) * scale, -1.0e30)
    peak = tl.max(logits, 1)
    weights = tl.exp(logits - peak[:, None])
    total = tl.sum(weights, 1)
    out = tl.dot(weights.to(tl.bfloat16), v) / total[:, None]
    slot = block * QUERIES + rows
    tl.store(Partial + slot[:, None] * HEAD + dims[None, :], out.to(tl.bfloat16),
             mask=inside[:, None])
    tl.store(Stats + slot * 2, peak, mask=inside)
    tl.store(Stats + slot * 2 + 1, total, mask=inside)


@triton.jit
def merge_blocks(Partial, Stats, Valid, Out, QUERIES: tl.constexpr, CHUNK: tl.constexpr,
                 BLOCKS: tl.constexpr, BLOCK_N: tl.constexpr, ROWS: tl.constexpr,
                 HEAD: tl.constexpr):
    """ROWS query rows: every used block's partial in one load, merged by weight."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    blocks = tl.arange(0, BLOCKS)
    used = tl.cdiv(tl.load(Valid) + CHUNK, BLOCK_N)
    keep = (blocks[:, None] < used) & (rows[None, :] < QUERIES)
    slot = blocks[:, None] * QUERIES + rows[None, :]
    peaks = tl.load(Stats + slot * 2, mask=keep, other=-1.0e30)
    totals = tl.load(Stats + slot * 2 + 1, mask=keep, other=0.0)
    weights = tl.exp(peaks - tl.max(peaks, 0)[None, :]) * totals
    weights = weights / tl.sum(weights, 0)[None, :]
    dims = tl.arange(0, HEAD)
    partials = tl.load(Partial + slot[:, :, None] * HEAD + dims[None, None, :],
                       mask=keep[:, :, None], other=0.0).to(tl.float32)
    tl.store(Out + rows[:, None] * HEAD + dims[None, :],
             tl.sum(weights[:, :, None] * partials, 0).to(tl.bfloat16),
             mask=(rows < QUERIES)[:, None])


def make_wrappers(scratch: Scratch, selected_names: frozenset[str] | None = None
                  ) -> dict[str, Wrapper]:
    """The attention wrapper, and the replay hook that writes the valid prefix
    rows it reads (allocated by the first hook, a bucket's preparation before
    warmup, so declaring the wrapper touches no device)."""

    def write_valid_rows(valid_rows: int, bucket_rows: int) -> None:
        scratch(VALID_ROWS, (1,), torch.int32, scratch.device).fill_(valid_rows)

    scratch.on_replay(write_valid_rows)

    def action_expert_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                mask: torch.Tensor, out: torch.Tensor,
                                prefix_len: int) -> torch.Tensor:
        queries = Q.shape[0]
        valid = scratch(VALID_ROWS, (1,), torch.int32, scratch.device)
        chunk = K.shape[0] - prefix_len
        blocks = triton.cdiv(K.shape[0], BLOCK_N)
        block_m = 32 if queries > WIDE_QUERIES else 16
        partial = scratch("pi05_expert_partial", (blocks, queries, HEAD_DIM), Q.dtype, Q.device)
        stats = scratch("pi05_expert_partial_stats", (blocks, queries, 2), torch.float32, Q.device)
        attend_block[(triton.cdiv(queries, block_m), blocks)](
            Q, K, V, valid, partial, stats, HEAD_DIM ** -0.5, queries, prefix_len, chunk,
            block_m, BLOCK_N, HEAD_DIM, num_warps=WARPS, num_stages=1)
        # `out` may be `Q`: every block has read its queries before the merge writes.
        merge_blocks[(triton.cdiv(queries, MERGE_ROWS),)](
            partial, stats, valid, out, queries, chunk, triton.next_power_of_2(blocks), BLOCK_N,
            MERGE_ROWS, HEAD_DIM, num_warps=MERGE_WARPS)
        return out

    wrappers = {"action_expert_attention": action_expert_attention}
    return {name: wrappers[name] for name in (NAMES if selected_names is None else selected_names)}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=NAMES, make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
