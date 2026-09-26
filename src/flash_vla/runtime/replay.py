"""The replay-time axis of a model, and the row buckets a Target captures it at.

Some lengths change with every inference inside one construction: how many of
a prompt's slots the host slot's tokenizer fills, how much of a padded text
sequence is valid. A model declares that length as its `ReplayAxis`; the graph
is still built at the axis's limit, so every buffer keeps one address.

A Target that runs the axis's rows on kernels whose cost follows them declares
a row granularity (`Target.replay_granularity`). The runner then builds and
captures the stages the axis reaches once per bucket (`replay_buckets`), all
over the same buffers, and replays the bucket each inference's length falls in
(`runtime/runner.py`). Inside a bucket every shape is static: a GEMM plans for
its rows, and a backend whose kernel can run the exact length is told it before
the replay (`Scratch.on_replay`).

A model whose Target captures more than one bucket also accepts the axis's
name as a construction option that fixes it (a synthetic input of that
length, Pi0.5's `prompt_tokens`), so the correctness gates reach every bucket.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class ReplayAxis:
    """A length that changes with every inference.

    `name` is the shape number a graph build reads it from (`at`); the runner
    sets it to a bucket's value, `Target.graph` to `limit` when it builds the
    graph alone. `limit` names the shape number bounding it and `offset` the one
    counting the rows before it (the image tokens ahead of a prompt), `None`
    when it starts at row 0. `slot` is the host slot that decides it, `None`
    when the staged inputs do. `stages` are the stages whose graph follows it.
    `extent(host_state, inputs)` is its value for one inference, read once
    `slot` has run (or the inputs are staged).
    """
    name: str
    limit: str
    offset: str | None
    slot: str | None
    stages: tuple[str, ...]
    extent: Callable[[object, Mapping[str, torch.Tensor]], int]

    def at(self, shape: Mapping[str, int], value: int) -> dict[str, int]:
        """`shape` with this axis at `value`: what a bucket's graph builds and
        its routes resolve at."""
        return {**shape, self.name: value}

    def rows(self, shape: Mapping[str, int], value: int) -> int:
        """The rows `value` of this axis occupies, counted from row 0."""
        return (0 if self.offset is None else shape[self.offset]) + value


def replay_buckets(axis: ReplayAxis, shape: Mapping[str, int],
                   replay_range: tuple[int, int] | None,
                   granularity: int | None) -> tuple[int, ...]:
    """The axis values the stages it reaches are captured at, ascending.

    Each bucket ends on a multiple of `granularity` rows that a length in
    `replay_range` (the workload's, in axis units) rounds up to, and the last is
    the axis limit, which serves any length. Without a range or a granularity
    the limit is the only bucket.
    """
    limit = shape[axis.limit]
    if replay_range is None or granularity is None:
        return (limit,)
    start = axis.rows(shape, 0)
    low, high = (-(-axis.rows(shape, value) // granularity) * granularity for value in replay_range)
    values = {rows - start for rows in range(low, high + 1, granularity) if rows - start < limit}
    return (*sorted(values), limit)


__all__ = ["ReplayAxis", "replay_buckets"]
