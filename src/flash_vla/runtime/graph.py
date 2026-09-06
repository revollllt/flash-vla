"""The explicit computation graph of a Target, and the API that builds it.

A Target writes its forward pass as data: an ordered list of nodes, each one
call site of the op vocabulary (`runtime/ops.py`) with its arguments named
explicitly, over buffers the graph declares and weights the model contract
names. Python loops are construction only -- eighteen layers by ten steps
unroll into as many nodes as the captured CUDA graph has launches -- and
nothing here touches a device, so a graph builds on a login node.

    g.stage("action_expert")
    x = g.buf("action_expert_x", (chunk, dim))
    for step in range(steps):
        for i in range(layers):
            g.op("action_expert_attention", q=q, k=kv_k[i], v=kv_v[i],
                 mask=mask, out=q, prefix_len=prefix_len)

What the graph then gives the runner, without model knowledge: the buffer
plan to materialize; per stage, the node sequence to execute and capture; the
set of call sites a plan must route; per node, the tensors read and written;
and the minimal traffic and math of every call site (`costs`), derived from
the op specs and the argument shapes.

References are immutable descriptions, not tensors. `BufRef` supports the two
operations the pipelines need on a buffer -- leading-axis indexing / slicing
and `view` -- and `WeightRef` integer indexing; both resolve against the real
tensors once, after allocation. Every op writes its outputs in place; the
graph never carries a value an op returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Mapping, Sequence

import torch

from .cost import Cost, Invocation, SegmentCosts
from .cuda.arena import Buffer, Init
from .cuda.program import Step
from .ops import OpSpec, Vocabulary


def _index_shape(shape: tuple[int, ...], key: Any) -> tuple[int, ...]:
    """The shape of `tensor[key]` for integer and slice keys over leading axes."""
    keys = key if isinstance(key, tuple) else (key,)
    if len(keys) > len(shape):
        raise IndexError(f"too many indices for shape {shape}: {keys}")
    out: list[int] = []
    for k, n in zip(keys, shape):
        if isinstance(k, bool):
            raise TypeError("boolean indices are not supported on a reference")
        if isinstance(k, int):
            if not -n <= k < n:
                raise IndexError(f"index {k} out of range for extent {n}")
            continue
        if isinstance(k, slice):
            out.append(len(range(*k.indices(n))))
            continue
        raise TypeError(f"unsupported index {k!r} on a reference; use ints and slices")
    return tuple(out) + tuple(shape[len(keys):])


def _view_shape(shape: tuple[int, ...], new: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve one `-1` in `new` against the element count of `shape`."""
    total = prod(shape)
    if new.count(-1) > 1:
        raise ValueError(f"only one dimension may be -1 in a view, got {new}")
    if -1 in new:
        known = prod(d for d in new if d != -1)
        if known == 0 or total % known:
            raise ValueError(f"cannot view {shape} as {new}")
        new = tuple(total // known if d == -1 else d for d in new)
    if prod(new) != total:
        raise ValueError(f"cannot view {shape} ({total} elements) as {new}")
    return new


@dataclass(frozen=True, eq=False)
class BufRef:
    """A view of a declared buffer: its name and the index / view steps applied."""
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    steps: tuple[tuple[str, Any], ...] = ()

    def __getitem__(self, key: Any) -> "BufRef":
        return replace(self, shape=_index_shape(self.shape, key),
                       steps=self.steps + (("index", key),))

    def view(self, *shape: int) -> "BufRef":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        new = _view_shape(self.shape, tuple(shape))
        return replace(self, shape=new, steps=self.steps + (("view", new),))

    @property
    def numel(self) -> int:
        return prod(self.shape)

    def resolve(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The tensor this reference denotes, given the materialized buffers."""
        tensor = tensors[self.name]
        for kind, arg in self.steps:
            tensor = tensor[arg] if kind == "index" else tensor.view(*arg)
        return tensor

    def __repr__(self) -> str:
        return f"BufRef({self.name}, {self.shape}, steps={len(self.steps)})"


@dataclass(frozen=True, eq=False)
class WeightRef:
    """One model weight, optionally indexed by leading integers (a layer, a step)."""
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    index: tuple[int, ...] = ()

    def __getitem__(self, key: Any) -> "WeightRef":
        keys = key if isinstance(key, tuple) else (key,)
        if not all(isinstance(k, int) and not isinstance(k, bool) for k in keys):
            raise TypeError(f"weights are indexed by integers only, got {keys}")
        if len(keys) > len(self.shape):
            raise IndexError(f"too many indices for weight {self.name} of shape {self.shape}")
        for k, n in zip(keys, self.shape):
            if not -n <= k < n:
                raise IndexError(f"index {k} out of range for weight {self.name} extent {n}")
        return replace(self, shape=tuple(self.shape[len(keys):]), index=self.index + tuple(keys))

    @property
    def numel(self) -> int:
        return prod(self.shape)

    def resolve(self, weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor = weights[self.name]
        return tensor[self.index] if self.index else tensor

    def __repr__(self) -> str:
        return f"WeightRef({self.name}{list(self.index) if self.index else ''}, {self.shape})"


Ref = BufRef | WeightRef


@dataclass(frozen=True, eq=False)
class Node:
    """One launch site of the graph: a call site with its ordered arguments."""
    index: int
    stage: str
    call_site: str
    args: tuple[Any, ...]

    @property
    def is_copy(self) -> bool:
        return self.call_site == COPY

    def refs(self) -> tuple[Ref, ...]:
        return tuple(a for a in self.args if isinstance(a, (BufRef, WeightRef)))


#: The one node kind that is not an op: `dst.copy_(src)` between two references.
COPY = "copy"


class Graph:
    """An ordered list of stages, host slots, nodes and buffer declarations."""

    def __init__(self, vocabulary: Vocabulary, weight_shapes: Mapping[str, Sequence[int]],
                 stages: Sequence[str], weight_dtype: torch.dtype = torch.bfloat16) -> None:
        self.vocabulary = vocabulary
        self.weight_shapes = {name: tuple(shape) for name, shape in weight_shapes.items()}
        self.weight_dtype = weight_dtype
        self.stages = tuple(stages)
        self.buffers: dict[str, Buffer] = {}
        self._refs: dict[str, BufRef] = {}
        self.nodes: list[Node] = []
        self.program: list[Step] = []
        self.derived: dict[str, int] = {}
        self._current: str | None = None

    # -- structure ----------------------------------------------------------

    def stage(self, name: str) -> None:
        """Open stage `name`; nodes declared until the next `stage` or `host` belong to it."""
        if name not in self.stages:
            raise KeyError(f"stage {name!r} is not one of {self.stages}")
        if any(step.name == name for step in self.program):
            raise ValueError(f"stage {name!r} opened twice")
        opened = [step.name for step in self.program if step.kind == "segment"]
        if self.stages.index(name) != len(opened):
            raise ValueError(f"stages must open in order {self.stages}; got {name!r} after {opened}")
        self.program.append(Step("segment", name))
        self._current = name

    def host(self, name: str) -> None:
        """Declare a host slot at this point of the program, between two stages."""
        if any(step.name == name for step in self.program):
            raise ValueError(f"host slot {name!r} declared twice")
        self.program.append(Step("host", name))
        self._current = None

    @property
    def segment_names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.program if step.kind == "segment")

    @property
    def host_slots(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.program if step.kind == "host")

    # -- declarations -------------------------------------------------------

    def buf(self, name: str, shape: Sequence[int] | None = None,
            dtype: torch.dtype = torch.bfloat16, init: Init = "empty",
            view: tuple[slice, ...] | None = None, alias: str | None = None) -> BufRef:
        """Declare buffer `name`, or fetch its reference when already declared.

        `shape` is the allocation, padding included; `view` the region the
        pipeline sees; `alias` names another declaration whose allocation this
        one views (shape and dtype are then taken from it). A second
        declaration must repeat the first exactly.
        """
        if name in self._refs:
            if shape is not None or view is not None or alias is not None:
                spec = self.buffers[name]
                same = (spec.alias == alias and spec.view == view
                        and (alias is not None or tuple(spec.shape) == tuple(shape or ()))
                        and (alias is not None or spec.dtype == dtype))
                if not same:
                    raise ValueError(f"buffer {name!r} redeclared with a different spec")
            return self._refs[name]
        if alias is not None:
            if alias not in self.buffers or self.buffers[alias].alias is not None:
                raise KeyError(f"buffer {name!r} aliases {alias!r}, which is not an allocation")
            base = self.buffers[alias]
            spec = Buffer(alias=alias, view=view)
            exposed = Buffer(shape=base.shape, view=view).exposed_shape()
            dtype = base.dtype
        else:
            if shape is None:
                raise ValueError(f"buffer {name!r} is not declared; give its shape")
            spec = Buffer(shape=tuple(shape), dtype=dtype, init=init, view=view)
            exposed = spec.exposed_shape()
        self.buffers[name] = spec
        self._refs[name] = BufRef(name, tuple(exposed), dtype)
        return self._refs[name]

    def w(self, name: str) -> WeightRef:
        """A reference to model weight `name`."""
        if name not in self.weight_shapes:
            raise KeyError(f"unknown weight {name!r}; the model contract names "
                           f"{len(self.weight_shapes)} tensors")
        return WeightRef(name, self.weight_shapes[name], self.weight_dtype)

    # -- nodes --------------------------------------------------------------

    def op(self, call_site: str, **kwargs: Any) -> Node:
        """Append one call of `call_site`; arguments by the spec's parameter names."""
        if self._current is None:
            raise RuntimeError(f"op {call_site!r} declared outside a stage")
        spec = self.vocabulary[call_site]
        unknown = set(kwargs) - set(spec.params)
        if unknown:
            raise TypeError(f"{call_site}: unknown parameters {sorted(unknown)}; "
                            f"expected {spec.params}")
        args = tuple(kwargs.get(param) for param in spec.params)
        for param in spec.outputs:
            if not isinstance(kwargs.get(param), BufRef):
                raise TypeError(f"{call_site}: output {param!r} must be a buffer reference")
        for param, value in kwargs.items():
            if not isinstance(value, (BufRef, WeightRef, int, float, type(None))):
                raise TypeError(f"{call_site}: argument {param!r} must be a reference or a "
                                f"scalar, got {type(value).__name__}")
        node = Node(len(self.nodes), self._current, call_site, args)
        self.nodes.append(node)
        return node

    def copy(self, dst: BufRef, src: Ref) -> Node:
        """Append a `dst.copy_(src)` node."""
        if self._current is None:
            raise RuntimeError("copy declared outside a stage")
        if not isinstance(dst, BufRef) or not isinstance(src, (BufRef, WeightRef)):
            raise TypeError("copy takes a buffer destination and a buffer or weight source")
        if dst.shape != src.shape:
            raise ValueError(f"copy shape mismatch: {dst.shape} <- {src.shape}")
        node = Node(len(self.nodes), self._current, COPY, (dst, src))
        self.nodes.append(node)
        return node

    # -- queries ------------------------------------------------------------

    def nodes_of(self, stage: str) -> list[Node]:
        return [node for node in self.nodes if node.stage == stage]

    @property
    def call_sites(self) -> tuple[str, ...]:
        """Distinct op call sites in first-use order (copies excluded)."""
        seen: dict[str, None] = {}
        for node in self.nodes:
            if not node.is_copy:
                seen.setdefault(node.call_site, None)
        return tuple(seen)

    def reads_writes(self, node: Node) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Buffer names a node reads and writes, from its spec."""
        if node.is_copy:
            dst, src = node.args
            return ((src.name,) if isinstance(src, BufRef) else ()), (dst.name,)
        spec = self.vocabulary[node.call_site]
        outputs = set(spec.outputs)
        reads = tuple(a.name for p, a in zip(spec.params, node.args)
                      if isinstance(a, BufRef) and (p not in outputs or p in spec.inout))
        writes = tuple(a.name for p, a in zip(spec.params, node.args)
                       if isinstance(a, BufRef) and p in outputs)
        return reads, writes

    def node_cost(self, node: Node) -> Cost:
        if node.is_copy:
            dst, src = node.args
            return Cost(bytes_read=src.numel * src.dtype.itemsize,
                        bytes_written=dst.numel * dst.dtype.itemsize, flops=0)
        spec = self.vocabulary[node.call_site]
        shapes = {p: (a.shape if isinstance(a, (BufRef, WeightRef)) else None)
                  for p, a in zip(spec.params, node.args)}
        sizes = {p: (a.dtype.itemsize if isinstance(a, (BufRef, WeightRef)) else 0)
                 for p, a in zip(spec.params, node.args)}
        return spec.cost(shapes, sizes)

    def costs(self) -> SegmentCosts:
        """Per stage, one `Invocation` per call site: its per-call cost and count.

        A call site whose calls differ in cost (a bisected leading row) reports
        the mean per-call cost so that count times cost stays the total.
        """
        out: dict[str, list[Invocation]] = {}
        for stage in self.segment_names:
            groups: dict[str, list[Cost]] = {}
            for node in self.nodes_of(stage):
                groups.setdefault(node.call_site, []).append(self.node_cost(node))
            rows = []
            for call_site, costs in groups.items():
                n = len(costs)
                mean = Cost(bytes_read=sum(c.bytes_read for c in costs) // n,
                            bytes_written=sum(c.bytes_written for c in costs) // n,
                            flops=sum(c.flops for c in costs) // n)
                rows.append(Invocation(call_site, mean, n))
            out[stage] = rows
        return out

    def check(self) -> None:
        """Raise on a graph that cannot run: empty stages, undeclared outputs, bad weights."""
        if not self.segment_names:
            raise ValueError("the graph declares no stage")
        for stage in self.segment_names:
            if not self.nodes_of(stage):
                raise ValueError(f"stage {stage!r} has no nodes")
        for node in self.nodes:
            for ref in node.refs():
                if isinstance(ref, BufRef) and ref.name not in self.buffers:
                    raise KeyError(f"node {node.index} {node.call_site} uses undeclared "
                                   f"buffer {ref.name!r}")
                if isinstance(ref, WeightRef) and ref.name not in self.weight_shapes:
                    raise KeyError(f"node {node.index} {node.call_site} uses unknown "
                                   f"weight {ref.name!r}")


__all__ = ["COPY", "BufRef", "Graph", "Node", "Ref", "WeightRef"]
