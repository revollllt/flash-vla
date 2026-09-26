"""The minimal work of a model's forward, read from its reference: FLOPs and DRAM bytes.

    python -m measurement.work --target rtx5090/pi05 --workload libero

The model's end-to-end reference (`models/<model>/reference.py`) runs on the
meta device at a workload's shape -- no GPU, no weights -- under a dispatch
mode that records every aten op: its FLOPs (`torch.utils.flop_counter`'s
formulas, which count tensor-core work: matmuls, convolutions, attention) and
the tensors it reads and writes. The model's work table
(`ModelDefinition.work_rules`, declared in `models/<model>/work.py`)
attributes each op to a runtime call site; the Target's graph gives the call
site's stage.

Only necessary work counts:

- an op whose result never reaches the model's output is dead (the last
  backbone layer's MLP, whose output no KV depends on);
- an op that reads no inference input is constant (the flow schedule's
  timestep MLP): it folds away, and its result is read like a weight;
- an op that only rearranges data (`MOVES`: the concatenation of a cached
  prefix with a step's own keys, a dtype cast) moves nothing in a kernel that
  reads its sources directly: reading its result reads the sources it came from;
- an op repeated on unchanged inputs (a loop invariant the reference recomputes
  every denoising step, such as Pi0's state token or the expert's masks) is
  computed once.

Tensors are keyed by the first tensor object that held their storage in the
trace (a parameter, an input, an op's output): a view shares its base's key,
an in-place op writes its argument's. Bytes are counted per distinct view and
never beyond the base.

Two bounds, by where intermediates live (`KernelBound`):

  launch_kernel_bound  every call-site invocation is one kernel launch that reads
                       its inputs from and writes its outputs to DRAM; the time
                       is the sum over launches
  flow_kernel_bound    every pass of a stage is one kernel whose intermediates
                       stay on chip as flow: only weights, stage boundaries and
                       state carried between passes (a KV cache) touch DRAM, and a
                       repeated pass rereads what the last one read except for
                       what the L2 holds; this is the floor

A launch ends where its call site changes or where it reads a parameter it
already read -- the next denoising step inside a monolithic call site. A flow
pass is a run of one stage's launches, ending where a launch rereads a
parameter the pass read; every pass is therefore a union of launches, and the
flow bound never exceeds the launch bound.

Each invocation costs max(bytes / DRAM bandwidth, FLOPs / tensor peak of its
format). Floating tensors are priced at the Target's precision; a quantization
recipe prices its call sites' matmul weights, and the activations they pass
each other, at its own formats (`runtime.vla.Pricing`). The reference's own
representation shows through where it is heavier than necessary: an additive
attention mask is materialized and read by every layer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from enum import Enum
import json
import math
import re
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry

from flash_vla.hardware.nvidia import HARDWARE_ROOFLINES
from flash_vla.hardware.roofline import Roofline
from flash_vla.inference import build_runner, get_target
from flash_vla.runtime.vla import DTYPES, ConfigValue, Pricing, Target
from flash_vla.runtime.work import CallSiteRule, ReferenceRun
from measurement.cli import WORKLOAD_HELP
from measurement.constants import ConstantRow

#: Ops that gather rows of their first tensor: they read as many of its elements as they write.
GATHERS = frozenset({"aten.embedding", "aten.index_select", "aten.index", "aten.gather"})
#: Ops that overwrite their mutated argument without reading it.
OVERWRITES = frozenset({"aten.copy_", "aten.fill_", "aten.zero_"})
#: A reshape that is not a schema view but moves no data.
ALIASES = frozenset({"aten._unsafe_view"})
#: Ops that only rearrange, repeat or recast their inputs' elements.
MOVES = frozenset({"aten.cat", "aten.stack", "aten.repeat", "aten.clone", "aten._to_copy",
                   "aten.constant_pad_nd"})

#: A view of a storage: its offset, shape and strides.
View = tuple[int, tuple[int, ...], tuple[int, ...]]


class KernelBound(Enum):
    """Where a bound keeps intermediates: through DRAM per launch, or on chip per stage pass."""
    LAUNCH = "launch"
    FLOW = "flow"


@dataclass(frozen=True)
class Access:
    """One read or write of a view of a storage: the storage's key and element
    count (`extent`), the view and how many of its elements are touched, their
    element size and whether they are floating (priced at the Target's
    precision), and whether the tensor is a matrix (a quantizable operand)."""
    key: int
    view: View
    elements: int
    extent: int
    itemsize: int
    floating: bool
    matrix: bool


@dataclass(frozen=True)
class TracedOp:
    """One recorded aten op: the innermost module running it, its FLOPs, its
    accesses, whether it reads an inference input (`dynamic`) and whether its
    result reaches an output (`live`)."""
    module: str
    op: str
    flops: int
    reads: tuple[Access, ...]
    writes: tuple[Access, ...]
    dynamic: bool
    live: bool = False

    @property
    def counted(self) -> bool:
        return self.live and self.dynamic


@dataclass(frozen=True)
class Invocation:
    """One launch (launch bound) or stage pass (flow bound) and its DRAM traffic;
    `flops` is keyed by tensor-core format."""
    stage: str
    call_sites: tuple[str, ...]
    flops: Mapping[str, int]
    bytes_read: float
    bytes_written: float

    def seconds(self, roofline: Roofline) -> float:
        """max(bytes / DRAM bandwidth, FLOPs / tensor peak), at datasheet rates."""
        compute = sum(count / roofline.tensor_peaks[fmt].flops_per_second
                      for fmt, count in self.flops.items())
        return max((self.bytes_read + self.bytes_written) / roofline.dram_bytes_per_second, compute)


@dataclass
class Recorder(TorchDispatchMode):
    """Records every aten op dispatched while active, with the module stack the
    forward hooks maintain (see the module docstring for keys, moves and
    repeats)."""
    dynamic_keys: set[int]
    parameter_keys: set[int]
    modules: list[str] = field(default_factory=list)
    keys: dict[int, int] = field(default_factory=dict)
    extents: dict[int, int] = field(default_factory=dict)
    #: A moved tensor's key -> what reading all of it reads.
    sources: dict[int, tuple[Access, ...]] = field(default_factory=dict)
    #: Writes so far to each key, so a repeat is recognised only on unchanged tensors.
    versions: dict[int, int] = field(default_factory=dict)
    #: An op's signature -> its outputs' keys and their versions when it ran.
    computed: dict[tuple[object, ...], tuple[tuple[int, int], ...]] = field(default_factory=dict)
    # Every seen tensor stays referenced, so no `id` is reused during a trace.
    seen: list[torch.Tensor] = field(default_factory=list)
    ops: list[TracedOp] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()

    def key(self, tensor: torch.Tensor) -> int:
        if id(tensor) in self.keys:
            return self.keys[id(tensor)]
        self.keys[id(tensor)] = id(tensor)
        self.extents[id(tensor)] = tensor.numel()
        self.seen.append(tensor)
        return id(tensor)

    def fresh(self, tensor: torch.Tensor, *, dynamic: bool) -> int:
        """Key a new storage an op produced."""
        self.keys[id(tensor)] = id(tensor)
        self.extents[id(tensor)] = tensor.numel()
        self.seen.append(tensor)
        self.dynamic_keys.update({id(tensor)} if dynamic else set())
        return id(tensor)

    def own(self, tensor: torch.Tensor, elements: int | None = None) -> Access:
        """`tensor`'s own storage; a broadcast view (stride 0) holds only its distinct elements."""
        distinct = math.prod(size for size, stride in zip(tensor.shape, tensor.stride()) if stride != 0)
        key = self.key(tensor)
        return Access(key, (tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride())),
                      distinct if elements is None else elements, self.extents[key],
                      tensor.element_size(), tensor.is_floating_point(), tensor.dim() >= 2)

    def accesses(self, tensor: torch.Tensor, elements: int | None = None) -> tuple[Access, ...]:
        """Reading `elements` of `tensor` (all of them by default): its own storage,
        or, of each source a moved tensor came from, at most as many elements --
        exact for a cast or a copy, at worst over for a slice of a concatenation."""
        read = self.own(tensor, elements)
        if read.key not in self.sources:
            return (read,)
        return tuple(replace(piece, elements=min(piece.elements, read.elements))
                     for piece in self.sources[read.key])

    def __torch_dispatch__(self, func: torch._ops.OpOverload, types: tuple[type, ...],
                           args: tuple[object, ...] = (),
                           kwargs: dict[str, object] | None = None) -> object:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = str(func.overloadpacket)
        leaves, structure = tree_flatten((args, kwargs))
        inputs = [t for t in leaves if isinstance(t, torch.Tensor)]
        outputs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        if func.is_view or name in ALIASES:
            for tensor in outputs:
                self.keys[id(tensor)] = self.key(inputs[0])
                self.seen.append(tensor)
            return out
        mutated = [args[index] if index < len(args) else kwargs[argument.name]
                   for index, argument in enumerate(func._schema.arguments)
                   if argument.alias_info is not None and argument.alias_info.is_write
                   and (index < len(args) or argument.name in kwargs)]
        mutated_ids = {id(t) for t in mutated}
        read_tensors = [t for t in inputs if id(t) not in mutated_ids or name not in OVERWRITES]
        gathered = sum(t.numel() for t in outputs)
        reads = tuple(access for index, t in enumerate(read_tensors)
                      for access in self.accesses(t, gathered if name in GATHERS and index == 0
                                                  else None))
        dynamic = any(access.key in self.dynamic_keys for access in reads)
        # The overload, the argument structure, and every tensor's version, view and dtype.
        signature = (func, repr(structure), *(
            (self.key(t), self.versions.get(self.key(t), 0), tuple(t.shape), t.stride(),
             t.storage_offset(), t.dtype) if isinstance(t, torch.Tensor) else repr(t)
            for t in leaves))
        repeatable = not mutated and torch.Tag.nondeterministic_seeded not in func.tags
        earlier = self.computed.get(signature, ()) if repeatable else ()
        if earlier and all(self.versions.get(key, 0) == version for key, version in earlier):
            for tensor, (key, _version) in zip(outputs, earlier):
                self.keys[id(tensor)] = key
                self.seen.append(tensor)
            return out
        if name in MOVES:
            pieces = tuple(dict.fromkeys(reads))
            keys = tuple(self.fresh(tensor, dynamic=dynamic) for tensor in outputs)
            self.sources.update({key: pieces for key in keys})
            self.computed[signature] = tuple((key, 0) for key in keys)
            return out
        created = [tensor for tensor in outputs if id(tensor) not in mutated_ids]
        keys = tuple(self.fresh(tensor, dynamic=dynamic) for tensor in created)
        writes = tuple(self.own(t) for t in (*mutated, *created))
        for tensor, written in zip(mutated, writes):
            self.versions[written.key] = self.versions.get(written.key, 0) + 1
            self.dynamic_keys.update({written.key} if dynamic else set())
            # A moved buffer written in place now also holds what was written.
            moved = self.sources.get(written.key)
            self.sources.update({} if moved is None else {written.key: (*moved, written)})
        # An operand gathered from a parameter (a per-embodiment weight) is one.
        self.parameter_keys.update(keys if name in GATHERS and reads[0].key in self.parameter_keys
                                   else ())
        self.computed.update({signature: tuple((key, 0) for key in keys)} if repeatable else {})
        formula = flop_registry.get(func.overloadpacket)
        flops = 0 if formula is None else int(formula(*args, **kwargs, out_val=out))
        self.ops.append(TracedOp(self.modules[-1] if self.modules else "", name, flops,
                                 reads, writes, dynamic))
        return out


def trace(run: ReferenceRun) -> tuple[tuple[TracedOp, ...], frozenset[int], frozenset[int]]:
    """Every op of `run`'s forward, the keys of its outputs and of the model's
    parameters (and operands gathered from them), liveness marked."""
    parameter_keys = {id(parameter) for parameter in run.model.parameters()}
    recorder = Recorder(dynamic_keys={id(t) for t in run.inputs}, parameter_keys=parameter_keys)

    def leave(_module: nn.Module, _args: tuple[object, ...], _out: object) -> None:
        # A forward hook that returns a value replaces the module's output.
        recorder.modules.pop()

    hooks = []
    for path, module in run.model.named_modules():
        hooks.append(module.register_forward_pre_hook(
            lambda _module, _args, path=path: recorder.modules.append(path)))
        hooks.append(module.register_forward_hook(leave))
    try:
        with recorder:
            outputs = run.forward()
    finally:
        for hook in hooks:
            hook.remove()
    output_keys = frozenset(access.key for t in outputs for access in recorder.accesses(t))
    needed = set(output_keys)
    ops = list(recorder.ops)
    for index in range(len(ops) - 1, -1, -1):
        if not any(access.key in needed for access in ops[index].writes):
            continue
        ops[index] = replace(ops[index], live=True)
        needed.update(access.key for access in ops[index].reads)
    return tuple(ops), output_keys, frozenset(recorder.parameter_keys)


def call_sites(ops: Sequence[TracedOp], rules: Sequence[CallSiteRule]) -> tuple[str | None, ...]:
    """The call site of every op, `None` for an op that does not count."""
    compiled = [(re.compile(rule.module), None if rule.op is None else re.compile(rule.op),
                 rule.call_site) for rule in rules]
    assigned: list[str | None] = []
    current: str | None = None
    for op in ops:
        if not op.counted:
            assigned.append(None)
            continue
        matched = next((site for module, name, site in compiled
                        if module.search(op.module) and (name is None or name.fullmatch(op.op))),
                       current)
        if matched is None:
            raise ValueError(f"no work rule covers the first counted op {op.op} in {op.module!r}")
        current = matched
        assigned.append(matched)
    return tuple(assigned)


def invocations(ops: Sequence[TracedOp], sites: Sequence[str | None], output_keys: frozenset[int],
                parameter_keys: frozenset[int], *, bound: KernelBound, stage_of: Mapping[str, str],
                precision: str, pricing: Pricing | None, l2_bytes: int) -> tuple[Invocation, ...]:
    """The launches (`LAUNCH`) or stage passes (`FLOW`) of a traced forward, with
    their FLOPs and the DRAM bytes each must read and write."""
    precision_itemsize = DTYPES[precision].itemsize
    recipe_sites = frozenset() if pricing is None else pricing.call_sites

    def parameters_read(index: int) -> set[int]:
        return {access.key for access in ops[index].reads if access.key in parameter_keys}

    launches: list[list[int]] = []
    launch_parameters: set[int] = set()
    for index, site in enumerate(sites):
        if site is None:
            continue
        read = parameters_read(index)
        if not launches or site != sites[launches[-1][-1]] or read & launch_parameters:
            launches.append([index])
            launch_parameters = read
        else:
            launches[-1].append(index)
            launch_parameters |= read
    groups: list[list[int]] = []
    pass_parameters: set[int] = set()
    for launch in launches:
        read = set().union(*(parameters_read(index) for index in launch))
        stage = stage_of[sites[launch[0]]]
        if (bound is KernelBound.LAUNCH or not groups
                or stage != stage_of[sites[groups[-1][0]]] or read & pass_parameters):
            groups.append(list(launch))
            pass_parameters = read
        else:
            groups[-1].extend(launch)
            pass_parameters |= read

    writer: dict[int, int] = {}
    writer_site: dict[int, str] = {}
    written: dict[tuple[int, int], dict[View, float]] = {}
    reads: list[dict[int, dict[View, float]]] = [{} for _ in groups]
    stored: list[dict[int, dict[View, float]]] = [{} for _ in groups]
    extents: dict[int, float] = {}
    for group, members in enumerate(groups):
        for index in members:
            op, site = ops[index], sites[index]
            for access in op.reads:
                source = writer.get(access.key)
                if source == group:
                    continue
                # An activation one recipe call site passes another is in its format;
                # of the weights, only a matmul's (a matrix operand) is quantized.
                passed = (access.floating and site in recipe_sites
                          and writer_site.get(access.key) in recipe_sites)
                quantized = (access.floating and site in recipe_sites and op.flops > 0
                             and access.matrix and access.key in parameter_keys)
                itemsize = (pricing.weight_itemsize if quantized
                            else pricing.activation_itemsize if passed
                            else precision_itemsize if access.floating
                            else access.itemsize)
                views = reads[group].setdefault(access.key, {})
                views[access.view] = max(views.get(access.view, 0), access.elements * itemsize)
                extents[access.key] = max(extents.get(access.key, 0), access.extent * itemsize)
                if source is None:
                    continue
                ratio = pricing.activation_itemsize / precision_itemsize if passed else 1.0
                kept = stored[source].setdefault(access.key, {})
                for view, size in written[(source, access.key)].items():
                    kept[view] = max(kept.get(view, 0), size * ratio)
            for access in op.writes:
                itemsize = precision_itemsize if access.floating else access.itemsize
                writer[access.key] = group
                writer_site[access.key] = site
                views = written.setdefault((group, access.key), {})
                views[access.view] = max(views.get(access.view, 0), access.elements * itemsize)
                extents[access.key] = max(extents.get(access.key, 0), access.extent * itemsize)
    for key in output_keys & writer.keys():
        stored[writer[key]][key] = dict(written[(writer[key], key)])

    def total(per_key: Mapping[int, Mapping[View, float]]) -> float:
        """Distinct views summed, never beyond their storage."""
        return sum(min(sum(views.values()), extents[key]) for key, views in per_key.items())

    result = []
    previous_reads: dict[str, dict[int, Mapping[View, float]]] = {}
    for group, members in enumerate(groups):
        stage = stage_of[sites[members[0]]]
        # A repeated pass finds up to the L2's capacity of what the last pass read.
        before = previous_reads.get(stage, {}) if bound is KernelBound.FLOW else {}
        reused = total({key: views for key, views in reads[group].items() if key in before})
        previous_reads[stage] = reads[group]
        flops: dict[str, int] = {}
        for index in members:
            fmt = pricing.tensor if sites[index] in recipe_sites else precision
            flops[fmt] = flops.get(fmt, 0) + ops[index].flops
        result.append(Invocation(
            stage=stage, call_sites=tuple(dict.fromkeys(sites[index] for index in members)),
            flops=flops, bytes_read=total(reads[group]) - min(reused, l2_bytes),
            bytes_written=total(stored[group])))
    return tuple(result)


@dataclass(frozen=True)
class WorkReport:
    """Both bounds of one Target at one workload."""
    target: str
    model_revision: str
    workload: str
    shape: Mapping[str, int]
    prompt_tokens: int | None
    quantization: str
    launch: tuple[Invocation, ...]
    flow: tuple[Invocation, ...]


def work(target: Target, *, workload: str | None = None, prompt_tokens: int | None = None,
         quantization: str | None = None, **options: ConfigValue) -> WorkReport:
    """Trace `target`'s model reference at `workload`'s shape and derive both bounds.

    `prompt_tokens` fills that many of the prompt's slots (the replay-time
    axis); `None` traces every slot, as the physical layout computes them.
    `options` are further construction options (a cut depth: `steps`, `layers`)."""
    recipe_name = target.precision if quantization is None else quantization
    runner = build_runner(target, workload=workload, quantization=recipe_name, declare=True,
                          device="cpu", **options)
    stage_of = {node.call_site: stage for stage in runner.graph.segment_names
                for node in runner.graph.nodes_of(stage) if not node.is_copy}
    model = target.model
    ops, output_keys, parameter_keys = trace(model.reference_run(runner.shape, prompt_tokens))
    # The rules name standard call sites; a Target that renames them says how.
    sites = tuple(None if site is None else target.call_site_aliases.get(site, site)
                  for site in call_sites(ops, model.work_rules))
    unknown = sorted({site for site in sites if site is not None} - set(stage_of))
    if unknown:
        raise ValueError(f"work rules name call sites the graph does not have: {unknown}")
    roofline = HARDWARE_ROOFLINES[target.hardware]
    bounds = {bound: invocations(ops, sites, output_keys, parameter_keys, bound=bound,
                                 stage_of=stage_of, precision=target.precision,
                                 pricing=target.quantization_recipe(recipe_name).pricing,
                                 l2_bytes=roofline.l2_bytes)
              for bound in KernelBound}
    return WorkReport(target=target.name, model_revision=model.model_revision,
                      workload=runner.workload, shape=dict(runner.shape),
                      prompt_tokens=prompt_tokens, quantization=recipe_name,
                      launch=bounds[KernelBound.LAUNCH], flow=bounds[KernelBound.FLOW])


def stage_floors(report: WorkReport, roofline: Roofline,
                 constants: Mapping[str, ConstantRow]) -> dict[str, dict[str, float]]:
    """Per stage, its flow bound in microseconds at datasheet rates, at the rates
    this machine was measured to deliver (`measurement.constants`), and the floor:
    each invocation at the better of the two rates per resource. A part that runs
    above its rated clock outruns its datasheet FLOPs (RTX 5090 BF16) while
    falling short of its rated bandwidth, so neither column alone bounds a stage
    that mixes GEMMs with streaming kernels."""
    stream = constants["stream"]["value"] * 1e12
    tensor = {fmt: constants[peak.role]["value"] * 1e12
              for fmt, peak in roofline.tensor_peaks.items()}
    best_stream = max(stream, roofline.dram_bytes_per_second)
    best_tensor = {fmt: max(rate, roofline.tensor_peaks[fmt].flops_per_second)
                   for fmt, rate in tensor.items()}
    floors: dict[str, dict[str, float]] = {}
    for invocation in report.flow:
        nbytes = invocation.bytes_read + invocation.bytes_written
        measured = max(nbytes / stream,
                       sum(count / tensor[fmt] for fmt, count in invocation.flops.items()))
        best = max(nbytes / best_stream,
                   sum(count / best_tensor[fmt] for fmt, count in invocation.flops.items()))
        entry = floors.setdefault(invocation.stage,
                                  {"datasheet_us": 0.0, "measured_us": 0.0, "floor_us": 0.0})
        entry["datasheet_us"] += invocation.seconds(roofline) * 1e6
        entry["measured_us"] += measured * 1e6
        entry["floor_us"] += best * 1e6
    return floors


def summary(report: WorkReport, roofline: Roofline) -> dict[str, object]:
    """Per call site (launch) and per stage (both bounds): invocations, FLOPs,
    bytes and the bound in microseconds at datasheet rates."""
    def rollup(rows: Sequence[Invocation],
               key: Callable[[Invocation], str]) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for row in rows:
            entry = table.setdefault(key(row), {"invocations": 0, "flops": 0, "bytes": 0.0, "us": 0.0})
            entry["invocations"] += 1
            entry["flops"] += sum(row.flops.values())
            entry["bytes"] += row.bytes_read + row.bytes_written
            entry["us"] += row.seconds(roofline) * 1e6
        return table

    return {
        "target": report.target, "model_revision": report.model_revision, "torch": torch.__version__,
        "workload": report.workload, "shape": dict(report.shape),
        "prompt_tokens": report.prompt_tokens, "quantization": report.quantization,
        "rates": {"dram_bytes_per_second": roofline.dram_bytes_per_second,
                  "l2_bytes": roofline.l2_bytes,
                  "tensor_flops_per_second": {fmt: peak.flops_per_second
                                              for fmt, peak in roofline.tensor_peaks.items()},
                  "source": f"{roofline.spec.__module__}.{roofline.spec.__name__}"},
        "launch_kernel_bound": {
            "us": sum(row.seconds(roofline) for row in report.launch) * 1e6,
            "stages": rollup(report.launch, lambda row: row.stage),
            "call_sites": rollup(report.launch, lambda row: row.call_sites[0])},
        "flow_kernel_bound": {
            "us": sum(row.seconds(roofline) for row in report.flow) * 1e6,
            "stages": rollup(report.flow, lambda row: row.stage)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workload", default=None, help=WORKLOAD_HELP)
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="valid prompt tokens (replay-time); default: every slot")
    parser.add_argument("--quantization", default=None, help="one of the Target's recipes")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    target = get_target(args.target)
    report = work(target, workload=args.workload, prompt_tokens=args.prompt_tokens,
                  quantization=args.quantization)
    text = json.dumps(summary(report, HARDWARE_ROOFLINES[target.hardware]), indent=2)
    print(text)
    if args.out is None:
        return 0
    with open(args.out, "w") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
