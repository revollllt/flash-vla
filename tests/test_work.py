"""The work a reference's forward must do (`measurement.work`), on the meta device.

Two toy modules pin the tracer's semantics -- what counts, what folds, what is
computed once and where a pass ends -- and every model's work table is checked
to cover its Target's graph.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from flash_vla.hardware.nvidia import HARDWARE_ROOFLINES
from flash_vla.inference import declare, get_target
from flash_vla.runtime.vla import Pricing
from flash_vla.runtime.work import CallSiteRule, ReferenceRun
from measurement.work import KernelBound, call_sites, invocations, trace, work

META = torch.device("meta")
ROWS, WIDTH = 64, 256
MATMUL = 2 * ROWS * WIDTH * WIDTH


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.live = nn.Linear(WIDTH, WIDTH, bias=False, device=META)
        self.dead = nn.Linear(WIDTH, WIDTH, bias=False, device=META)
        self.constant = nn.Linear(WIDTH, WIDTH, bias=False, device=META)
        self.table = nn.Parameter(torch.empty(ROWS, WIDTH, device=META))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.dead(x)                                   # never reaches the output
        bias = self.constant(self.table)               # reads no input: folds away
        repeated = torch.cat((x, x), dim=0)[:ROWS]     # a move: reading it reads x
        return self.live(repeated) + self.live(repeated) + bias   # the second is the first


def test_only_live_input_dependent_work_counts_and_repeats_are_computed_once() -> None:
    model, x = Toy(), torch.empty(ROWS, WIDTH, device=META)
    ops, outputs, parameters = trace(ReferenceRun(model, lambda: (model(x),), (x,)))
    sites = call_sites(ops, (CallSiteRule("", None, "site"),))
    [launch] = invocations(ops, sites, outputs, parameters, bound=KernelBound.LAUNCH,
                           stage_of={"site": "stage"}, precision="bf16", pricing=None, l2_bytes=0)
    assert launch.flops == {"bf16": MATMUL}
    # x, the live weight and the folded bias in; the output out; all bf16.
    assert launch.bytes_read == 2 * (ROWS * WIDTH + WIDTH * WIDTH + ROWS * WIDTH)
    assert launch.bytes_written == 2 * ROWS * WIDTH


class Loop(nn.Module):
    """Three denoising steps through one layer, as one monolithic call site."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(WIDTH, WIDTH, bias=False, device=META)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = x + self.layer(x)
        return x


@pytest.mark.parametrize("l2_bytes", [0, 1 << 30])
def test_a_repeated_pass_rereads_its_weights_except_what_the_l2_holds(l2_bytes: int) -> None:
    model, x = Loop(), torch.empty(ROWS, WIDTH, device=META)
    ops, outputs, parameters = trace(ReferenceRun(model, lambda: (model(x),), (x,)))
    sites = call_sites(ops, (CallSiteRule("", None, "site"),))
    common = dict(stage_of={"site": "stage"}, precision="bf16", pricing=None, l2_bytes=l2_bytes)
    launch = invocations(ops, sites, outputs, parameters, bound=KernelBound.LAUNCH, **common)
    flow = invocations(ops, sites, outputs, parameters, bound=KernelBound.FLOW, **common)
    weight = 2 * WIDTH * WIDTH
    assert len(launch) == len(flow) == 3
    assert all(step.bytes_read >= weight for step in launch)
    rereads = [step.bytes_read for step in flow[1:]]
    assert (all(read >= weight for read in rereads) if l2_bytes == 0
            else all(read < weight for read in rereads))


class Buffers(nn.Module):
    """A moved buffer written in place, a mask rebuilt every step, and two
    same-shaped scratch buffers each filled differently."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(WIDTH, WIDTH, bias=False, device=META)

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        moved = x.clone()
        moved[:, :8] = update                          # the write must stay live
        total = self.layer(moved)
        for _ in range(2):
            mask = torch.cat((x, x), dim=0) > 0        # rebuilt, but the same each step
            total = total + mask[:ROWS].to(x.dtype)
        first = torch.zeros(ROWS, WIDTH, device=META).add_(x)
        second = torch.zeros(ROWS, WIDTH, device=META).add_(update.sum())
        return total + first + second


def test_moved_writes_stay_live_repeats_fold_and_filled_buffers_stay_apart() -> None:
    model = Buffers()
    x, update = torch.empty(ROWS, WIDTH, device=META), torch.empty(ROWS, 8, device=META)
    ops, _outputs, _parameters = trace(ReferenceRun(model, lambda: (model(x, update),), (x, update)))
    counted = [op.op for op in ops if op.counted]
    assert "aten.copy_" in counted                     # the in-place write into the clone
    assert counted.count("aten.gt") == 1               # the mask, computed once
    assert counted.count("aten.add_") == 2             # two buffers, not one


class Recipe(nn.Module):
    """Two quantized call sites passing an activation, then an unquantized one."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(WIDTH, WIDTH, bias=False, device=META)
        self.down = nn.Linear(WIDTH, WIDTH, bias=False, device=META)
        self.out = nn.Linear(WIDTH, WIDTH, bias=False, device=META)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.down(self.up(x)))


def test_a_recipe_prices_its_matmul_weights_and_passed_activations_in_both_bounds() -> None:
    model, x = Recipe(), torch.empty(ROWS, WIDTH, device=META)
    ops, outputs, parameters = trace(ReferenceRun(model, lambda: (model(x),), (x,)))
    sites = call_sites(ops, tuple(CallSiteRule(rf"^{name}$", None, name) for name in ("up", "down", "out")))
    pricing = Pricing(call_sites=frozenset({"up", "down"}), tensor="mxfp8",
                      weight_itemsize=1, activation_itemsize=1)
    common = dict(stage_of={"up": "stage", "down": "stage", "out": "stage"}, precision="bf16",
                  pricing=pricing, l2_bytes=0)
    launch = invocations(ops, sites, outputs, parameters, bound=KernelBound.LAUNCH, **common)
    [flow] = invocations(ops, sites, outputs, parameters, bound=KernelBound.FLOW, **common)
    up, down, out = launch
    assert up.flops == {"mxfp8": MATMUL} and out.flops == {"bf16": MATMUL}
    # up: bf16 input, 1-byte weight, 1-byte hidden to down; down reads it at 1 byte.
    assert up.bytes_read == 2 * ROWS * WIDTH + WIDTH * WIDTH and up.bytes_written == ROWS * WIDTH
    assert down.bytes_read == ROWS * WIDTH + WIDTH * WIDTH
    assert flow.bytes_read + flow.bytes_written <= sum(i.bytes_read + i.bytes_written for i in launch)


@pytest.mark.parametrize("name,depth", [
    ("rtx5090/pi05", {"steps": 1, "layers": 2}),
    ("rtx5090/pi0", {"steps": 1, "layers": 2}),
    ("h100/lingbot_vla", {"steps": 1, "layers": 2}),
    ("rtx5090/groot_n17", {}),
])
def test_the_work_table_covers_the_graph(name: str, depth: dict[str, int]) -> None:
    """Every op the reference runs lands on a call site of the Target's graph,
    every call site does work -- except Pi0's prompt embedding, which the
    reference computes from a prompt fixed at load and so folds away -- and no
    fusion bound exceeds the launch bound it fuses."""
    target = get_target(name)
    graph_sites = set(declare(name, **depth).graph.call_sites)
    report = work(target, **depth)
    traced = {site for launch in report.launch for site in launch.call_sites}
    folded = {"llm_backbone_embed_prompt"} if target.model.name == "pi0" else set()
    assert traced <= graph_sites and graph_sites - traced == folded
    roofline = HARDWARE_ROOFLINES[target.hardware]
    assert (sum(flow.seconds(roofline) for flow in report.flow)
            <= sum(launch.seconds(roofline) for launch in report.launch))
