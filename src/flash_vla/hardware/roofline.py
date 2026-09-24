"""What the latency floor model reads of one device.

The floor model (`tools/profiling/floor.py`) divides a call site's bytes and
FLOPs by datasheet peaks and by what the machine was measured to deliver. The
roles it consumes are the model's; which datasheet field carries each peak,
and which measured constant fills each role, are the machine's -- sm_90 names
its tensor rate after `wgmma`, an instruction sm_120 does not have -- so each
device's `spec.py` declares its `ROOFLINE`, and
`hardware.nvidia.HARDWARE_ROOFLINES` finds it from an identity's hardware axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, kw_only=True)
class TensorPeak:
    """One tensor-core format's dense datasheet peak, and the measured-constants
    role (`Roofline.constant_tags`) of its observed rate."""
    flops_per_second: int
    role: str


@dataclass(frozen=True, kw_only=True)
class Roofline:
    """One device's datasheet peaks and measured table, as the floor model reads them.

    `spec` is the datasheet class the peaks come from, named in reports.
    `tensor_peaks` maps a call site's tensor-core format (`Invocation.tensor`)
    to its peak. `constant_tags` maps each floor role (`stream`, `burst`,
    `tensor`, ...) to the tag of the `constants_file` row that fills it;
    `burst` is optional, and without a measured burst curve the ceiling falls
    back to the stream model at every size.
    """
    spec: type
    dram_bytes_per_second: int
    tensor_peaks: Mapping[str, TensorPeak]
    constants_file: Path
    constant_tags: Mapping[str, str]


__all__ = ["Roofline", "TensorPeak"]
