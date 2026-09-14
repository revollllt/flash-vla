"""Pi0.5 on the RTX 5090: the same graph, a different machine underneath.

The bring-up route, deliberately slow: every call site runs in plain torch
(`backends/torch_ops.py`), so what it measures starts the optimization workflow
rather than claiming anything.

Nothing is inherited from H100/Pi0.5's routing -- `wgmma` ptxas refuses on
sm_120a [isa.wgmma.absent], and its tiles assume 227 KB of shared memory against
this part's 99 KB [smem.bytes.cta.max]. It does import H100/Pi0.5's graph and
host slot, the same bring-up shortcut `rtx5090/pi0` takes: the rule against
importing another Target exists to stop two Targets sharing kernel routing and
tuning, and this one shares neither.
"""
from __future__ import annotations

from typing import Any, Mapping

from flash_vla.hardware.nvidia.h100.pi05.target import Pi05, forward_prefix, set_task

from .backends import REGISTRY


class Pi05RTX5090(Pi05):
    """Pi0.5 on one RTX 5090: three stages, one host slot, bf16, torch only.

    Identity is what makes this a separate Target rather than a flag. A
    measurement taken here must not be recorded against `h100-sxm5-80gb`: the
    two machines differ by 1.8x on streaming bandwidth, 3.4x on tensor-core
    throughput and 2.29x on shared memory per block, so a latency compared
    across them is not a comparison at all.
    """

    name = "hardware/nvidia/rtx5090/pi05"
    hardware = "rtx5090-32gb"

    registry = REGISTRY

    #: One backend, so the shipped route and the numerical reference route are
    #: the same program. `eval.correctness` therefore reports agreement rather
    #: than evidence until a second backend exists; what gates this Target today
    #: is the official comparison in `eval/pi05/`, against OpenPI's own forward.
    plan: Mapping[str, str] = {}
    reference_plan: Mapping[str, str] = {}

    #: H100's `action_expert_norm_gated_ffn` ceiling is a measured H100 number
    #: (`tma.bw.dev.burst`, job 591174) and says nothing about this part. Cleared
    #: rather than inherited; the floor model falls back to this machine's own
    #: constants until a `tma_ring`-equivalent sweep has been run here.
    CEILINGS: Mapping[str, Any] = {}


#: The Target instance the factory in `flash_vla.inference` hands the runner.
TARGET = Pi05RTX5090()

__all__ = ["TARGET", "Pi05RTX5090", "forward_prefix", "set_task"]
