"""Pi0 on the RTX 5090: the same graph, a different machine underneath.

Pi0's forward graph is model semantics -- three stages, no host slot, bf16 --
and it does not change with the GPU. What changes is which backend each call
site routes to, and that is what this Target owns.

**This is the bring-up route and it is deliberately slow.** Every call site runs
in plain torch (`backends/torch_ops.py`). Nothing is fused, nothing is tuned,
and the number it measures is a starting point for the optimization workflow
rather than a performance claim.

Nothing is inherited from H100/Pi0's routing, and that is the point. Its
`siglip-cuda` and `gemma-cuda` backends are hand-written sm_90a CUDA whose GEMM
mainloop is `wgmma`, which ptxas refuses on `sm_120a` [isa.wgmma.absent]. Its
TileLang tile shapes were tuned against 227 KB of shared memory; ten of nineteen
exceed this part's 99 KB [smem.bytes.cta.max] and fail at launch rather than
running slower. Re-tiling those by hand is guesswork against another machine's
tuning, and the architecture asks for a route customised per hardware -- so the
kernels get written fresh, against this machine's measured constants, in the
optimization loop.

ARCHITECTURE.md says a Target never imports another Target's kernels, and this
one imports H100/Pi0's graph. That is a deliberate bring-up shortcut with a
stated cost: the alternative is copying 342 lines of graph that is byte-for-byte
the same model and would drift the first time either side is touched. The rule
exists to stop two Targets sharing kernel ROUTING and TUNING, and this Target
shares neither -- its registry holds one backend of its own. The two torch
attention helpers it reuses were never kernels; they are `torch.compile`d SDPA
and a matmul chain, already hardware-neutral.
"""
from __future__ import annotations

from typing import Mapping

from flash_vla.hardware.nvidia.h100.pi0.target import Pi0

from .backends import REGISTRY


class Pi0RTX5090(Pi0):
    """Pi0 on one RTX 5090: three stages, no host slot, bf16, TileLang only.

    Identity is what makes this a separate Target rather than a flag. A
    measurement taken here must not be recorded against `h100-sxm5-80gb`: the
    two machines differ by 1.8x on streaming bandwidth, 3.4x on tensor-core
    throughput and 2.29x on shared memory per block, so a latency compared
    across them is not a comparison at all.
    """

    name = "hardware/nvidia/rtx5090/pi0"
    hardware = "rtx5090-32gb"

    registry = REGISTRY

    #: Every call site on the torch backend, which is also the registry
    #: default, so both plans are empty. They are stated separately rather than
    #: shared so that the first hand-written kernel can move `plan` without
    #: silently moving the numerical reference with it.
    plan: Mapping[str, str] = {}

    reference_plan: Mapping[str, str] = {}


#: The Target instance the factory in `flash_vla.inference` hands the runner.
TARGET = Pi0RTX5090()

__all__ = ["TARGET", "Pi0RTX5090"]
