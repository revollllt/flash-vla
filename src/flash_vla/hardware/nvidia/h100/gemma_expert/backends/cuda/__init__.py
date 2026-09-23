"""Raw-CUDA backend for the H100 Gemma action expert.

`wrappers` provides the five call sites a Target's plan can route here -- the
attention half and the FFN half -- as a stateful backend, so each op table owns
its own libraries, scratch, and packed weights. `FFNTaskloop` is the persistent
FFN kernel's host side.

`ROUTE_CONSTRAINTS` declares which call sites share a buffer contract that
holds only when they resolve here together; the runtime validates a plan
against them at engine construction (`flash_vla.runtime.binding`).
`graph_contract` is the same statement about the captured program, over the
call sites the registry has routed to the backend that wraps this package.

This module imports `runtime/`, the shared tile library and this package's
TileLang producers, and no Target.
"""

from flash_vla.runtime.binding import RouteConstraint
from flash_vla.runtime.registry import GraphContract


from . import wrappers
from .taskloop import FFNTaskloop, build_table

ATTENTION_NAMES = wrappers.ATTENTION_NAMES
NAMES = wrappers.NAMES
make_wrappers = wrappers.make_wrappers

#: Call sites of the FFN half: the producer and the persistent consumer.
FFN_NAMES = ("action_expert_norm_gated_ffn", "action_expert_ffn_down_residual")
#: The producer that folds the out projection into the FFN's K-major input.
OUT_PROJ_NAME = "action_expert_out_proj_residual"

ROUTE_CONSTRAINTS = (
    # The qkv wrapper leaves the pipeline's token-major Q unwritten: its real Q
    # is head-major scratch its own attention reads. A plan that splits the
    # pair would feed another backend's attention a stale buffer.
    RouteConstraint.atomic(
        ATTENTION_NAMES,
        "Q crosses the pair head-major in implementation-owned scratch"),
    # The XFS producer resets the readiness counters the persistent consumer
    # waits on, and the two must be adjacent launches for the PDL contract.
    RouteConstraint.atomic(
        FFN_NAMES,
        "the XFS producer and the persistent FFN share readiness counters"),
    # The fused producer writes the K-major FFN input directly, so it is only
    # meaningful ahead of this backend's FFN pair; the pair may run without it.
    RouteConstraint.requires(
        OUT_PROJ_NAME, FFN_NAMES,
        "the fused out-projection producer feeds the persistent FFN's K-major input"),
)


def graph_contract(routed: frozenset[str]) -> GraphContract:
    """Kernel names the captured program must and must not contain, given the
    call sites `routed` to this package.

    The XFS producer resets the FFN readiness counters itself, so a standalone
    reset kernel in the graph means the pair is wired wrong. On the fused
    three-call route the cooperative producer replaces both the TileLang
    residual GEMM and the split producer pair, and exactly one cooperative
    producer kernel must be present.
    """
    reset_forbidden = ("reset_ffn_counters_kernel",) if set(FFN_NAMES) <= routed else ()
    if OUT_PROJ_NAME not in routed:
        return GraphContract(forbid=reset_forbidden)
    return GraphContract(
        forbid=reset_forbidden + ("_matmul_gated_res", "tl_rms_xfs_kmajor",
                           "tl_out_proj_residual_partials", "tl_rms_xfs_from_partials"),
        require_one=("tl_out_proj_residual_rms_xfs",))


__all__ = [
    "ATTENTION_NAMES", "FFN_NAMES", "NAMES", "OUT_PROJ_NAME",
    "ROUTE_CONSTRAINTS", "build_table", "graph_contract", "make_wrappers", "wrappers",
    "FFNTaskloop",
]
