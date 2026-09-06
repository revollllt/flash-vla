"""Raw-CUDA backend for the Pi0.5 H100 target.

`wrappers` provides the call sites a plan can route here -- the encoder
attention, the decoder attention half and the FFN half -- as a stateful
backend, so each op table owns its own libraries, scratch, and packed weights.
`FFNTaskloop` is the persistent FFN kernel's host side.

`ROUTE_CONSTRAINTS` declares which call sites share a buffer contract that
holds only when they resolve here together; the runtime validates a plan
against it at engine construction (`flash_vla.runtime.binding`). The module
satisfies the registry contract of `flash_vla.runtime.registry`.
"""

from flash_vla.runtime.binding import RouteConstraint

from . import wrappers
from .taskloop import FFNTaskloop, build_table

ATTENTION_NAMES = wrappers.ATTENTION_NAMES
NAMES = wrappers.NAMES
OPS = wrappers.OPS
make_wrappers = wrappers.make_wrappers

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
        ("action_expert_norm_gated_ffn", "action_expert_ffn_down_residual"),
        "the XFS producer and the persistent FFN share readiness counters"),
    # The fused producer writes the K-major FFN input directly, so it is only
    # meaningful ahead of this backend's FFN pair; the pair may run without it.
    RouteConstraint.requires(
        "action_expert_out_proj_residual",
        ("action_expert_norm_gated_ffn", "action_expert_ffn_down_residual"),
        "the fused out-projection producer feeds the persistent FFN's K-major input"),
)

def graph_contract(routes) -> dict[str, list[str]]:
    """Kernel names the captured program must and must not contain on these routes.

    The XFS producer resets the FFN readiness counters itself, so a standalone
    reset kernel in the graph means the pair is wired wrong. On the fused
    three-call route the cooperative producer replaces both the TileLang
    residual GEMM and the split producer pair, and exactly one cooperative
    producer kernel must be present.
    """
    mine = {name for name, backend in routes.items() if backend in ("cuda", "cuda-pdl")}
    forbid: list[str] = []
    require_one: list[str] = []
    if {"action_expert_norm_gated_ffn", "action_expert_ffn_down_residual"} <= mine:
        forbid.append("reset_ffn_counters_kernel")
    if "action_expert_out_proj_residual" in mine:
        forbid += ["_matmul_gated_res", "tl_rms_xfs_kmajor",
                   "tl_out_proj_residual_partials", "tl_rms_xfs_from_partials"]
        require_one.append("tl_out_proj_residual_rms_xfs")
    return {"forbid": forbid, "require_one": require_one}


# `llm_backbone_attention` deliberately carries no constraint. Both routes read the
# pipeline's own encoder_Q/K/V in the layout `llm_backbone_norm_qkv_rope` writes and
# neither owns scratch that crosses a call site, so it may route alone. What
# keeps that layout single-sourced is that the encoder QKV projection is
# TileLang-only. A CUDA encoder QKV writing head-major scratch would need a
# constraint like the decoder pair's.

__all__ = [
    "ATTENTION_NAMES", "NAMES", "OPS", "ROUTE_CONSTRAINTS",
    "graph_contract", "make_wrappers", "FFNTaskloop", "build_table", "wrappers",
]
