"""Raw-CUDA backend for the Pi0.5 H100 target.

`wrappers` provides the call sites a plan can route here -- the encoder
attention, the decoder attention half and the FFN half -- as a stateful
backend, so each op table owns its own libraries, scratch, and packed weights.
`FFNTaskloop` is the persistent FFN kernel's host side.

`ROUTE_CONSTRAINTS` declares which call sites share a buffer contract that
holds only when they resolve here together; the runtime validates a plan
against it at engine construction (`flash_vla.runtime.binding`).
"""

from flash_vla.runtime.binding import RouteConstraint

from . import wrappers
from .taskloop import FFNTaskloop, build_table

ATTENTION_NAMES = wrappers.ATTENTION_NAMES
WRAPPER_NAMES = wrappers.WRAPPER_NAMES
FUSED_WRAPPERS = dict(wrappers.FUSED_WRAPPERS)
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
        ("decoder_norm_gated_ffn", "decoder_ffn_down_residual"),
        "the XFS producer and the persistent FFN share readiness counters"),
    # The fused producer writes the K-major FFN input directly, so it is only
    # meaningful ahead of this backend's FFN pair; the pair may run without it.
    RouteConstraint.requires(
        "decoder_out_proj_residual",
        ("decoder_norm_gated_ffn", "decoder_ffn_down_residual"),
        "the fused out-projection producer feeds the persistent FFN's K-major input"),
)

# `encoder_attention` deliberately carries no constraint. Both routes read the
# pipeline's own encoder_Q/K/V in the layout `encoder_norm_qkv_rope` writes and
# neither owns scratch that crosses a call site, so it may route alone. What
# keeps that layout single-sourced is that the encoder QKV projection is
# TileLang-only. A CUDA encoder QKV writing head-major scratch would need a
# constraint like the decoder pair's.

__all__ = [
    "ATTENTION_NAMES", "WRAPPER_NAMES", "FUSED_WRAPPERS", "ROUTE_CONSTRAINTS",
    "make_wrappers", "FFNTaskloop", "build_table", "wrappers",
]
