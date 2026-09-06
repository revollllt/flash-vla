"""Raw-CUDA backend for the Pi0.5 H100 target.

`wrappers` provides the call sites a plan can route here -- the backbone
attention from the shared `gemma_backbone` component package, and the
action-expert halves the shared `gemma_expert` package owns -- as a stateful
backend, so each op table owns its own libraries, scratch, and packed weights.
`FFNTaskloop` is the persistent FFN kernel's host side, re-exported from the
expert package.

`ROUTE_CONSTRAINTS` declares which call sites share a buffer contract that
holds only when they resolve here together; the runtime validates a plan
against it at engine construction (`flash_vla.runtime.binding`). The module
satisfies the registry contract of `flash_vla.runtime.registry`.
"""

from ....gemma_expert.backends import cuda as _expert
from ....gemma_expert.backends.cuda import FFNTaskloop, build_table
from . import wrappers

ATTENTION_NAMES = wrappers.ATTENTION_NAMES
NAMES = wrappers.NAMES
OPS = wrappers.OPS
make_wrappers = wrappers.make_wrappers

#: The backend names of this Target that mean the expert package.
_EXPERT_BACKENDS = ("cuda", "cuda-pdl")

#: Every constraint here is the expert package's; the encoder attention
#: deliberately carries none. Both routes read the pipeline's own
#: encoder_Q/K/V in the layout `llm_backbone_norm_qkv_rope` writes and neither
#: owns scratch that crosses a call site, so it may route alone. What keeps
#: that layout single-sourced is that the encoder QKV projection is
#: TileLang-only. A CUDA encoder QKV writing head-major scratch would need a
#: constraint like the expert attention pair's.
ROUTE_CONSTRAINTS = _expert.ROUTE_CONSTRAINTS


def graph_contract(routes) -> dict[str, list[str]]:
    """Kernel names the captured program must and must not contain on these routes.

    Everything the contract names belongs to the expert package; this Target
    only says which of its backend names mean that package.
    """
    return _expert.graph_contract_from_routes(routes, _EXPERT_BACKENDS)


__all__ = [
    "ATTENTION_NAMES", "NAMES", "OPS", "ROUTE_CONSTRAINTS",
    "graph_contract", "make_wrappers", "FFNTaskloop", "build_table", "wrappers",
]
