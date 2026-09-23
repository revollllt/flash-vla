"""Raw-CUDA backend for the Pi0.5 H100 target.

`wrappers` provides the call sites a plan can route here -- the backbone
attention from the shared `gemma_backbone` component package, and the
action-expert halves the shared `gemma_expert` package owns -- as a stateful
backend, so each op table owns its own libraries, scratch, and packed weights.
`FFNTaskloop` is the persistent FFN kernel's host side, re-exported from the
expert package.

`BACKEND` is the `flash_vla.runtime.registry.Backend` the Target registers,
under `cuda` with shipped launch semantics and, as a variant, under `cuda-pdl`
with the PDL chain armed. Its route constraints and graph contract are the
expert package's: the encoder attention deliberately carries none. Both routes
read the pipeline's own encoder_Q/K/V in the layout `llm_backbone_norm_qkv_rope`
writes and neither owns scratch that crosses a call site, so it may route
alone. What keeps that layout single-sourced is that the encoder QKV projection
is TileLang-only. A CUDA encoder QKV writing head-major scratch would need a
constraint like the expert attention pair's.
"""


from flash_vla.runtime.registry import Backend

from ....gemma_expert.backends import cuda as _expert
from ....gemma_expert.backends.cuda import FFNTaskloop, build_table
from . import wrappers

ATTENTION_NAMES = wrappers.ATTENTION_NAMES
NAMES = wrappers.NAMES
make_wrappers = wrappers.make_wrappers
ROUTE_CONSTRAINTS = _expert.ROUTE_CONSTRAINTS

BACKEND = Backend(names=NAMES, make_wrappers=make_wrappers,
                  route_constraints=ROUTE_CONSTRAINTS, graph_contract=_expert.graph_contract)


__all__ = [
    "ATTENTION_NAMES", "BACKEND", "NAMES", "ROUTE_CONSTRAINTS",
    "make_wrappers", "FFNTaskloop", "build_table", "wrappers",
]
