"""Hand-written CUDA call sites for Pi0.5: the prefix attention, plus the expert.

This Target owns no kernel of its own any more. `llm_backbone_attention` is
the prefix's full bidirectional multi-query attention from the device
component package `hardware/nvidia/h100/gemma_backbone` (the same wrapper the
`gemma-cuda` backend provides, kept here so the plans that route it to `cuda`
keep binding); it reads the pipeline's own Q/K/V buffers in the layout the
TileLang encoder QKV wrapper writes and owns no cross-call scratch, so a plan
may select it alone. The five action-expert call sites come from
`hardware/nvidia/h100/gemma_expert`. This module merges the two sets into the
op table the `cuda` and `cuda-pdl` backends provide, so a plan's call-site
names and route constraints are unchanged by where the kernels live.
"""

from __future__ import annotations

from ....gemma_backbone.backends import cuda as _backbone
from ....gemma_expert.backends import cuda as _expert

#: Re-exported from the package: the chunk both expert halves are compiled for.
M = _expert.wrappers.M
#: The two expert attention call sites are one unit; the package's constraint
#: refuses to split them.
ATTENTION_NAMES = _expert.ATTENTION_NAMES

NAMES = frozenset({"llm_backbone_attention"}) | _expert.NAMES
#: No extension ops: every call site here is one of the standard vocabulary.
OPS: tuple = ()


def make_wrappers(
        scratch,
        selected_names: set[str] | None = None,
        pdl_chain: bool = False) -> dict[str, object]:
    """Build this Target's CUDA op table: the prefix attention and the expert.

    ``scratch`` is the runner's workspace allocator, which the package's
    persistent FFN uses for its K-major input, counters and partials.

    ``pdl_chain`` extends PDL over every boundary the expert package owns: the
    rms factor kernel triggers early, qkv / attention / combine launch with the
    programmatic attribute and wait at their first dependent read, and the
    persistent FFN uses the role-split wait (mode 2) that releases its
    dependency-free weight loaders ahead of the XFS producer. Off, launch
    semantics are exactly the shipped ones. It does not reach
    `llm_backbone_attention`: that kernel is one launch between two TileLang
    neighbours and arms no programmatic dependency.
    """
    selected = set(NAMES if selected_names is None else selected_names)
    unknown = selected - NAMES
    if unknown:
        raise KeyError(f"cuda backend does not implement {sorted(unknown)}")

    table: dict[str, object] = {}
    if "llm_backbone_attention" in selected:
        table.update(_backbone.make_wrappers(scratch, selected_names={"llm_backbone_attention"}))
    expert = selected & _expert.NAMES
    if expert:
        table.update(_expert.make_wrappers(
            scratch, selected_names=expert, pdl_chain=pdl_chain))
    return table


__all__ = ["ATTENTION_NAMES", "M", "NAMES", "OPS", "make_wrappers"]
