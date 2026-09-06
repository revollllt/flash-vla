"""Hand-written CUDA call sites for Pi0.5: the prefix attention, plus the expert.

`llm_backbone_attention` is the prefix's full bidirectional multi-query
attention (`kernels/enc_attn.cu`, host side in `enc_attn.py`) and is the only
kernel this Target still owns directly. It is independent of every other call
site: it reads the pipeline's own Q/K/V buffers in the layout the TileLang
encoder QKV wrapper writes and owns no cross-call scratch, so a plan may select
it alone.

The five action-expert call sites come from the device component package
`hardware/nvidia/h100/gemma_expert`, which both H100 Targets share. This module
merges the two sets into the op table the `cuda` and `cuda-pdl` backends
provide, so a plan's call-site names and route constraints are unchanged by
where the kernels live.
"""

from __future__ import annotations

from ....gemma_expert.backends import cuda as _expert
from . import enc_attn as _enc

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

    def llm_backbone_attention(Q, K, V, scale, mask, out):
        """One fused kernel for the prefix's QK^T / softmax / PV chain.

        Same contract as the TileLang call site: `Q` is (M*heads, head_dim) with
        row = token * head, `K`/`V` are (M, head_dim), `mask` is (M,) additive,
        and the (M, heads*head_dim) result is returned. Here it is a view of
        `out`, which the kernel writes in full.

        Stateless by construction -- no packed weights, no scratch that has to
        outlive the call -- so this wrapper carries none of the engine-lifetime
        state the expert ones do. The tensor maps are cached in `enc_attn` on
        the buffer triple, because encoding them is a driver call and must stay
        out of graph capture.
        """
        if out.shape != Q.shape:
            raise ValueError(
                f"encoder attention writes into `out`; got out {tuple(out.shape)} "
                f"for Q {tuple(Q.shape)}")
        return _enc.attention(Q, K, V, scale, mask, out).view(K.shape[0], -1)

    table: dict[str, object] = {}
    if "llm_backbone_attention" in selected:
        table["llm_backbone_attention"] = llm_backbone_attention
    expert = selected & _expert.NAMES
    if expert:
        table.update(_expert.make_wrappers(
            scratch, selected_names=expert, pdl_chain=pdl_chain))
    return table


__all__ = ["ATTENTION_NAMES", "M", "NAMES", "OPS", "make_wrappers"]
