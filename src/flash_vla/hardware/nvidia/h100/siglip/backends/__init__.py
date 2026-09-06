"""Backend factories for the SigLIP vision encoder on H100.

Each subpackage satisfies the registry contract of `flash_vla.runtime.registry`
(`NAMES`, `make_wrappers(scratch, selected_names)`, `ROUTE_CONSTRAINTS`, `OPS`)
and is registered by a Target under a name of the Target's choosing. Nothing
here imports a Target.

    cublas   the two pre-norm projections as torch's LayerNorm + a fused
             cuBLASLt epilogue, the form Pi0.5 already reaches through its own
             TileLang backend and Pi0 does not
    cuda     the fused attention kernel, and the same two projections with a
             hand-written LayerNorm in place of torch's
"""
from __future__ import annotations

from . import cublas, cuda

__all__ = ["cublas", "cuda"]
