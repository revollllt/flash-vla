"""Backend factories for the SigLIP vision encoder on H100.

Each subpackage exposes a `BACKEND` (`flash_vla.runtime.registry.Backend`)
that a Target registers under a name of its choosing. Nothing
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
