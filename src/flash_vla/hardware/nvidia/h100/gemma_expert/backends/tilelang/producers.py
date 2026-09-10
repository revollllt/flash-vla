"""The expert chain's TileLang producers, and the compile cache they share.

Three wrappers, all launched by this package's CUDA backend rather than by a
plan: none of them is a call site.

- `rms_factor` is the pre-QKV RMSNorm factor, one launch, optionally carrying
  the PDL entry trigger.
- `rms_xfs` writes the persistent FFN's K-major input directly from an already
  updated residual, and resets the readiness counters the consumer waits on.
- `out_proj_residual_rms_xfs` is the production form: one cooperative launch
  that writes the gated residual, the exact fp32 row-square partials and the
  counter reset, crosses a grid sync, then emits the same K-major XFS
  (`.agents/notes/implemented/architecture/2026-08-28-cooperative-xfs-pdl.md`).

`M` and `M_PAD` are parameters rather than the Pi0.5 constants they used to
be, because both Targets' experts share these kernels and differ in row count
(50 rows on Pi0.5, 51 on Pi0) while padding to the same 64.
"""
from __future__ import annotations

import torch

from .kernels import rms as rms_kernels
from .kernels import xfs as xfs_kernels

#: One 128-byte TMA row of BF16: the row extent every expert buffer pads to,
#: and the M extent the persistent FFN's K-major activation requires.
M_PAD = 64

_RMS = dict(BLOCK_M=2, BLOCK_K=256, THREADS=128,
            TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH=False)
_XFS = dict(
    BLOCK_M=8,
    BLOCK_K=256,
    OUTPUT_K=32,
    THREADS=128,
    M_PAD=M_PAD,
    TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH=False,
    RESET_READINESS_COUNTERS=True,
)
_OUT_PROJ_PARTIALS = dict(
    BLOCK_M=16,
    BLOCK_N=32,
    BLOCK_K=256,
    NUM_STAGES=4,
    THREADS=128,
    M_PAD=M_PAD,
)
_XFS_FROM_PARTIALS = dict(
    BLOCK_M=16,
    BLOCK_N=32,
    ROWS_PER_CTA=16,
    THREADS=128,
    M_PAD=M_PAD,
    TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH=True,
)

_CACHE: dict[tuple, object] = {}


def _compiled(kernel, **const):
    """Compile `kernel` for one shape and config, memoized on both.

    Keyed on `kernel.tl_name` rather than the kernel object, which TileLang
    leaves unhashable. The name distinguishes the two variants of a shared
    body, so a WS-on and a WS-off call site never collide.
    """
    key = (kernel.tl_name, tuple(sorted(const.items())))
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = kernel.compile(**const)
        _CACHE[key] = compiled
    return compiled


def _check_counters(**arrays) -> None:
    for name, counters in arrays.items():
        if (tuple(counters.shape) != (32,)
                or counters.dtype != torch.int32
                or not counters.is_contiguous()):
            raise ValueError(f"{name} must be contiguous int32[32]")


def rms_factor(x, out, cfg=_RMS, *, trigger_programmatic_launch=False):
    """Write rsqrt(mean(x^2)+eps) into `out`, which the consuming GEMM scales by.

    Pi0's fused TileLang path folds this into `tl_fused_rms_gate`. The chain
    cannot: AdaRMSNorm needs the shared tile unscaled for the norm and scaled
    for the GEMM, so the factor is its own launch here on both Targets.

    ``trigger_programmatic_launch`` selects the JIT variant that releases the
    PDL dependency at entry; the successor must carry the wait.
    """
    M, K = x.shape
    config = dict(cfg)
    config["TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH"] = trigger_programmatic_launch
    _compiled(rms_kernels.tl_rms_factor, M=M, K=K, **config)(x, out)
    return out


def rms_xfs(x, scale, hidden_ready, down_ready, out,
            *, trigger_programmatic_launch=False, reset_readiness=True):
    """Write the next FFN's exact BF16 input as contiguous ``[D, M_PAD]``.

    ``x`` is the BF16 expert activation *after* the out-projection has applied
    its residual update. This replaces the row factor for the persistent
    GatedProjection path; neither the factor nor a row-major normalized
    activation is materialized. When ``trigger_programmatic_launch`` is true
    the persistent consumer must be the direct successor on the same stream.
    The producer resets both readiness arrays before publishing XFS.
    """
    M, K = x.shape
    if (tuple(scale.shape) != (K,) or tuple(out.shape) != (K, M_PAD)):
        raise ValueError(
            f"rms_xfs requires scale[{K}] and out[{K},{M_PAD}] for x[{M},{K}]; "
            f"got scale {tuple(scale.shape)} out {tuple(out.shape)}")
    if any(t.dtype != torch.bfloat16 for t in (x, scale, out)):
        raise ValueError("rms_xfs tensors must be BF16")
    if any(not t.is_contiguous() for t in (x, scale, out)):
        raise ValueError("rms_xfs tensors must be contiguous")
    _check_counters(hidden_ready=hidden_ready, down_ready=down_ready)
    config = dict(_XFS)
    config["TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH"] = trigger_programmatic_launch
    config["RESET_READINESS_COUNTERS"] = reset_readiness
    _compiled(xfs_kernels.tl_rms_xfs_kmajor, M=M, K=K, **config)(
        x, scale, hidden_ready, down_ready, out)
    return out


def out_proj_residual_rms_xfs(
        attention, weight, attention_gate, residual, ffn_scale,
        hidden_ready, down_ready, square_partials, xfs,
        *, trigger_at_entry=False):
    """Cooperatively produce the exact residual and contiguous K-major XFS.

    ``trigger_at_entry`` selects the JIT variant that releases the PDL
    dependency at kernel entry instead of after the grid sync; the persistent
    consumer's grid-dependency wait carries correctness either way.
    """
    M, K = attention.shape
    N = weight.shape[1]
    if (tuple(weight.shape) != (K, N)
            or tuple(attention_gate.shape) != (N,)
            or tuple(residual.shape) != (M, N)
            or tuple(ffn_scale.shape) != (N,)
            or tuple(square_partials.shape) != (M_PAD // _OUT_PROJ_PARTIALS["BLOCK_M"],
                                                N // _OUT_PROJ_PARTIALS["BLOCK_N"],
                                                _OUT_PROJ_PARTIALS["BLOCK_M"])
            or tuple(xfs.shape) != (N, M_PAD)):
        raise ValueError("invalid fixed-shape fused out-projection/XFS tensors")
    bf16_tensors = (attention, weight, attention_gate, residual, ffn_scale, xfs)
    if any(tensor.dtype != torch.bfloat16 for tensor in bf16_tensors):
        raise ValueError("fused out-projection/XFS data tensors must be BF16")
    if square_partials.dtype != torch.float32:
        raise ValueError("square_partials must be FP32")
    if any(not tensor.is_contiguous() for tensor in (*bf16_tensors, square_partials)):
        raise ValueError("fused out-projection/XFS tensors must be contiguous")
    _check_counters(hidden_ready=hidden_ready, down_ready=down_ready)
    _compiled(
        xfs_kernels.tl_out_proj_residual_rms_xfs,
        M=M, N=N, K=K, **_OUT_PROJ_PARTIALS,
        TRIGGER_AT_ENTRY=trigger_at_entry,
    )(
        attention, weight, attention_gate, residual, ffn_scale,
        hidden_ready, down_ready, square_partials, xfs,
    )
    return residual, xfs


__all__ = ["M_PAD", "out_proj_residual_rms_xfs", "rms_factor", "rms_xfs"]
