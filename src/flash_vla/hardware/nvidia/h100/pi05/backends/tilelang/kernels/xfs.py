"""FFN input preparation for the fixed Pi0.5 action-expert shape.

The preceding attention output projection has already completed its gated
residual update when this kernel runs.  This kernel replaces the standalone
row-factor launch and writes exactly the layout consumed by the persistent
GatedProjection kernel::

    XFS[k, m] = bf16(bf16(X[m, k] * rstd[m]) * scale[k])

where ``rstd[m] = bf16(rsqrt(mean_k(float(X[m,k])**2) + 1e-6))``.
Rows ``m >= M`` are zero padding, so the fixed result is contiguous BF16
``[K=1024, M_PAD=64]`` with M as its unit-stride axis.
"""
from __future__ import annotations

import tilelang.language as T

from .base import kernel


@kernel(warp_spec=False)
def tl_rms_xfs_kmajor(
        X, Scale, XFS, BLOCK_M: int, BLOCK_K: int, OUTPUT_K: int,
        THREADS: int, M_PAD: int):
    """Compute row RMS and emit exact-rounding, zero-padded K-major XFS.

    Output-K CTAs intentionally recompute the small row RMS reduction.  For
    M=50, K=1024 this supplies enough independent work to fill H100 while all
    repeated X reads remain a few MiB in L2.  The shared transpose changes the
    final stores from a 64-BF16 stride into contiguous segments of K-major rows.
    """
    M, K = T.const("M, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    X: T.Tensor((M, K), dtype)
    Scale: T.Tensor((K,), dtype)
    XFS: T.Tensor((K, M_PAD), dtype)

    with T.Kernel(T.ceildiv(K, OUTPUT_K), T.ceildiv(M_PAD, BLOCK_M),
                  threads=THREADS) as (pid_k, pid_m):
        X_norm = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
        X_square = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)
        square_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
        rstd = T.alloc_fragment((BLOCK_M,), dtype)
        X_tile = T.alloc_fragment((BLOCK_M, OUTPUT_K), dtype)
        scale_tile = T.alloc_fragment((OUTPUT_K,), dtype)
        xfs_transposed = T.alloc_shared((OUTPUT_K, BLOCK_M), dtype)

        T.clear(X_square)
        for ko in T.Serial(T.ceildiv(K, BLOCK_K)):
            # Clearing makes the M=50 -> M_PAD=64 mask explicit: invalid rows
            # participate in the reduction as zero rather than stale registers.
            T.clear(X_norm)
            T.copy(X[pid_m * BLOCK_M, ko * BLOCK_K], X_norm)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                value = X_norm[i, j].astype(accum_dtype)
                X_square[i, j] += value * value
        T.reduce_sum(X_square, square_sum, dim=1)
        for i in T.Parallel(BLOCK_M):
            # The BF16 cast is part of the consumer's numerical contract.
            rstd[i] = T.rsqrt(square_sum[i] / K + 1e-6).astype(dtype)

        T.clear(X_tile)
        T.copy(X[pid_m * BLOCK_M, pid_k * OUTPUT_K], X_tile)
        T.copy(Scale[pid_k * OUTPUT_K], scale_tile)
        for i, j in T.Parallel(BLOCK_M, OUTPUT_K):
            # Keep both BF16 rounding points from tl_ada_scaled_gate.
            normalized = (X_tile[i, j] * rstd[i]).astype(dtype)
            xfs_transposed[j, i] = (
                normalized * scale_tile[j]).astype(dtype)
        T.sync_threads()
        T.copy(
            xfs_transposed,
            XFS[pid_k * OUTPUT_K, pid_m * BLOCK_M],
            disable_tma=True,
        )
