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
        X, Scale, HiddenReady, DownReady, XFS,
        BLOCK_M: int, BLOCK_K: int, OUTPUT_K: int,
        THREADS: int, M_PAD: int, TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH: bool,
        RESET_READINESS_COUNTERS: bool):
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
    HiddenReady: T.Tensor((32,), "int32")
    DownReady: T.Tensor((32,), "int32")
    XFS: T.Tensor((K, M_PAD), dtype)

    with T.Kernel(T.ceildiv(K, OUTPUT_K), T.ceildiv(M_PAD, BLOCK_M),
                  threads=THREADS) as (pid_k, pid_m):
        thread_id = T.get_thread_binding()
        # The producer replaces the standalone one-block reset. Only CTA(0,0)
        # touches the two contiguous arrays; its block barrier ensures all 64
        # stores are issued before that CTA releases its PDL dependency.
        if RESET_READINESS_COUNTERS and pid_k == 0 and pid_m == 0:
            for index in T.Parallel(32):
                HiddenReady[index] = 0
                DownReady[index] = 0
            T.sync_threads()
        # Every primary CTA releases its programmatic launch dependency as
        # soon as it is resident.  The consumer still performs the mandatory
        # dependency wait before its first XFS TMA, so this early signal only
        # exposes scheduling overlap; it does not provide memory visibility.
        if TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH:
            if thread_id == 0:
                T.evaluate(T.call_extern(
                    "void", "cudaTriggerProgrammaticLaunchCompletion"))

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


@kernel(warp_spec=False)
def tl_out_proj_residual_rms_xfs(
        A, W, AttentionGate, Residual, FFNScale,
        HiddenReady, DownReady, SquarePartials, RstdPerCTA, XFS,
        BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
        THREADS: int, M_PAD: int):
    """Fuse the fixed decoder out-projection, residual, RMS, and K-major XFS.

    The 32x4 grid owns one M16/N32 output tile per CTA. Each CTA publishes one
    FP32 row-square partial, then all CTAs join once. Only the first 16 threads
    in each CTA reduce one row apiece, avoiding the fragment mapping that would
    otherwise make every warp reload the same 2-KiB partial tile.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    W: T.Tensor((K, N), dtype)
    AttentionGate: T.Tensor((N,), dtype)
    Residual: T.Tensor((M, N), dtype)
    FFNScale: T.Tensor((N,), dtype)
    HiddenReady: T.Tensor((32,), "int32")
    DownReady: T.Tensor((32,), "int32")
    SquarePartials: T.Tensor((4, 32, 16), accum_dtype)
    RstdPerCTA: T.Tensor((4, 32, 16), dtype)
    XFS: T.Tensor((N, M_PAD), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M_PAD, BLOCK_M),
                  threads=THREADS) as (pid_n, pid_m):
        thread_id = T.get_thread_binding()
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        gate_local = T.alloc_fragment((BLOCK_N,), dtype)
        scale_local = T.alloc_fragment((BLOCK_N,), dtype)
        residual_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
        rounded_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
        accumulator = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        row_partial = T.alloc_fragment((BLOCK_M,), accum_dtype)
        scalar_sum = T.alloc_fragment((1,), accum_dtype)
        # Padding the token stride from 16 to 18 avoids the 8-way conflict of
        # shared[j, i] under the common j-fast lane mapping.
        xfs_transposed = T.alloc_shared((BLOCK_N, BLOCK_M + 2), dtype)

        T.copy(AttentionGate[pid_n * BLOCK_N], gate_local)
        T.clear(residual_local)
        T.copy(
            Residual[pid_m * BLOCK_M, pid_n * BLOCK_N], residual_local)
        T.clear(accumulator)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W[ko * BLOCK_K, pid_n * BLOCK_N], W_shared)
            T.gemm(A_shared, W_shared, accumulator)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            rounded_local[i, j] = (
                accumulator[i, j] * gate_local[j].astype(accum_dtype)
                + residual_local[i, j].astype(accum_dtype)).astype(dtype)
            value = rounded_local[i, j].astype(accum_dtype)
            accumulator[i, j] = value * value
        T.copy(
            rounded_local,
            Residual[pid_m * BLOCK_M, pid_n * BLOCK_N],
        )
        T.reduce_sum(accumulator, row_partial, dim=1)
        T.copy(row_partial, SquarePartials[pid_m, pid_n, 0])
        if pid_n == 0 and pid_m == 0:
            for index in T.Parallel(32):
                HiddenReady[index] = 0
                DownReady[index] = 0
        T.sync_grid()

        if thread_id < BLOCK_M:
            scalar_sum[0] = 0.0
            for n_block in T.Serial(32):
                scalar_sum[0] += SquarePartials[
                    pid_m, n_block, thread_id]
            RstdPerCTA[pid_m, pid_n, thread_id] = T.rsqrt(
                scalar_sum[0] / N + 1e-6).astype(dtype)
        T.sync_threads()

        T.copy(FFNScale[pid_n * BLOCK_N], scale_local)
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            normalized = (
                rounded_local[i, j]
                * RstdPerCTA[pid_m, pid_n, i]).astype(dtype)
            xfs_transposed[j, i] = T.if_then_else(
                pid_m * BLOCK_M + i < M,
                (normalized * scale_local[j]).astype(dtype),
                T.cast(0, dtype),
            )
        T.sync_threads()
        T.copy(
            xfs_transposed[:, :BLOCK_M],
            XFS[pid_n * BLOCK_N, pid_m * BLOCK_M],
            disable_tma=True,
        )
