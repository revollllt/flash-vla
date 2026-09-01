"""FFN input producers for the fixed Pi0.5 action-expert shape.

The Phase-1 path converts an existing gated residual directly. The Phase-2
path first writes the gated residual and exact row-square partials, then a
small successor reduces those partials and writes the layout consumed by the
persistent GatedProjection kernel::

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
def tl_out_proj_residual_partials(
        A, W, AttentionGate, Residual,
        HiddenReady, DownReady, SquarePartials,
        BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
        THREADS: int, M_PAD: int):
    """Baseline-only split A: write residual and exact FP32 square partials.

    Production uses ``tl_out_proj_residual_rms_xfs`` so the partials and XFS
    tail share one cooperative launch. Keep this kernel for isolated/split
    comparisons only.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    W: T.Tensor((K, N), dtype)
    AttentionGate: T.Tensor((N,), dtype)
    Residual: T.Tensor((M, N), dtype)
    HiddenReady: T.Tensor((32,), "int32")
    DownReady: T.Tensor((32,), "int32")
    SquarePartials: T.Tensor((4, 32, 16), accum_dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M_PAD, BLOCK_M),
                  threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        gate_local = T.alloc_fragment((BLOCK_N,), dtype)
        accumulator = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        row_partial = T.alloc_fragment((BLOCK_M,), accum_dtype)

        T.copy(AttentionGate[pid_n * BLOCK_N], gate_local)
        T.clear(accumulator)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W[ko * BLOCK_K, pid_n * BLOCK_N], W_shared)
            T.gemm(A_shared, W_shared, accumulator)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            row = pid_m * BLOCK_M + i
            residual = T.if_then_else(
                row < M,
                Residual[row, pid_n * BLOCK_N + j],
                T.cast(0, dtype),
            )
            accumulator[i, j] = (
                accumulator[i, j] * gate_local[j].astype(accum_dtype)
                + residual.astype(accum_dtype)).astype(dtype).astype(
                    accum_dtype)
        T.copy(
            accumulator,
            Residual[pid_m * BLOCK_M, pid_n * BLOCK_N],
        )
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            accumulator[i, j] *= accumulator[i, j]
        T.reduce_sum(accumulator, row_partial, dim=1, batch=2)
        T.copy(row_partial, SquarePartials[pid_m, pid_n, 0])
        if pid_n == 0 and pid_m == 0:
            for index in T.Parallel(32):
                HiddenReady[index] = 0
                DownReady[index] = 0


@kernel(warp_spec=False)
def tl_out_proj_residual_rms_xfs(
        A, W, AttentionGate, Residual, FFNScale,
        HiddenReady, DownReady, SquarePartials, XFS,
        BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
        THREADS: int, M_PAD: int, TRIGGER_AT_ENTRY: bool):
    """Production cooperative residual, exact RMS partials, and XFS producer."""
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
    XFS: T.Tensor((N, M_PAD), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M_PAD, BLOCK_M),
                  threads=THREADS) as (pid_n, pid_m):
        thread_id = T.get_thread_binding()
        # Entry trigger (PDL-chain variant): scheduling-only. The consumer's
        # grid-dependency wait spans full completion and visibility (its
        # role-split form releases only warps that read static weights), so
        # releasing here lets the persistent FFN's CTAs place during this
        # grid's lifetime and retirement wave instead of after it.
        if TRIGGER_AT_ENTRY:
            if thread_id == 0:
                T.evaluate(T.call_extern(
                    "void", "cudaTriggerProgrammaticLaunchCompletion"))
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        gate_local = T.alloc_fragment((BLOCK_N,), dtype)
        accumulator = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        row_partial = T.alloc_fragment((BLOCK_M,), accum_dtype)

        T.copy(AttentionGate[pid_n * BLOCK_N], gate_local)
        T.clear(accumulator)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W[ko * BLOCK_K, pid_n * BLOCK_N], W_shared)
            T.gemm(A_shared, W_shared, accumulator)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            row = pid_m * BLOCK_M + i
            residual = T.if_then_else(
                row < M,
                Residual[row, pid_n * BLOCK_N + j],
                T.cast(0, dtype),
            )
            accumulator[i, j] = (
                accumulator[i, j] * gate_local[j].astype(accum_dtype)
                + residual.astype(accum_dtype)).astype(dtype).astype(
                    accum_dtype)
        T.copy(
            accumulator,
            Residual[pid_m * BLOCK_M, pid_n * BLOCK_N],
        )
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            accumulator[i, j] *= accumulator[i, j]
        T.reduce_sum(accumulator, row_partial, dim=1, batch=2)
        T.copy(row_partial, SquarePartials[pid_m, pid_n, 0])
        if pid_n == 0 and pid_m == 0:
            for index in T.Parallel(32):
                HiddenReady[index] = 0
                DownReady[index] = 0

        # The cooperative launch makes every residual/partial/reset visible
        # before any CTA starts the current N32/R16 XFS tail.
        T.sync_grid()
        if not TRIGGER_AT_ENTRY:
            if thread_id == 0:
                T.evaluate(T.call_extern(
                    "void", "cudaTriggerProgrammaticLaunchCompletion"))

        scalar_sum = T.alloc_fragment((1,), accum_dtype)
        scale_local = T.alloc_fragment((BLOCK_N,), dtype)
        residual_tile = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
        rstd_shared = T.alloc_shared((BLOCK_M,), dtype)
        xfs_transposed = T.alloc_shared((BLOCK_N, BLOCK_M + 2), dtype)

        # Preserve the current tail's latency hiding: issue the L2-hot
        # residual and scale loads before the strict exact reduction chain.
        T.clear(residual_tile)
        T.copy(
            Residual[pid_m * BLOCK_M, pid_n * BLOCK_N],
            residual_tile,
        )
        T.copy(FFNScale[pid_n * BLOCK_N], scale_local)
        if thread_id < BLOCK_M:
            row = pid_m * BLOCK_M + thread_id
            scalar_sum[0] = 0.0
            for n_block in T.Unroll(32):
                scalar_sum[0] += SquarePartials[
                    row // BLOCK_M,
                    n_block,
                    row % BLOCK_M,
                ]
            rstd_shared[thread_id] = T.rsqrt(
                scalar_sum[0] / N + 1e-6).astype(dtype)
        T.sync_threads()

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            value = residual_tile[i, j]
            normalized = (value * rstd_shared[i]).astype(dtype)
            xfs_transposed[j, i] = (
                normalized * scale_local[j]).astype(dtype)
        T.sync_threads()
        T.copy(
            xfs_transposed[:, :BLOCK_M],
            XFS[pid_n * BLOCK_N, pid_m * BLOCK_M],
            disable_tma=True,
        )


@kernel(warp_spec=False)
def tl_rms_xfs_from_partials(
        Residual, FFNScale, SquarePartials, XFS,
        BLOCK_M: int, BLOCK_N: int, ROWS_PER_CTA: int,
        THREADS: int, M_PAD: int,
        TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH: bool):
    """Baseline-only split B: reduce partials and emit contiguous K-major XFS.

    Production performs this tail after the grid sync inside
    ``tl_out_proj_residual_rms_xfs``. Keep this kernel for split baselines.
    """
    M, N = T.const("M, N")
    dtype = T.bfloat16
    accum_dtype = T.float32
    Residual: T.Tensor((M, N), dtype)
    FFNScale: T.Tensor((N,), dtype)
    SquarePartials: T.Tensor((4, 32, 16), accum_dtype)
    XFS: T.Tensor((N, M_PAD), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M_PAD, ROWS_PER_CTA),
                  threads=THREADS) as (pid_n, pid_m):
        thread_id = T.get_thread_binding()
        scalar_sum = T.alloc_fragment((1,), accum_dtype)
        scale_local = T.alloc_fragment((BLOCK_N,), dtype)
        residual_tile = T.alloc_fragment((ROWS_PER_CTA, BLOCK_N), dtype)
        rstd_shared = T.alloc_shared((ROWS_PER_CTA,), dtype)
        # The padded row stride avoids shared-bank conflicts during transpose.
        xfs_transposed = T.alloc_shared((BLOCK_N, ROWS_PER_CTA + 2), dtype)

        if TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH and thread_id == 0:
            T.evaluate(T.call_extern(
                "void", "cudaTriggerProgrammaticLaunchCompletion"))

        # Issue the L2-hot residual and scale loads before the dependent row
        # reduction. Other warps can make progress while the first half-warp
        # walks the exact 32-add chain.
        T.clear(residual_tile)
        T.copy(
            Residual[pid_m * ROWS_PER_CTA, pid_n * BLOCK_N],
            residual_tile,
        )
        T.copy(FFNScale[pid_n * BLOCK_N], scale_local)

        if thread_id < ROWS_PER_CTA:
            row = pid_m * ROWS_PER_CTA + thread_id
            scalar_sum[0] = 0.0
            for n_block in T.Unroll(32):
                scalar_sum[0] += SquarePartials[
                    row // BLOCK_M,
                    n_block,
                    row % BLOCK_M,
                ]
            rstd_shared[thread_id] = T.rsqrt(
                scalar_sum[0] / N + 1e-6).astype(dtype)
        T.sync_threads()

        for i, j in T.Parallel(ROWS_PER_CTA, BLOCK_N):
            value = residual_tile[i, j]
            normalized = (value * rstd_shared[i]).astype(dtype)
            xfs_transposed[j, i] = (
                normalized * scale_local[j]).astype(dtype)
        T.sync_threads()
        T.copy(
            xfs_transposed[:, :ROWS_PER_CTA],
            XFS[pid_n * BLOCK_N, pid_m * ROWS_PER_CTA],
            disable_tma=True,
        )
