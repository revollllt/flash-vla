"""Lazy pre-norm kernels: the RMS factor is computed inside the consuming GEMM.

Row scaling commutes with the GEMM's K reduction, so rms(x) @ W and
(x @ W) * rstd(x)[:, None] are the same value and the factor does not have to
exist before the GEMM starts. Both kernels here accumulate the row sum of
squares from the same shared-memory tile the GEMM consumes -- so x is read from
global exactly once and the reduction rides the existing copy/compute overlap --
then apply the factor to the fp32 accumulator in the epilogue.

PRO_K bounds the register cost of that reduction: the squares fragment is
BLOCK_M x PRO_K in fp32, and PRO_K must divide BLOCK_K.

One TileLang constraint governs how this is written: every statement inside a
pipelined body must be a fully inlined expression. A named temporary
(`xv = A_shared[i, j]`) lowers to a bind statement that the warp-specialization
role pass cannot classify, and the compile aborts. That is why the squares below
repeat the indexing expression instead of binding it once.
"""
from __future__ import annotations

import tilelang.language as T

from .kernels import _gelu, kernel


@kernel
def tl_fused_rms_gate(A, W1, W2, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                      THREADS: int, SWIZZLE: int = 0, PRO_K: int = 64):
    """C = gelu(rms(A) @ W1) * (rms(A) @ W2), replacing a factor kernel plus a gated GEMM.

    Warp specialization must stay on, as in `kernels.tl_scaled_gate`: the dual
    GEMM reuses one A_shared across the W1 and W2 pipeline stages, which the
    no-WS pipeline planner rejects.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    W1: T.Tensor((K, N), dtype)
    W2: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        if SWIZZLE > 0:
            T.use_swizzle(panel_size=SWIZZLE, order="row")
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W1_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        W2_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        C1_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        C2_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        Pow = T.alloc_fragment((BLOCK_M, PRO_K), accum_dtype)
        S = T.alloc_fragment((BLOCK_M,), accum_dtype)
        SS_shared = T.alloc_shared((BLOCK_M,), accum_dtype)

        T.clear(Pow)
        T.clear(C1_local)
        T.clear(C2_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W1[ko * BLOCK_K, pid_n * BLOCK_N], W1_shared)
            T.gemm(A_shared, W1_shared, C1_local)
            T.copy(W2[ko * BLOCK_K, pid_n * BLOCK_N], W2_shared)
            T.gemm(A_shared, W2_shared, C2_local)
            for kc in T.Serial(BLOCK_K // PRO_K):
                for i, j in T.Parallel(BLOCK_M, PRO_K):
                    Pow[i, j] += (A_shared[i, kc * PRO_K + j].astype(accum_dtype)
                                  * A_shared[i, kc * PRO_K + j].astype(accum_dtype))
        T.reduce_sum(Pow, S, dim=1)
        T.copy(S, SS_shared)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            fv = T.rsqrt(SS_shared[i] / K + 1e-6).astype(dtype)
            C1_local[i, j] = C1_local[i, j] * fv.astype(accum_dtype)
            C2_local[i, j] = C2_local[i, j] * fv.astype(accum_dtype)
            C1_local[i, j] = _gelu(C1_local[i, j])
            C1_local[i, j] = C1_local[i, j] * C2_local[i, j]

        T.copy(C1_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


@kernel
def tl_fused_rms_matmul_bias_res(A, B, Bias, R, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                                 NUM_STAGES: int, THREADS: int, PRO_K: int = 64):
    """C = R + Bias + rms(A) @ B, written in place with R and C the same buffer."""
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    Bias: T.Tensor((N,), dtype)
    R: T.Tensor((M, N), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        Bias_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        Pow = T.alloc_fragment((BLOCK_M, PRO_K), accum_dtype)
        S = T.alloc_fragment((BLOCK_M,), accum_dtype)
        SS_shared = T.alloc_shared((BLOCK_M,), accum_dtype)

        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(Pow)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
            for kc in T.Serial(BLOCK_K // PRO_K):
                for i, j in T.Parallel(BLOCK_M, PRO_K):
                    Pow[i, j] += (A_shared[i, kc * PRO_K + j].astype(accum_dtype)
                                  * A_shared[i, kc * PRO_K + j].astype(accum_dtype))
        T.reduce_sum(Pow, S, dim=1)
        T.copy(S, SS_shared)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            fv = T.rsqrt(SS_shared[i] / K + 1e-6).astype(dtype)
            C_local[i, j] = (
                C_local[i, j] * fv.astype(accum_dtype)
                + Bias_local[j].astype(accum_dtype)
                + R[pid_m * BLOCK_M + i, pid_n * BLOCK_N + j].astype(accum_dtype)
            )

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
