"""The expert's RMSNorm scale-factor kernel.

Emitted as a factor rather than a normalized tensor because the consuming GEMM
applies it, which is how the upstream decoder splits the work. Its PDL variant
releases the programmatic launch dependency at entry, for the chain in
`.agents/notes/implemented/architecture/2026-09-02-decoder-pdl-chain.md`.

Squares accumulate over K in chunks rather than in one whole-row fragment,
which would spill the register file at the block sizes used here.
"""
from __future__ import annotations

import tilelang.language as T

from ..jit import kernel


@kernel
def tl_rms_factor(A, F, BLOCK_M: int, BLOCK_K: int, THREADS: int,
                  TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH: bool):
    """F[m] = rsqrt(mean_k(A[m, k]^2) + 1e-6), the decoder's RMSNorm scale factor.

    Emits the factor, not the normalized tensor: the consuming GEMM applies it,
    which is how the upstream decoder splits the work.
    """
    M, K = T.const("M, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    F: T.Tensor((M,), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=THREADS) as bx:
        # Release the programmatic launch dependency at entry: the PDL qkv
        # consumer performs the mandatory grid-dependency wait before it reads
        # F, so this early signal only exposes scheduling overlap; it does not
        # provide memory visibility.
        if TRIGGER_PROGRAMMATIC_DEPENDENT_LAUNCH:
            thread_id = T.get_thread_binding()
            if thread_id == 0:
                T.evaluate(T.call_extern(
                    "void", "cudaTriggerProgrammaticLaunchCompletion"))
        A_local = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
        A_pow_local = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)
        A_powsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
        T.clear(A_pow_local)
        for k in T.Serial(T.ceildiv(K, BLOCK_K)):
            T.copy(A[bx * BLOCK_M, k * BLOCK_K], A_local)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                x = A_local[i, j].astype(accum_dtype)
                A_pow_local[i, j] += x * x
        T.reduce_sum(A_pow_local, A_powsum, dim=1)
        for i in T.Parallel(BLOCK_M):
            A_powsum[i] = T.rsqrt(A_powsum[i] / K + 1e-6)
        T.copy(A_powsum, F[bx * BLOCK_M])


__all__ = ["tl_rms_factor"]
