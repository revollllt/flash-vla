"""TileLang kernels for Pi0 inference, one per upstream Triton kernel.

Every kernel is a 1:1 port: same fusion boundaries, same epilogue math, same
constants as the Triton original, so a call site can swap backends without any
other change. Shapes are compile-time constants; `wrappers.py` specializes each
kernel per Pi0 call site and owns the tuned tile configs.

Two conventions run through the file.

Output parameters. Almost every kernel takes its destination as a `T.Tensor`
parameter and writes it with `T.copy`, rather than allocating a fresh `T.empty`
and returning it. Returning forces the wrapper to copy the result into the real
buffer, which costs one device-to-device graph node per call; the upstream
Triton kernels write through a pointer argument for the same reason.

Warp specialization. TileLang lowers `T.copy` to TMA plus a producer/consumer
warp split. That split pays for itself only when there is enough work to hide:
above one wave it wins, below one wave the producer warp sits idle and still
costs warps and mbarrier traffic. Several kernels therefore exist as a WS-on and
a WS-off variant sharing one body -- the decoder calls the sub-wave variant, the
encoder and vision stages call the other. The two differ only in `pass_configs`,
which TileLang treats as part of the compile cache key.
"""
from __future__ import annotations

import tilelang.language as T

from flash_vla.hardware.nvidia.h100.gemma_expert.backends.tilelang.kernels import primitives as expert

# FAST_MATH / NO_WARP_SPEC are re-exported: `autotune.rewrap` reads them
# from this module when it re-wraps a raw builder.
from flash_vla.hardware.nvidia.tilelang.jit import (  # noqa: F401
    FAST_MATH,
    NO_WARP_SPEC,
    KernelSet,
)

#: This module's own JIT namespace. Per module rather than shared: the
#: two Targets and the component packages declare kernels under the same
#: names with bodies that have diverged, and one registry would hand the
#: autotuner whichever module imported last.
_KERNELS = KernelSet()
RAW_KERNELS = _KERNELS.RAW_KERNELS
variant = _KERNELS.variant
kernel = _KERNELS.kernel


GELU_C0 = 1.5957691216057308
GELU_C1 = 0.044715


def _silu(v):
    return v * (1.0 / (1.0 + T.__exp(-v)))


def _gelu(v):
    return v * (1.0 / (1.0 + T.__exp(-(GELU_C0 * v * (1.0 + GELU_C1 * v * v)))))


# ---------------------------------------------------------------------------
# GEMM family (mirrors matmul_small and its bias / residual / activation forms)
# ---------------------------------------------------------------------------
def _matmul(A, B, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int, THREADS: int):
    """C = A @ B.

    Body shared by `tl_matmul` (decoder attn @ V, sub-wave, WS off) and
    `tl_matmul_ws` (encoder QKV, high occupancy, WS on).
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


tl_matmul = variant(_matmul, "tl_matmul", warp_spec=False)
tl_matmul_ws = variant(_matmul, "tl_matmul_ws", warp_spec=True)


def _matmul_res(A, B, R, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                THREADS: int, SWIZZLE: int = 0):
    """C = A @ B + R.

    The wrapper passes one buffer as both R and C for an in-place residual: each
    thread reads its R element into the accumulator before the copy writes that
    same element, so the aliasing is safe.

    SWIZZLE > 0 groups threadblocks into L2-sized panels. It matters only when
    the weight working set exceeds L2 (encoder ffn-down streams a 64 MB weight at
    K=16384); leave it at 0 otherwise.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    R: T.Tensor((M, N), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        if SWIZZLE > 0:
            T.use_swizzle(panel_size=SWIZZLE, order="row")
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = C_local[i, j] + R[pid_m * BLOCK_M + i, pid_n * BLOCK_N + j].astype(accum_dtype)

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


tl_matmul_res = variant(_matmul_res, "tl_matmul_res", warp_spec=False)
tl_matmul_res_ws = variant(_matmul_res, "tl_matmul_res_ws", warp_spec=True)


def _matmul_bias(A, B, Bias, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                 THREADS: int):
    """C = A @ B + bias[None, :].

    Body shared by `tl_matmul_bias` (encoder projector, WS on) and
    `tl_matmul_bias_nows` (vision QKV, WS off).
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    Bias: T.Tensor((N,), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        Bias_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = C_local[i, j] + Bias_local[j].astype(accum_dtype)

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


tl_matmul_bias = variant(_matmul_bias, "tl_matmul_bias", warp_spec=True)
tl_matmul_bias_nows = variant(_matmul_bias, "tl_matmul_bias_nows", warp_spec=False)


@kernel
def tl_matmul_bias_res(A, B, Bias, R, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                       THREADS: int):
    """C = A @ B + bias[None, :] + R."""
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

        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = (
                C_local[i, j]
                + Bias_local[j].astype(accum_dtype)
                + R[pid_m * BLOCK_M + i, pid_n * BLOCK_N + j].astype(accum_dtype)
            )

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


@kernel(warp_spec=False)
def tl_matmul_bias_res_mod(A, B, Bias, R, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                           NUM_STAGES: int, THREADS: int, I_MOD: int):
    """C = A @ B + bias[None, :] + R[row % I_MOD, :].

    The vision patch embedding, where the positional-embedding residual has one
    row per patch position and is broadcast across views by the modulo.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    Bias: T.Tensor((N,), dtype)
    R: T.Tensor((I_MOD, N), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        Bias_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = (
                C_local[i, j]
                + Bias_local[j].astype(accum_dtype)
                + R[(pid_m * BLOCK_M + i) % I_MOD, pid_n * BLOCK_N + j].astype(accum_dtype)
            )

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


@kernel
def tl_matmul_bias_gelu(A, B, Bias, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                        THREADS: int):
    """C = gelu_tanh(A @ B + bias[None, :])."""
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    Bias: T.Tensor((N,), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        Bias_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = C_local[i, j] + Bias_local[j].astype(accum_dtype)
            C_local[i, j] = _gelu(C_local[i, j])

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


@kernel(infer_output=True)
def tl_matmul_bias_silu(A, B, Bias, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                        THREADS: int):
    """C = silu(A @ B + bias[None, :]). Allocates and returns C (one small call per step)."""
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    Bias: T.Tensor((N,), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        Bias_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = C_local[i, j] + Bias_local[j].astype(accum_dtype)
            C_local[i, j] = _silu(C_local[i, j])

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])

    return C


# ---------------------------------------------------------------------------
# Normalization
#
# All three accumulate squares over K in chunks rather than in one whole-row
# fragment, which would spill the register file at the block sizes used here.
# ---------------------------------------------------------------------------
@kernel
def tl_rms_factor(A, F, BLOCK_M: int, BLOCK_K: int, THREADS: int):
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


@kernel
def tl_rms_norm(X, O, BLOCK_M: int, BLOCK_K: int, THREADS: int):
    """O = X * rsqrt(mean_k(X^2) + 1e-6), the encoder's RMSNorm.

    Emits the normalized tensor, unlike `tl_rms_factor`: the upstream encoder
    normalizes before the GEMM, so the two are not interchangeable.
    """
    M, K = T.const("M, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    X: T.Tensor((M, K), dtype)
    O: T.Tensor((M, K), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=THREADS) as bx:
        A_local = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
        A_pow_local = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)
        A_powsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
        T.clear(A_pow_local)
        for k in T.Serial(T.ceildiv(K, BLOCK_K)):
            T.copy(X[bx * BLOCK_M, k * BLOCK_K], A_local)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                x = A_local[i, j].astype(accum_dtype)
                A_pow_local[i, j] += x * x
        T.reduce_sum(A_pow_local, A_powsum, dim=1)
        for i in T.Parallel(BLOCK_M):
            A_powsum[i] = T.rsqrt(A_powsum[i] / K + 1e-6)
        for k in T.Serial(T.ceildiv(K, BLOCK_K)):
            T.copy(X[bx * BLOCK_M, k * BLOCK_K], A_local)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                A_local[i, j] = (A_local[i, j].astype(accum_dtype) * A_powsum[i]).astype(dtype)
            T.copy(A_local, O[bx * BLOCK_M, k * BLOCK_K])


@kernel
def tl_layer_norm(X, Wn, Bn, O, BLOCK_M: int, BLOCK_K: int, THREADS: int, EPS: float):
    """O = (X - mean) / sqrt(var + EPS) * Wn + Bn, with var from E[x^2] - mean^2 in fp32."""
    M, K = T.const("M, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    X: T.Tensor((M, K), dtype)
    Wn: T.Tensor((K,), dtype)
    Bn: T.Tensor((K,), dtype)
    O: T.Tensor((M, K), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=THREADS) as pid_m:
        Xc = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
        Tmp = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)
        Wl = T.alloc_fragment((BLOCK_K,), dtype)
        Bl = T.alloc_fragment((BLOCK_K,), dtype)
        s = T.alloc_fragment((BLOCK_M,), accum_dtype)
        ss = T.alloc_fragment((BLOCK_M,), accum_dtype)
        mean = T.alloc_fragment((BLOCK_M,), accum_dtype)
        inv = T.alloc_fragment((BLOCK_M,), accum_dtype)
        T.clear(s)
        T.clear(ss)
        for ko in T.Serial(T.ceildiv(K, BLOCK_K)):
            T.copy(X[pid_m * BLOCK_M, ko * BLOCK_K], Xc)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                Tmp[i, j] = Xc[i, j].astype(accum_dtype)
            T.reduce_sum(Tmp, s, dim=1, clear=False)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                x = Xc[i, j].astype(accum_dtype)
                Tmp[i, j] = x * x
            T.reduce_sum(Tmp, ss, dim=1, clear=False)
        for i in T.Parallel(BLOCK_M):
            mean[i] = s[i] / K
            inv[i] = T.rsqrt(ss[i] / K - mean[i] * mean[i] + EPS)
        for ko in T.Serial(T.ceildiv(K, BLOCK_K)):
            T.copy(X[pid_m * BLOCK_M, ko * BLOCK_K], Xc)
            T.copy(Wn[ko * BLOCK_K], Wl)
            T.copy(Bn[ko * BLOCK_K], Bl)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                xn = (Xc[i, j].astype(accum_dtype) - mean[i]) * inv[i]
                Xc[i, j] = (xn * Wl[j].astype(accum_dtype) + Bl[j].astype(accum_dtype)).astype(dtype)
            T.copy(Xc, O[pid_m * BLOCK_M, ko * BLOCK_K])


# ---------------------------------------------------------------------------
# Gated FFN
# ---------------------------------------------------------------------------
tl_scaled_gate = kernel(expert.tl_scaled_gate)


@kernel
def tl_matmul_gate(A, W1, W2, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                   THREADS: int, SWIZZLE: int = 0):
    """C = gelu_tanh(A @ W1) * (A @ W2), the encoder's gated FFN on pre-normalized input.

    Not interchangeable with `tl_scaled_gate`, which expects a raw x plus a
    factor. Keeping W1 and W2 co-resident caps NUM_STAGES at 2, which is the
    binding limit on this kernel -- it is the single largest kernel in the model.
    SWIZZLE restores L2 reuse at N=16384, where the default rasterization order
    keeps far more weight columns co-resident than L2 holds.
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

        T.clear(C1_local)
        T.clear(C2_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W1[ko * BLOCK_K, pid_n * BLOCK_N], W1_shared)
            T.gemm(A_shared, W1_shared, C1_local)
            T.copy(W2[ko * BLOCK_K, pid_n * BLOCK_N], W2_shared)
            T.gemm(A_shared, W2_shared, C2_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C1_local[i, j] = _gelu(C1_local[i, j])
            C1_local[i, j] = C1_local[i, j] * C2_local[i, j]

        T.copy(C1_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


tl_scaled_matmul_bias_res = kernel(infer_output=True)(expert.tl_scaled_matmul_bias_res)


# ---------------------------------------------------------------------------
# QKV projection with RoPE
#
# Both kernels scatter their output straight into Q, K and V. The N tiling is
# chosen so each tile falls entirely inside one of the three, which lets a
# runtime branch on the tile index pick the destination -- no packed buffer and
# no device-to-device scatter afterwards.
# ---------------------------------------------------------------------------
tl_qkv_gemm_rope = kernel(expert.tl_qkv_gemm_rope)


@kernel
def tl_rope_scatter_bf16(C, Rope, OutQ, OutK, OutV, BLOCK_M: int, BLOCK_PAIR: int, THREADS: int,
                         HEAD_DIM: int, NUM_HEADS: int):
    """Encoder RoPE + scatter, applied to an already-projected bf16 buffer.

    Rounds to bf16 before rotating, matching the upstream encoder kernel -- the
    opposite order from `tl_qkv_gemm_rope`, which is why the two are separate.
    Requires BLOCK_PAIR to divide HEAD_DIM // 2; the loop is over column pairs,
    so each tile stays inside one of Q, K, V.
    """
    M, N = T.const("M, N")
    dtype = T.bfloat16
    accum_dtype = T.float32
    q_dim = NUM_HEADS * HEAD_DIM
    C: T.Tensor((M, N), dtype)
    Rope: T.Tensor((M, HEAD_DIM), dtype)
    OutQ: T.Tensor((M, q_dim), dtype)
    OutK: T.Tensor((M, HEAD_DIM), dtype)
    OutV: T.Tensor((M, HEAD_DIM), dtype)
    rope_cols = (NUM_HEADS + 1) * HEAD_DIM

    with T.Kernel(T.ceildiv(N // 2, BLOCK_PAIR), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_p, pid_m):
        pair0 = pid_p * BLOCK_PAIR
        j0_tile = pair0 * 2
        if j0_tile < q_dim:
            for i, p in T.Parallel(BLOCK_M, BLOCK_PAIR):
                row = pid_m * BLOCK_M + i
                j0 = (pair0 + p) * 2
                j1 = j0 + 1
                rj0 = j0 % HEAD_DIM
                x0 = C[row, j0].astype(accum_dtype)
                x1 = C[row, j1].astype(accum_dtype)
                cos_v = Rope[row, rj0].astype(accum_dtype)
                sin_v = Rope[row, rj0 + 1].astype(accum_dtype)
                OutQ[row, j0] = (x0 * cos_v - x1 * sin_v).astype(dtype)
                OutQ[row, j1] = (x1 * cos_v + x0 * sin_v).astype(dtype)
        elif j0_tile < rope_cols:
            for i, p in T.Parallel(BLOCK_M, BLOCK_PAIR):
                row = pid_m * BLOCK_M + i
                j0 = (pair0 + p) * 2
                j1 = j0 + 1
                rj0 = j0 % HEAD_DIM
                x0 = C[row, j0].astype(accum_dtype)
                x1 = C[row, j1].astype(accum_dtype)
                cos_v = Rope[row, rj0].astype(accum_dtype)
                sin_v = Rope[row, rj0 + 1].astype(accum_dtype)
                OutK[row, j0 - q_dim] = (x0 * cos_v - x1 * sin_v).astype(dtype)
                OutK[row, j1 - q_dim] = (x1 * cos_v + x0 * sin_v).astype(dtype)
        else:
            for i, p in T.Parallel(BLOCK_M, BLOCK_PAIR):
                row = pid_m * BLOCK_M + i
                j0 = (pair0 + p) * 2
                j1 = j0 + 1
                OutV[row, j0 - q_dim - HEAD_DIM] = C[row, j0]
                OutV[row, j1 - q_dim - HEAD_DIM] = C[row, j1]


# ---------------------------------------------------------------------------
# Decoder attention
#
# Two implementations of the same maths. The scores + softmax + attn@V chain is
# the literal 1:1 port and materializes the (queries, keys) score matrix. The
# FlashDecoding pair below keeps the scores in SRAM and replaces all three with
# two kernels; it is what the default path runs.
# ---------------------------------------------------------------------------
@kernel(warp_spec=False)
def tl_matmul_abT_scale(Qt, Kt, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                        THREADS: int, SCALE: float):
    """C = (Qt @ Kt^T) * SCALE."""
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    Qt: T.Tensor((M, K), dtype)
    Kt: T.Tensor((N, K), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(Qt[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(Kt[pid_n * BLOCK_N, ko * BLOCK_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = C_local[i, j] * SCALE

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


tl_softmax_mask0 = kernel(expert.tl_softmax_mask0)


tl_fd_flat_split = kernel(expert.tl_fd_flat_split)


tl_fd_flat_combine = kernel(expert.tl_fd_flat_combine)
