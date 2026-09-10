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

    `tl_matmul` (decoder attn @ V, sub-wave, WS off). The encoder QKV site
    that used the WS variant now runs `tl_matmul_rope_scatter`.
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


# The plain residual GEMM (`C = A @ B + R`) that served the encoder's o_proj and
# ffn:down sites was retired: cuBLAS `addmm_` is the exact same epilogue and,
# in place, measured 13.9 us vs 21.4 (o_proj) and 86.4 vs 97.3 (ffn:down) per
# layer at M=968 against the best TileLang bodies including an smem-staged
# epilogue (artifacts/ktasks/prefix-gemm-epilogue, job 588745, cold weights).
# The residual+bias forms below stay hand-written: cuBLAS has no such epilogue.


def _matmul_bias(A, B, Bias, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                 THREADS: int):
    """C = A @ B + bias[None, :].

    `tl_matmul_bias` (encoder projector and decoder action in-projection, WS
    on). The vision QKV site that used a no-WS variant of this body now runs
    cuBLASLt (`torch.addmm`), see `wrappers.py`.

    The result is staged through a bf16 shared tile before the global store.
    Storing the accumulator fragment directly leaves each thread writing
    4-byte pieces in the wgmma layout, and at K=1152 (M=768 vision shapes)
    that traffic capped the body at ~300 TFLOP/s regardless of tiling; the
    staged store recovered 8-19% at the same tile configs (sweeps 585341-3
    vs 585399-402, artifacts/ktasks/vision-gemm-retune).
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
        C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
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

        T.copy(C_local, C_shared)
        T.copy(C_shared, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


tl_matmul_bias = variant(_matmul_bias, "tl_matmul_bias", warp_spec=True)


@kernel
def tl_matmul_bias_res(A, B, Bias, R, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                       THREADS: int):
    """C = A @ B + bias[None, :] + R. R and C may alias (in-place residual).

    The residual tile arrives through one shared-memory copy and the result
    leaves through the same tile: per-element bf16 residual loads and
    fragment-layout stores were the binding cost of this body at K=1152
    (vision o_proj / ffn_down), see `_matmul_bias`. Aliasing is safe because
    the whole R tile is read into smem before any element of C is written.
    """
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
        R_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
        Bias_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(R[pid_m * BLOCK_M, pid_n * BLOCK_N], R_shared)
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = (
                C_local[i, j]
                + Bias_local[j].astype(accum_dtype)
                + R_shared[i, j].astype(accum_dtype)
            )

        T.copy(C_local, R_shared)
        T.copy(R_shared, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


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
# `tl_rms_factor` is the action expert's and lives in the shared component
# package (`gemma_expert`), which owns the expert chain on this device; it is
# re-exported here so this Target's wrappers and its lab scripts keep one
# import. The two below accumulate squares over K in chunks rather than in one
# whole-row fragment, which would spill the register file at the block sizes
# used here.
# ---------------------------------------------------------------------------
from .....gemma_expert.backends.tilelang.kernels.rms import (  # noqa: E402,F401
    tl_rms_factor,
)


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

    The result is staged through a bf16 shared tile before the global store:
    the 32 MB output per call written straight from the fragment cost ~5% of
    the kernel (219 -> 208 us/layer at the production config, job 588745, cold
    weights); the extra 32 KB tile fits beside the two 2-stage weight rings.
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
        C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
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
            C1_local[i, j] = _gelu(C1_local[i, j]) * C2_local[i, j]
        T.copy(C1_local, C_shared)
        T.copy(C_shared, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


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


@kernel(warp_spec=False)
def tl_matmul_rope_scatter(A, W, Rope, OutQ, OutK, OutV, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                           NUM_STAGES: int, THREADS: int, HEAD_DIM: int, NUM_HEADS: int):
    """Encoder QKV: project, round to bf16, rotate Q/K pairs in fp32, scatter to Q/K/V.

    A is the already-normalized (M, K) activation, W (K, N) with
    N = (NUM_HEADS + 2) * HEAD_DIM packed as [Q | K | V], Rope (M, HEAD_DIM)
    interleaved (cos, sin). OutQ is (M, NUM_HEADS * HEAD_DIM), OutK and OutV
    are (M, HEAD_DIM); every row < M of each is written, nothing else.

    The order is round-then-rotate, the opposite of `tl_qkv_gemm_rope`: the
    fp32 accumulator is staged through a bf16 shared tile, which is exactly
    the rounding the former packed projection buffer applied, and the rotation
    runs in fp32 on those rounded values before the final bf16 store. The
    output is therefore bit-identical to the retired GEMM + rope-scatter pair.

    Requires BLOCK_N to divide HEAD_DIM so a tile never straddles a head or
    the Q/K/V boundary. M need not be a multiple of BLOCK_M: the rope read is
    row-guarded and the stores rely on T.copy's boundary predication. Warp
    specialization is off: at the encoder shape the no-WS 128x64 tiling is
    what makes the epilogue free (see the owning Agent Note).
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    q_dim = NUM_HEADS * HEAD_DIM
    A: T.Tensor((M, K), dtype)
    W: T.Tensor((K, N), dtype)
    Rope: T.Tensor((M, HEAD_DIM), dtype)
    OutQ: T.Tensor((M, q_dim), dtype)
    OutK: T.Tensor((M, HEAD_DIM), dtype)
    OutV: T.Tensor((M, HEAD_DIM), dtype)
    rope_cols = (NUM_HEADS + 1) * HEAD_DIM

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W[ko * BLOCK_K, pid_n * BLOCK_N], W_shared)
            T.gemm(A_shared, W_shared, C_local)

        # The bf16 staging tile IS the packed-buffer rounding of the old route.
        T.copy(C_local, C_shared)

        n0 = pid_n * BLOCK_N
        if n0 < rope_cols:
            for i, p in T.Parallel(BLOCK_M, BLOCK_N // 2):
                row = pid_m * BLOCK_M + i
                if row < M:
                    rj0 = (n0 + 2 * p) % HEAD_DIM
                    x0 = C_shared[i, 2 * p].astype(accum_dtype)
                    x1 = C_shared[i, 2 * p + 1].astype(accum_dtype)
                    cos_v = Rope[row, rj0].astype(accum_dtype)
                    sin_v = Rope[row, rj0 + 1].astype(accum_dtype)
                    C_shared[i, 2 * p] = (x0 * cos_v - x1 * sin_v).astype(dtype)
                    C_shared[i, 2 * p + 1] = (x1 * cos_v + x0 * sin_v).astype(dtype)

        if n0 < q_dim:
            T.copy(C_shared, OutQ[pid_m * BLOCK_M, n0])
        elif n0 < q_dim + HEAD_DIM:
            T.copy(C_shared, OutK[pid_m * BLOCK_M, n0 - q_dim])
        else:
            T.copy(C_shared, OutV[pid_m * BLOCK_M, n0 - q_dim - HEAD_DIM])


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
