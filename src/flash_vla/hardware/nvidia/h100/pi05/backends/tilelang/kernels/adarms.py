"""AdaRMSNorm decoder kernels.

The one decision these four kernels share is where the AdaRMSNorm scale `s` is
applied; it is spelled out below, because `s` is indexed by the contraction axis
and so cannot ride the epilogue the way `b` and `g` do.

Pi0.5's action expert replaces plain RMSNorm with AdaRMSNorm. The modulation
Dense layers are folded away at checkpoint load (`models.pi05.weights.fold`),
leaving three constant per-(step, layer) vectors:

    s = (1 + scale)   indexed by K (hidden)   -> scales the GEMM's A operand
    b = shift @ W     indexed by N (output)   -> a plain bias
    g = gate          indexed by N (output)   -> a multiply in the residual

so that `x_hat @ W = rstd(x) * ((x * s) @ W) + b` and `y = x + (h @ W) * g`.

Two of those are free: `b` and `g` are indexed by the output axis, so they ride
the epilogue exactly as `Bias_local[j]` already does in `_matmul_bias`. The
third is not. `s` is indexed by the contraction axis, so it sits *inside* the
reduction and has to meet A before `T.gemm` reads it -- the reason this file
exists at all.

The design decision, from the spec: apply `s` to the A tile in shared memory,
inside the mainloop, between the tile copy and the GEMM. That is not a new
technique here. `base.tl_scaled_gate` already scales `A_shared` in place for the
per-row RMS factor; AdaRMS adds one factor to that same line.

Where `F` goes differs per kernel and follows each origin kernel rather than a
uniform rule, because `F` is per-row and therefore commutes with the reduction:
`tl_ada_qkv_gemm_rope` keeps it in the epilogue where `tl_qkv_gemm_rope` had it,
and `tl_ada_scaled_gate` keeps it in the mainloop where `tl_scaled_gate` had it.
Recorded as a deviation in the spec.
"""
from __future__ import annotations

import tilelang.language as T

from .base import _gelu, kernel, variant


@kernel
def tl_ada_qkv_gemm_rope(A, F, S, W, Bias, Rope, OutQ, OutK, OutV, BLOCK_M: int, BLOCK_N: int,
                         BLOCK_K: int, NUM_STAGES: int, THREADS: int, HEAD_DIM: int,
                         NUM_HEADS: int):
    """Decoder QKV under AdaRMSNorm: scale A by S, project, add bias, rotate, scatter.

    `tl_qkv_gemm_rope` with two additions. `S` scales the A tile inside the
    mainloop; `Bias` is added to the fp32 accumulator **between the F multiply
    and the rotation**.

    That ordering is load-bearing and silent if wrong. The folded form is
    `q = rstd * ((x*s) @ W_q) + b` and only then `RoPE(q)`; adding `b` after the
    rotation computes a different function that still looks plausible. Because
    the F multiply happens before the Q/K/V branch, one bias add in the same
    place covers all three slices, and only the Q and K branches rotate.

    `S` is staged one BLOCK_K slice per iteration, alongside A and W, rather
    than held whole. Holding it whole deadlocks: under warp specialization a
    global->shared `T.copy` outside `T.Pipelined` lowers to a producer-warp TMA
    with no matching consumer arrival, so the kernel compiles, launches and never
    returns. The origin kernels only load vectors into *fragments*, which is a
    per-thread load with no barrier, so there was no precedent for the shape.
    Staged, `S` is the same kind of operand as A and W.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    q_dim = NUM_HEADS * HEAD_DIM
    A: T.Tensor((M, K), dtype)
    F: T.Tensor((M,), dtype)
    S: T.Tensor((K,), dtype)
    W: T.Tensor((K, N), dtype)
    Bias: T.Tensor((N,), dtype)
    Rope: T.Tensor((M, HEAD_DIM), dtype)
    OutQ: T.Tensor((M, q_dim), dtype)
    OutK: T.Tensor((M, HEAD_DIM), dtype)
    OutV: T.Tensor((M, HEAD_DIM), dtype)
    rope_cols = (NUM_HEADS + 1) * HEAD_DIM

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        S_shared = T.alloc_shared((BLOCK_K,), dtype)
        C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), accum_dtype)
        F_local = T.alloc_fragment((BLOCK_M,), dtype)
        Bias_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(F[pid_m * BLOCK_M], F_local)
        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W[ko * BLOCK_K, pid_n * BLOCK_N], W_shared)
            T.copy(S[ko * BLOCK_K], S_shared)
            # spec: mainloop, per_iter_math -- the one place s can be applied
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                A_shared[i, j] = A_shared[i, j] * S_shared[j]
            T.gemm(A_shared, W_shared, C_local)

        # spec: epilogue -- F is per-row and rides here; bias lands before RoPE
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = (C_local[i, j] * F_local[i].astype(accum_dtype)
                             + Bias_local[j].astype(accum_dtype))
        T.copy(C_local, C_shared)

        for i, p in T.Parallel(BLOCK_M, BLOCK_N // 2):
            row = pid_m * BLOCK_M + i
            j0 = pid_n * BLOCK_N + 2 * p
            in_rope = j0 < rope_cols
            rj0 = j0 % HEAD_DIM
            x0 = C_shared[i, 2 * p]
            x1 = C_shared[i, 2 * p + 1]
            cos_v = Rope[row, rj0].astype(accum_dtype)
            sin_v = Rope[row, rj0 + 1].astype(accum_dtype)
            C_shared[i, 2 * p] = T.if_then_else(in_rope, x0 * cos_v - x1 * sin_v, x0)
            C_shared[i, 2 * p + 1] = T.if_then_else(in_rope, x1 * cos_v + x0 * sin_v, x1)

        n0 = pid_n * BLOCK_N
        if n0 < q_dim:
            T.copy(C_shared, OutQ[pid_m * BLOCK_M, n0])
        elif n0 < q_dim + HEAD_DIM:
            T.copy(C_shared, OutK[pid_m * BLOCK_M, n0 - q_dim])
        else:
            T.copy(C_shared, OutV[pid_m * BLOCK_M, n0 - q_dim - HEAD_DIM])


@kernel
def tl_ada_scaled_gate(A, F, S, W1, W2, B1, B2, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                       NUM_STAGES: int, THREADS: int):
    """Decoder gated FFN under AdaRMSNorm.

    `C = gelu_tanh((A*F*S) @ W1 + b1) * ((A*F*S) @ W2 + b2)`.

    Both constraints from `tl_scaled_gate` carry over unchanged and are still
    load-bearing. Warp specialization must stay ON: the dual GEMM reuses one
    `A_shared` across the W1 and W2 pipeline stages, and the no-WS pipeline
    planner rejects the overlapping-buffer writes outright. And the tile config
    stays at BLOCK_M=64, BLOCK_N=32, BLOCK_K=256, NUM_STAGES=3, THREADS=128 --
    correctness there is tiling-dependent, BLOCK_M=32 among the tilings that
    produce garbage rather than failing.

    The in-loop scale stays in bf16, as it did for `F` alone, both to match the
    original bit-for-bit on the `F` part and because bf16 packs two multiplies
    per lane.

    v1 builds on the two-kernel path rather than `fused_norm.tl_fused_rms_gate`:
    that kernel accumulates the row sum of squares from the same shared tile
    `T.gemm` consumes, and AdaRMSNorm needs the tile unscaled for the norm and
    scaled for the GEMM. The factor arrives here as `F` from a separate
    `tl_rms_factor` launch instead. See the spec for what v2 would have to fit.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    F: T.Tensor((M,), dtype)
    S: T.Tensor((K,), dtype)
    W1: T.Tensor((K, N), dtype)
    W2: T.Tensor((K, N), dtype)
    B1: T.Tensor((N,), dtype)
    B2: T.Tensor((N,), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W1_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        W2_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        S_shared = T.alloc_shared((BLOCK_K,), dtype)
        F_local = T.alloc_fragment((BLOCK_M,), dtype)
        B1_local = T.alloc_fragment((BLOCK_N,), dtype)
        B2_local = T.alloc_fragment((BLOCK_N,), dtype)
        C1_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        C2_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(F[pid_m * BLOCK_M], F_local)
        T.copy(B1[pid_n * BLOCK_N], B1_local)
        T.copy(B2[pid_n * BLOCK_N], B2_local)
        T.clear(C1_local)
        T.clear(C2_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W1[ko * BLOCK_K, pid_n * BLOCK_N], W1_shared)
            T.copy(W2[ko * BLOCK_K, pid_n * BLOCK_N], W2_shared)
            T.copy(S[ko * BLOCK_K], S_shared)
            # spec: mainloop, per_iter_math -- F (per-row) and S (per-column) together
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                A_shared[i, j] = A_shared[i, j] * F_local[i] * S_shared[j]
            T.gemm(A_shared, W1_shared, C1_local)
            T.gemm(A_shared, W2_shared, C2_local)

        # spec: epilogue -- each branch takes its own bias before the activation
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C1_local[i, j] = C1_local[i, j] + B1_local[j].astype(accum_dtype)
            C2_local[i, j] = C2_local[i, j] + B2_local[j].astype(accum_dtype)
            C1_local[i, j] = _gelu(C1_local[i, j])
            C1_local[i, j] = C1_local[i, j] * C2_local[i, j]

        T.copy(C1_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


def _matmul_gated_res(A, B, G, R, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                      THREADS: int):
    """C = R + (A @ B) * G[None, :], the AdaRMSNorm gated residual.

    `_matmul_res` with one per-N multiply in the epilogue, structurally identical
    to the `Bias_local[j]` add already in `_matmul_bias`. No mainloop change and
    no new shared buffer.

    The gate cannot be folded into B. It changes every flow step, so folding
    would need ten copies of `decoder_attn_o_w` and `decoder_ffn_down_w` --
    2.26 GB against a decoder that streams 629 MB per step.

    As in `_matmul_res`, the wrapper passes one buffer as both R and C for an
    in-place residual: each thread reads its R element into the accumulator
    before the copy writes that same element, so the aliasing is safe.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    G: T.Tensor((N,), dtype)
    R: T.Tensor((M, N), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        G_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(G[pid_n * BLOCK_N], G_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        # spec: epilogue -- gate then residual, in that order
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = (C_local[i, j] * G_local[j].astype(accum_dtype)
                             + R[pid_m * BLOCK_M + i, pid_n * BLOCK_N + j].astype(accum_dtype))

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


# Warp specialization off, matching `tl_matmul_res`: at BLOCK_M=16 the decoder's
# residual GEMM is far below one wave, where the producer warp sits idle and
# still costs warps and mbarrier traffic. The spec's `warp_groups` section
# assumed the decorator default of on; recorded as a deviation.
tl_matmul_gated_res = variant(_matmul_gated_res, "tl_matmul_gated_res", warp_spec=False)


@kernel
def tl_fd_flat_split_mask(Qt, Kt, Vt, Mask, PartialO, GLSE, BLOCK_M: int, BLOCK_N: int,
                          NUM_SPLIT: int, NUM_STAGES: int, THREADS: int, QPAD: int, KEYS: int,
                          CHUNK: int, CHUNK_BLOCKS: int, SCALE_L2: float):
    """FlashDecoding split with an additive per-key mask.

    `tl_fd_flat_split` with its mask initializer replaced. Pi0's predicate,
    `(gi >= NUM_HEADS) | (j <= ENC_LEN)`, exists only to stop the state token
    attending to the action block; Pi0.5 has no state token, so it is dead
    rather than replaced and all `M_flat` query rows share one mask row.

    What replaces it is prompt padding, which is a hole in the middle of the key
    range -- valid prefix, then padding, then the suffix -- rather than a suffix
    of it, so a length bound cannot express it and an additive vector can. The
    vector is the `prefix_mask_bias` buffer the prefix pass already fills.

    The `< KEYS` guard stays: it bounds the mask load as well as the key.
    `NUM_HEADS` is gone from the signature -- it existed only for the dead
    predicate, and a dead compile-time constant still keys the compile cache.

    Everything else is unchanged, including the three constraints that govern
    this kernel. BLOCK_M must be 64 -- smaller tiles hit a fragment-layout
    conflict between `s` and `s_cast` that fails to lower. No split may start at
    or past KEYS, since a TMA box whose first row is out of bounds reads garbage
    rather than zeros; the wrapper shrinks NUM_SPLIT until every split is
    non-empty. And every expression inside the pipelined body must stay inlined,
    which is why the mask is indexed in place rather than bound to a temporary.
    """
    M, HD = T.const("M, HD")
    dtype = T.bfloat16
    accum_dtype = T.float32
    Qt: T.Tensor((M, HD), dtype)
    Kt: T.Tensor((KEYS, HD), dtype)
    Vt: T.Tensor((KEYS, HD), dtype)
    Mask: T.Tensor((KEYS,), dtype)
    PartialO: T.Tensor((NUM_SPLIT, QPAD, HD), dtype)
    GLSE: T.Tensor((NUM_SPLIT, QPAD), accum_dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_M), NUM_SPLIT, threads=THREADS) as (bx, bz):
        q_sh = T.alloc_shared((BLOCK_M, HD), dtype)
        k_sh = T.alloc_shared((BLOCK_N, HD), dtype)
        v_sh = T.alloc_shared((BLOCK_N, HD), dtype)
        s = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        s_cast = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
        acc_o = T.alloc_fragment((BLOCK_M, HD), accum_dtype)
        m = T.alloc_fragment((BLOCK_M,), accum_dtype)
        mp = T.alloc_fragment((BLOCK_M,), accum_dtype)
        l = T.alloc_fragment((BLOCK_M,), accum_dtype)
        p_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
        alpha = T.alloc_fragment((BLOCK_M,), accum_dtype)

        T.copy(Qt[bx * BLOCK_M:(bx + 1) * BLOCK_M, :], q_sh, disable_tma=True)
        T.clear(acc_o)
        T.fill(l, 0.0)
        T.fill(m, -T.infinity(accum_dtype))

        for ki in T.Pipelined(CHUNK_BLOCKS, num_stages=NUM_STAGES):
            T.copy(Kt[bz * CHUNK + ki * BLOCK_N:bz * CHUNK + (ki + 1) * BLOCK_N, :], k_sh)
            # spec: variant D diff -- the state-token predicate becomes a vector load
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                s[i, j] = T.if_then_else(
                    (bz * CHUNK + ki * BLOCK_N + j) < KEYS,
                    Mask[bz * CHUNK + ki * BLOCK_N + j].astype(accum_dtype),
                    -T.infinity(accum_dtype))
            T.gemm(q_sh, k_sh, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(Vt[bz * CHUNK + ki * BLOCK_N:bz * CHUNK + (ki + 1) * BLOCK_N, :], v_sh)
            T.copy(m, mp)
            T.fill(m, -T.infinity(accum_dtype))
            T.reduce_max(s, m, dim=1, clear=False)
            for i in T.Parallel(BLOCK_M):
                m[i] = T.max(m[i], mp[i])
                m[i] = T.max(m[i], T.cast(-1e38, accum_dtype))
                alpha[i] = T.exp2(mp[i] * SCALE_L2 - m[i] * SCALE_L2)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                s[i, j] = T.exp2(s[i, j] * SCALE_L2 - m[i] * SCALE_L2)
            T.reduce_sum(s, p_sum, dim=1, clear=True)
            for i in T.Parallel(BLOCK_M):
                l[i] = l[i] * alpha[i] + p_sum[i]
            for i, j in T.Parallel(BLOCK_M, HD):
                acc_o[i, j] = acc_o[i, j] * alpha[i]
            T.copy(s, s_cast)
            T.gemm(s_cast, v_sh, acc_o, policy=T.GemmWarpPolicy.FullRow)

        for i in T.Parallel(BLOCK_M):
            l[i] = T.max(l[i], T.cast(1e-30, accum_dtype))
        for i, j in T.Parallel(BLOCK_M, HD):
            acc_o[i, j] = acc_o[i, j] / l[i]
        T.copy(acc_o, PartialO[bz, bx * BLOCK_M:(bx + 1) * BLOCK_M, :], disable_tma=True)
        for i in T.Parallel(BLOCK_M):
            GLSE[bz, bx * BLOCK_M + i] = T.log2(l[i]) + m[i] * SCALE_L2
