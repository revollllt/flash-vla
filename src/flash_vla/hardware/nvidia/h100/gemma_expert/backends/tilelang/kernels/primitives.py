"""Expert kernel builders shared by Pi0 and Pi0.5.

Targets register these raw builders in their own KernelSet; shapes, routing and
JIT pass choices stay with each Target. Builders retain the original arithmetic.
"""
from __future__ import annotations

import tilelang.language as T

GELU_C0 = 1.5957691216057308
GELU_C1 = 0.044715


def _gelu(v):
    return v * (1.0 / (1.0 + T.__exp(-(GELU_C0 * v * (1.0 + GELU_C1 * v * v)))))


def tl_scaled_gate(A, F, W1, W2, C, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, NUM_STAGES: int,
                   THREADS: int):
    """C = gelu_tanh((A * F) @ W1) * ((A * F) @ W2), the decoder's fused RMS + gated FFN.

    Two constraints, both load-bearing:

    Warp specialization must stay ON. The dual GEMM reuses one A_shared across
    the W1 and W2 pipeline stages, and the no-WS pipeline planner rejects the
    overlapping-buffer writes outright ("Stage 0 and 3 both write A_shared").

    The tile config is LOCKED at BLOCK_M=64, BLOCK_N=32, BLOCK_K=256,
    NUM_STAGES=3, THREADS=128. Correctness here is tiling-dependent -- some
    tilings, BLOCK_M=32 among them, produce garbage rather than failing. Any
    re-tune must re-validate numerically, not just time the result.

    The in-loop `A_shared * F` scale stays in bf16 to match Triton bit-for-bit
    and because it packs two multiplies per lane.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    F: T.Tensor((M,), dtype)
    W1: T.Tensor((K, N), dtype)
    W2: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W1_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        W2_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        F_local = T.alloc_fragment((BLOCK_M,), dtype)
        C1_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
        C2_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(F[pid_m * BLOCK_M], F_local)
        T.clear(C1_local)
        T.clear(C2_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W1[ko * BLOCK_K, pid_n * BLOCK_N], W1_shared)
            T.copy(W2[ko * BLOCK_K, pid_n * BLOCK_N], W2_shared)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                A_shared[i, j] = A_shared[i, j] * F_local[i]
            T.gemm(A_shared, W1_shared, C1_local)
            T.gemm(A_shared, W2_shared, C2_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C1_local[i, j] = _gelu(C1_local[i, j])
            C1_local[i, j] = C1_local[i, j] * C2_local[i, j]

        T.copy(C1_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])


def tl_scaled_matmul_bias_res(A, F, B, Bias, R, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                              NUM_STAGES: int, THREADS: int):
    """C = R + bias + (A * F) @ B, the decoder's RMS + out-projection + residual.

    Superseded on the default path by `fused_norm.tl_fused_rms_matmul_bias_res`;
    kept as the two-kernel reference that matches Triton bit-for-bit.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    F: T.Tensor((M,), dtype)
    B: T.Tensor((K, N), dtype)
    Bias: T.Tensor((N,), dtype)
    R: T.Tensor((M, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        F_local = T.alloc_fragment((BLOCK_M,), dtype)
        Bias_local = T.alloc_fragment((BLOCK_N,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(F[pid_m * BLOCK_M], F_local)
        T.copy(Bias[pid_n * BLOCK_N], Bias_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, pid_n * BLOCK_N], B_shared)
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                A_shared[i, j] = (A_shared[i, j].astype(accum_dtype) * F_local[i].astype(accum_dtype)).astype(dtype)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = (
                C_local[i, j]
                + Bias_local[j].astype(accum_dtype)
                + R[pid_m * BLOCK_M + i, pid_n * BLOCK_N + j].astype(accum_dtype)
            )

        T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])

    return C


def tl_qkv_gemm_rope(A, F, W, Rope, OutQ, OutK, OutV, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                     NUM_STAGES: int, THREADS: int, HEAD_DIM: int, NUM_HEADS: int):
    """Decoder QKV: scale by F, project, rotate Q/K pairs in fp32, scatter to Q/K/V.

    RoPE is applied to the fp32 accumulator before the bf16 rounding, matching
    the upstream decoder kernel. The rotation reads column pairs (2p, 2p+1),
    which is not a legal access pattern on a register fragment, so the scaled
    accumulator is staged through shared memory first.

    Requires BLOCK_N to divide HEAD_DIM, so a tile never straddles a head
    boundary or the Q/K/V boundary.
    """
    M, N, K = T.const("M, N, K")
    dtype = T.bfloat16
    accum_dtype = T.float32
    q_dim = NUM_HEADS * HEAD_DIM
    A: T.Tensor((M, K), dtype)
    F: T.Tensor((M,), dtype)
    W: T.Tensor((K, N), dtype)
    Rope: T.Tensor((M, HEAD_DIM), dtype)
    OutQ: T.Tensor((M, q_dim), dtype)
    OutK: T.Tensor((M, HEAD_DIM), dtype)
    OutV: T.Tensor((M, HEAD_DIM), dtype)
    rope_cols = (NUM_HEADS + 1) * HEAD_DIM

    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (pid_n, pid_m):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        W_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), accum_dtype)
        F_local = T.alloc_fragment((BLOCK_M,), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.copy(F[pid_m * BLOCK_M], F_local)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
            T.copy(A[pid_m * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(W[ko * BLOCK_K, pid_n * BLOCK_N], W_shared)
            T.gemm(A_shared, W_shared, C_local)

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            C_local[i, j] = C_local[i, j] * F_local[i].astype(accum_dtype)
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


def tl_softmax_mask0(S, O, BLOCK_M: int, THREADS: int, KEYS: int, NUM_HEADS: int, ENC_LEN: int,
                     FRAG_W: int):
    """Row softmax with the Pi0 mask: key j is dropped when row < NUM_HEADS and j > ENC_LEN.

    S and O are the real (queries, keys) tensors. FRAG_W is the softmax fragment
    width and must be a power of two at least KEYS -- it is decoupled from the
    tensor width on purpose, mirroring how Triton uses a power-of-two BLOCK_SIZE
    plus a mask. Columns past KEYS read as zero and are masked out again here.
    """
    Q, N = T.const("Q, N")
    dtype = T.bfloat16
    accum_dtype = T.float32
    S: T.Tensor((Q, N), dtype)
    O: T.Tensor((Q, N), dtype)
    NEG = T.float32(-3.0e38)

    with T.Kernel(T.ceildiv(Q, BLOCK_M), threads=THREADS) as pid_m:
        S_frag = T.alloc_fragment((BLOCK_M, FRAG_W), accum_dtype)
        row_max = T.alloc_fragment((BLOCK_M,), accum_dtype)
        row_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)

        T.copy(S[pid_m * BLOCK_M, 0], S_frag)
        for i, j in T.Parallel(BLOCK_M, FRAG_W):
            gi = pid_m * BLOCK_M + i
            keep = (j < KEYS) & ((gi >= NUM_HEADS) | (j <= ENC_LEN))
            S_frag[i, j] = T.if_then_else(keep, S_frag[i, j], NEG)
        T.reduce_max(S_frag, row_max, dim=1, clear=True)
        for i, j in T.Parallel(BLOCK_M, FRAG_W):
            S_frag[i, j] = T.__exp(S_frag[i, j] - row_max[i])
        T.reduce_sum(S_frag, row_sum, dim=1, clear=True)
        for i, j in T.Parallel(BLOCK_M, FRAG_W):
            S_frag[i, j] = S_frag[i, j] / row_sum[i]
        T.copy(S_frag, O[pid_m * BLOCK_M, 0])


def tl_fd_flat_split(Qt, Kt, Vt, PartialO, GLSE, BLOCK_M: int, BLOCK_N: int, NUM_SPLIT: int,
                     NUM_STAGES: int, THREADS: int, QPAD: int, KEYS: int, ENC_LEN: int,
                     NUM_HEADS: int, CHUNK: int, CHUNK_BLOCKS: int, SCALE_L2: float):
    """FlashDecoding split: online softmax over this split's slice of the keys.

    Pi0 decoder attention is multi-query -- all NUM_HEADS query heads share the
    one KV head -- so the token and head axes collapse into a single flat query
    axis, and tiling it at BLOCK_M is already the token x head split. Crossed
    with the key split that gives a 2-D grid and keeps every copy 2-D contiguous.

    Writes a locally normalized partial plus the local log-sum-exp in the log2
    domain, with scale * log2(e) folded into SCALE_L2 so the exponent costs one
    instruction. Rows that are fully masked end with l = 0; clamping l makes the
    partial zero and the lse about -9e36, so the combine weight is exactly zero
    and no NaN reaches the merge.

    Three constraints:
      BLOCK_M must be 64 -- smaller tiles hit a fragment-layout conflict between
      s and s_cast that fails to lower.
      No split may start at or past KEYS. A TMA box whose first row is out of
      bounds reads garbage rather than zeros, and the garbage survives into the
      partials; the wrapper shrinks NUM_SPLIT until every split is non-empty.
      Every expression inside the pipelined body must stay inlined. A named
      temporary lowers to a bind statement that the warp-specialization role
      pass cannot classify, and the compile aborts.
    """
    M, HD = T.const("M, HD")
    dtype = T.bfloat16
    accum_dtype = T.float32
    Qt: T.Tensor((M, HD), dtype)
    Kt: T.Tensor((KEYS, HD), dtype)
    Vt: T.Tensor((KEYS, HD), dtype)
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
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                s[i, j] = T.if_then_else(
                    ((bz * CHUNK + ki * BLOCK_N + j) < KEYS)
                    & (((bx * BLOCK_M + i) >= NUM_HEADS)
                       | ((bz * CHUNK + ki * BLOCK_N + j) <= ENC_LEN)),
                    T.cast(0, accum_dtype), -T.infinity(accum_dtype))
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


def tl_fd_flat_combine(PartialO, GLSE, O, BLOCK_M: int, THREADS: int, QPAD: int, NUM_SPLIT: int):
    """Merge the FlashDecoding partials, weighting each by its log2-domain lse.

    BLOCK_M must divide the flat query count and stay at most 4: the vectorizer
    fails to prove an index equality at BLOCK_M=8 for several split counts.
    """
    M, HD = T.const("M, HD")
    dtype = T.bfloat16
    accum_dtype = T.float32
    PartialO: T.Tensor((NUM_SPLIT, QPAD, HD), dtype)
    GLSE: T.Tensor((NUM_SPLIT, QPAD), accum_dtype)
    O: T.Tensor((M, HD), dtype)

    with T.Kernel(T.ceildiv(M, BLOCK_M), 1, threads=THREADS) as (bx, by):
        lse = T.alloc_fragment((BLOCK_M, NUM_SPLIT), accum_dtype)
        lmax = T.alloc_fragment((BLOCK_M,), accum_dtype)
        lsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
        o_acc = T.alloc_fragment((BLOCK_M, HD), accum_dtype)

        for i, ks in T.Parallel(BLOCK_M, NUM_SPLIT):
            lse[i, ks] = GLSE[ks, bx * BLOCK_M + i]
        T.reduce_max(lse, lmax, dim=1, clear=True)
        for i in T.Parallel(BLOCK_M):
            lmax[i] = T.max(lmax[i], T.cast(-1e38, accum_dtype))
        for i, ks in T.Parallel(BLOCK_M, NUM_SPLIT):
            lse[i, ks] = T.exp2(lse[i, ks] - lmax[i])
        T.reduce_sum(lse, lsum, dim=1, clear=True)
        T.clear(o_acc)
        for ks in T.serial(NUM_SPLIT):
            for i, d in T.Parallel(BLOCK_M, HD):
                o_acc[i, d] += (lse[i, ks] / lsum[i]) * PartialO[ks, bx * BLOCK_M + i, d].astype(accum_dtype)
        T.copy(o_acc, O[bx * BLOCK_M:(bx + 1) * BLOCK_M, :], disable_tma=True)
