---
spec_version: 1
kernel: pi05_adarms_decoder          # four variants, one design; see `variants`
status: approved
approved_by: revollllt
approved_at: 2026-08-20
source: derived from src/flash_vla/hardware/nvidia/h100/pi05/backends/tilelang/kernels/base.py

# ---------------------------------------------------------------- 0. problem
arch: sm90a
problem:
  op: >
    Pi0.5 action-expert decoder call sites under AdaRMSNorm. The modulation Dense
    layers are folded away at load (models/pi05/weights.fold), leaving three
    constant per-(step, layer) vectors:
      s = (1 + scale)   indexed by K (hidden)   -> scales the GEMM's A operand
      b = shift @ W     indexed by N (output)   -> a plain bias
      g = gate          indexed by N (output)   -> a multiply in the residual
    So  x_hat @ W  =  rstd(x) * ((x * s) @ W) + b   and   y = x + (h @ W) * g.
  dims: {M: 50, width: 1024, ffn: 4096, qkv: 2560, layers: 18, steps: 10, keys: 1018}
  dynamic: []                        # every extent is baked at capture; only values change
  dtypes: {a: bf16, b: bf16, acc: f32, d: bf16, s: bf16, bias: bf16, gate: bf16}
  layouts: {a: "row(M,K)", b: "row(K,N)", d: "row(M,N)"}
  regime: >
    Latency at tiny M. M=50 with no state token (Pi0 had 51). Every variant is
    under one wave on 132 SMs and every one is far below the H100 ridge point of
    295 FLOP/B, so all four are weight-bandwidth bound, not MMA bound. Tiles are
    chosen for occupancy and pipeline depth, not for arithmetic intensity.

# ---------------------------------------------------- shared design decision
adarms_placement:
  decision: >
    Apply s to the A tile in shared memory, inside the consuming GEMM's
    mainloop, immediately after the tile lands and before T.gemm reads it.
  precedent: >
    This is not a new technique here. tl_scaled_gate already does exactly this
    for the per-row RMS factor (base.py:495-497):
        for i, j in T.Parallel(BLOCK_M, BLOCK_K):
            A_shared[i, j] = A_shared[i, j] * F_local[i]
    The AdaRMS change is one extra factor on that same line -- F_local[i]
    becomes F_local[i] * S_shared[ko*BLOCK_K + j]. The line is already tuned,
    already numerically validated, and already lowers under warp specialization.
  rejected_alternative: >
    Fusing s into the producing residual kernel's epilogue. _DEC_RESIDUAL is
    BLOCK_N=32, so N=1024 is split across 32 CTAs and no single CTA holds a
    whole row; emitting the RMS row reduction there would need a cross-CTA
    reduction. Moot regardless, since the in-mainloop path is the proven one.
  s_residency: >
    S is staged per mainloop iteration, one BLOCK_K slice at a time, alongside A
    and W: S_shared is (BLOCK_K,) and the body does T.copy(S[ko*BLOCK_K],
    S_shared) before the scale. 256 B per stage for variant A, 512 B for B.

    The first version of this spec said the opposite -- hold the whole 1024-entry
    vector in shared memory, loaded once before the mainloop -- and that
    deadlocked on the GPU. Under warp specialization a global->shared T.copy
    lowers to a producer-warp TMA plus an mbarrier; placed outside T.Pipelined it
    has no matching consumer arrival, so the kernel compiled, launched, and never
    returned. The origin kernels only ever load vectors into *fragments*
    (F_local, Bias_local), which is a per-thread load with no barrier, so there
    was no precedent for the shape and the argument for it -- "keep the pipelined
    body free of an extra copy" -- had it exactly backwards.

    Staging it makes S the same shape of operand as A and W, which is the only
    global->shared form this codebase has validated.
  cost: >
    One extra bf16 multiply per A element per stage, plus a 256-512 B staged
    copy. Variant B, the largest: 64x256 per stage x 4 stages x 128 CTAs =
    8.4 M bf16 multiplies against 6.29 GB of weight traffic per 10-step decode.
    Not measurable.

# ------------------------------------------------------------------ variants
variants:
  - id: A                          # decoder_norm_qkv_rope
    name: tl_ada_qkv_gemm_rope
    from: tl_qkv_gemm_rope         # base.py:608
    shape: {M: 50, N: 2560, K: 1024}
    config: {BLOCK_M: 64, BLOCK_N: 32, BLOCK_K: 128, NUM_STAGES: 4, THREADS: 128}
    adds: [s on A, b in epilogue before RoPE]
    note: >
      The bias goes between the F multiply and the rotation. The folded form is
      q = rstd * ((x*s) @ W_q) + b, then RoPE(q); adding b after the rotation is
      a different function and a silent one. Because the existing kernel applies
      F to C_local before the Q/K/V branch, one bias add in the same place
      covers all three slices, and only the Q and K branches then rotate.
  - id: B                          # decoder_norm_gated_ffn
    name: tl_ada_scaled_gate
    from: tl_scaled_gate           # base.py:454
    shape: {M: 50, N: 4096, K: 1024}
    config: {BLOCK_M: 64, BLOCK_N: 32, BLOCK_K: 256, NUM_STAGES: 3, THREADS: 128}
    adds: [s on A, b_gate and b_up in epilogue before gelu]
    note: >
      v1 builds on tl_scaled_gate, not on fused_norm.tl_fused_rms_gate. The
      fused kernel accumulates the row sum of squares from the same shared tile
      T.gemm consumes, and AdaRMSNorm needs that tile unscaled for the norm and
      scaled for the GEMM; the gemms run before the accumulation, so in-place
      scaling is wrong on either side of it. v1 pays one extra tl_rms_factor
      node per layer instead. Recovering the fusion is a v2 item.
  - id: C                          # decoder_out_proj_residual, decoder_ffn_down_residual
    name: tl_matmul_gated_res
    from: tl_matmul_res
    shape: {M: 50, N: 1024, K: [2048, 4096]}
    config: {BLOCK_M: 16, BLOCK_N: 32, BLOCK_K: 256, NUM_STAGES: 4, THREADS: 128}
    adds: [g in epilogue]
    note: >
      One kernel, two call sites, differing only in K. No s and no F: the
      residual GEMM consumes an attention or FFN output, not a normed
      activation. g cannot be folded into W -- it changes per flow step, so
      folding would need ten copies of decoder_attn_o_w and decoder_ffn_down_w,
      2.26 GB.
  - id: D                          # decoder_attention
    name: tl_fd_flat_split_mask
    from: tl_fd_flat_split         # base.py:801
    shape: {M_flat: 400, HD: 256, KEYS: 1018}
    config: {BLOCK_M: 64, BLOCK_N: 64, NUM_SPLIT: 6, NUM_STAGES: 1, THREADS: 128,
             CHUNK: 192, CHUNK_BLOCKS: 3, QPAD: 448}
    adds: [additive per-key mask vector replacing the state-token predicate]
    note: >
      Pi0's predicate (gi >= NUM_HEADS) | (j <= ENC_LEN) exists only to stop the
      state token attending to the action block. Pi0.5 has no state token, so
      all 400 flat query rows share one mask row. What replaces it is prompt
      padding, which is a hole in the middle of the key range -- valid prefix
      [0,903), padding [903,968), suffix [968,1018) -- not a suffix of it, so a
      length bound cannot express it and an additive vector can.
      NUM_SPLIT falls 7 -> 6 through the existing _num_splits guard at
      KEYS=1018; the wrapper already does this and needs no change.

unchanged:
  - name: decoder_action_out_proj
    kernel: fused_norm.tl_fused_rms_matmul_bias_res
    why: >
      Its signature R + Bias + rms(A) @ B is already the folded form. The final
      norm's scale, its shift and the Euler dt all fold into per-step
      decoder_action_out_proj_w (10,1024,32) and _b (10,32), because W_out is
      small enough that ten copies cost 0.65 MB. Nothing to change; the wrapper
      indexes a different slice per step.

# ------------------------------------------------------------- 1. grid / CTA
grid:
  mode: wave                       # not persistent: every variant is under one wave
  ctas:   {A: 80, B: 128, C: 128, D: 42}
  shape:  {A: "(ceil(N/32), ceil(M/64))", B: "(ceil(N/32), ceil(M/64))",
           C: "(ceil(N/32), ceil(M/16))", D: "(ceil(M_flat/64), NUM_SPLIT)"}
  cta_tile: {A: {M: 64, N: 32}, B: {M: 64, N: 32}, C: {M: 16, N: 32},
             D: {M: 64, N: 256}}
  rasterization: >
    Row-major over (pid_n, pid_m) as TileLang emits it. No swizzle: the encoder
    uses T.use_swizzle at N=16384 to recover L2 reuse, but no decoder N here
    exceeds 4096 and every weight is a cold read once per step regardless.
  cluster:                         # deleted with a reason
    shape: [1, 1, 1]
    multicast: none
    why_deleted: >
      TileLang 0.1.11 does not expose cluster launch, and with 42-128 CTAs
      against 132 SMs there is no second CTA per tile to multicast to.
  launch:
    threads: 128                   # all four variants
    cta_per_sm: {A: 1, B: 1, C: 1, D: 1}
    smem_B: {A: 108544, B: 198656, C: 98304, D: 98304}
    max_regs_per_thread: 255       # not reconfigured; TileLang owns setmaxnreg

# ------------------------------------------------------------- 2. mainloop
mainloop:
  axis: {A: K, B: K, C: K, D: kv_chunk}
  step: {A: 128, B: 256, C: 256, D: 64}
  trip_count: {A: 8, B: 4, C: "8 (out_proj) | 16 (ffn_down)", D: 3}
  tail: >
    A/B/C: none needed on K -- 1024, 2048 and 4096 all divide their BLOCK_K.
    On M, BLOCK_M over-covers (64 vs 50, and 4x16 vs 50) and TileLang predicates
    the ragged rows; verified empirically in the prefix bring-up that its scalar
    stores are bounds-checked, and re-verified per variant in Phase 2.
    D: KEYS=1018 is not a multiple of CHUNK; the existing `(j_global < KEYS)`
    guard stays and the mask load sits behind it.
  operands_per_iter:
    - {name: A, tile: [BLOCK_M, BLOCK_K], dtype: bf16, bytes: "A 16384 | B 32768 | C 8192", src: gmem, via: cp.async}
    - {name: W, tile: [BLOCK_K, BLOCK_N], dtype: bf16, bytes: "A 8192 | B 16384 (x2) | C 16384", src: gmem, via: cp.async}
  loop_carried: >
    A/B/C: the fp32 accumulator in RF. D: acc_o plus the online-softmax running
    max m and running sum l, unchanged from Pi0.
  per_iter_math: >
    A/B: A_shared[i,j] *= F_local[i] * S_shared[ko*BLOCK_K + j], one T.Parallel
    over (BLOCK_M, BLOCK_K) between the tile copy and T.gemm. C: none.
    D: online softmax rescale, unchanged.

# ------------------------------------------------------------- 3. pipeline
pipeline:
  depth: {A: 4, B: 3, C: 4, D: 1}
  stage_index: "ko % NUM_STAGES"        # TileLang-owned, from T.Pipelined
  phase: "(ko // NUM_STAGES) & 1"       # TileLang-owned
  prologue: "NUM_STAGES-1 tiles prefetched; TileLang emits the peel"
  per_stage_bytes: {A: 24832, B: 66048, C: 24576, D: 32768}
  staged_buffers:
    - {name: A_shared, shape: [BLOCK_M, BLOCK_K], dtype: bf16, swizzle: TileLang-default}
    - {name: W_shared, shape: [BLOCK_K, BLOCK_N], dtype: bf16, swizzle: TileLang-default}
    - {name: "W2_shared (variant B only)", shape: [BLOCK_K, BLOCK_N], dtype: bf16, swizzle: TileLang-default}
    - {name: S_shared, shape: [BLOCK_K], dtype: bf16, swizzle: none}
  non_staged_buffers:
    - {name: C_shared, bytes: 8192, aliases: none,
       alias_safe_because: "variant A only; the RoPE rotation reads column pairs, which is not a legal fragment access, so the scaled accumulator stages through smem (base.py:613-615)"}
  barriers:
    - {name: full,  kind: mbarrier, count: NUM_STAGES, init_arrive_count: TileLang-owned,
       produced_by: "producer warps", waited_by: "math warps"}
    - {name: empty, kind: mbarrier, count: NUM_STAGES, init_arrive_count: TileLang-owned,
       produced_by: "math warps", waited_by: "producer warps"}
  barriers_note: >
    TileLang 0.1.11 owns barrier construction, arrival counts and the phase flip;
    T.Pipelined(num_stages=N) is the whole interface. The spec fixes depth and
    tile shapes, which is what determines them. Phase 2 records the emitted
    structure as a deviation if it differs from the above.

# ------------------------------------------- 4. warp specialization / roles
warp_groups:
  - id: "producer + math (TileLang-assigned)"
    warps: 4
    threads: 128
    regs: TileLang-owned
    role: >
      All four kernels compile with warp_spec=True, which is the `kernel`
      decorator default (base.py:62). TileLang derives the producer/consumer
      split from T.Pipelined; the source does not name warp roles and neither
      does this spec.
    issues: "cp.async / wgmma, assigned by the compiler"
    elected: TileLang-owned
inter_group_sync: >
  TileLang-owned. Two constraints inherited from Pi0 and load-bearing:
  tl_scaled_gate must keep warp specialization ON -- the no-WS pipeline planner
  rejects the dual GEMM's reuse of one A_shared across the W1 and W2 stages
  ("Stage 0 and 3 both write A_shared", base.py:459-462). And every expression
  inside a pipelined body must stay inlined: a named temporary lowers to a bind
  statement the role pass cannot classify and the compile aborts
  (fused_norm.py:15-18, base.py:822-824).

# ------------------------------------------------ 5. iters (instruction level)
math:
  - group: math
    stage_phase: "after the A-tile scale, once per stage (twice for B)"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 32, K: 16}
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: {A: 8, B: "16 per GEMM, 32 total", C: 16, D: "2 GEMMs, see D note"}
    a_source: smem-desc
    b_source: smem-desc
    acc: {location: RF, elems_per_thread: {A: 16, B: 32, C: 4, D: 176},
          dtype: f32, cleared: "T.clear before the mainloop; loop-carried after"}
    accumulate_across_iters: true
    after_batch: >
      A: scale by F, add bias, stage to C_shared, rotate Q/K pairs in fp32,
      scatter to Q/K/V. B: gelu_tanh on the gate branch, multiply by the up
      branch. C: multiply by g, add the residual read from gmem.
  - group: math
    stage_phase: "variant C only"
    unit: mma.sync                 # NOT wgmma -- see checks.mma_m
    inst_shape: {M: 16, N: 8, K: 16}
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: "BLOCK_K/16 = 16 k-steps x BLOCK_N/8 = 4 n-steps = 64 mma.sync"
    a_source: rf                   # via ptx_ldmatrix_x4 / _x2_trans
    b_source: rf
    acc: {location: RF, elems_per_thread: 4, dtype: f32, cleared: "T.clear before the mainloop"}
    accumulate_across_iters: true
    after_batch: "multiply by g, add the residual read from gmem"
  - group: math
    stage_phase: "variant D only"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 64, K: 16}
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: "S = q_sh @ k_shᵀ then acc_o += s_cast @ v_sh; unchanged from Pi0"
    a_source: smem-desc
    b_source: smem-desc
    acc: {location: RF, elems_per_thread: 176, dtype: f32, cleared: "s re-initialized from the mask each stage"}
    accumulate_across_iters: "acc_o yes; s no -- it is the mask, overwritten每 stage"
    after_batch: "online softmax rescale, unchanged"

# ------------------------------------------------------------- 6. epilogue
epilogue:
  position: after-mainloop
  math: >
    A: C_local *= F_local[i]; C_local += Bias_local[j]; stage to C_shared;
       rotate pairs (2p, 2p+1) in fp32 for the Q and K column ranges; scatter.
    B: C1 = gelu_tanh(C1 + b_gate[j]); C1 *= (C2 + b_up[j]).
    C: C_local = R[i,j] + C_local * g[j].
    D: normalize by l, write the partial and the log2-domain lse; unchanged.
  path: "rf -> gmem (A stages through smem first, for the pair-wise rotation)"
  output: {tile: "cta_tile", dtype: bf16}
  split_reduction: "D only: tl_fd_flat_combine, unchanged"

# ------------------------------------------------------------- 7. checks
checks:
  smem: >
    PASS. A 4x24832 + 8192 = 107520 B (105.0 KB).
    B 3x66048 = 198144 B (193.5 KB), 34.5 KB headroom -- the tightest, and
    NUM_STAGES=3 is already capped by keeping W1 and W2 co-resident.
    C 4x24576 = 98304 B (96.0 KB). D 3x32768 = 98304 B (96.0 KB).
    All under the 228 KB sm90 cap.
  threads: "PASS. 128 threads = one warp group of 4 warps, a legal wgmma issuer, on all four."
  acc_registers: >
    A 64x32/128 = 16 f32/thread. B 2x(64x32)/128 = 32. C 16x32/128 = 4.
    D acc_o 64x256/128 = 128, plus s 32 and s_cast ~16 -> ~176, the tightest
    item in the spec. D's register load is inherited from Pi0 unchanged and is
    not made worse by the mask change.
  mma_k: "PASS. A 8x16 = 128 = BLOCK_K. B 16x16 = 256. C 16x16 = 256. All K=16 atoms."
  mma_m: >
    PASS, once the right atom is compared against. Settled by dumping the
    generated CUDA rather than reasoned about: at BLOCK_M=16 TileLang does not
    emit wgmma at all. It falls back to Ampere warp-level MMA -- the source
    contains tl::mma_sync and tl::ptx_ldmatrix_x4 with no GmmaDescriptor and no
    warpgroup_* -- so variant C is mma.sync.m16n8k16 and 16 == cta_tile.M.
    At BLOCK_M=64 the same kernel emits the Hopper path (GmmaDescriptor,
    initialize_wgmma_descriptor, warpgroup_arrive/commit_batch/wait), so
    variants A and B are wgmma.m64n32k16 and 64 == cta_tile.M with one math
    group.
  mma_n_legal: >
    PASS. wgmma N=32 and N=64 are multiples of 8 and <= 256 (variants A, B, D).
    mma.sync N=8 is the only legal n for the m16n8k16 bf16 atom (variant C).
  trip_count: "PASS. A 8x128=1024=K. B 4x256=1024=K. C 8x256=2048 and 16x256=4096. D 3x64=192=CHUNK."
  output_coverage: >
    PASS with predication. A 80x32 = 2560 = N exactly; M 1x64 over-covers 50.
    B 128x32 = 4096 = N; same M over-cover. C 32x32 = 1024 = N, 4x16 = 64 over-covers 50.
    D 7x64 = 448 = QPAD over-covers M_flat=400; NUM_SPLIT x CHUNK = 1152 over-covers KEYS=1018,
    which the existing `< KEYS` guard handles and _num_splits keeps non-empty.
  occupancy: >
    PASS, and deliberately low. 1 CTA/SM on all four (smem-bound on B, tile-bound
    elsewhere). A 80, B 128, C 128, D 42 CTAs against 132 SMs: every variant is
    under one wave, which is the stated regime.
  barrier_arrivals: "TileLang-owned; recorded in Phase 2, not asserted here."
  arithmetic_intensity: >
    PASS in the sense that it confirms the regime rather than flagging a problem.
    Per-CTA tile intensity against the H100 bf16 ridge point of 295 FLOP/B:
    A 2*64*32*1024 / (131072+65536) = 21.3 FLOP/B.
    B 2*2*64*32*1024 / (131072+65536+65536) = 32.0.
    C (ffn_down) 2*16*32*4096 / (131072+262144) = 10.7.
    All an order of magnitude below the ridge, so all three are weight-bandwidth
    bound and no tiling change reaches peak. This is the 1.88 ms 10-step floor.

# ------------------------------------------------------------- 8. handover
verification:
  reference: >
    Two gates, in order. (1) eval/correctness/pi05/fused_vs_unfused.py, to be
    written: each variant against a plain-torch recomputation from the same
    folded tables, which is how the prefix bring-up localized its bug to the
    kernel rather than the weights. (2) suffix parity against
    PI0Pytorch(Pi0Config(pi05=True)) with --steps 1 and --layers bisection,
    reading layer 0 for structure.
  tolerance: >
    Layer 0 cosine > 0.9999 against the torch recomputation, matching the
    threshold the prefix gate settled on. Deep-layer drift is read for smoothness
    (no step > 0.005), not for absolute value.
  perf_target: >
    Record the per-stage split before and after. The v1 fallback adds one graph
    node per layer, 1260 -> 1440 over 18 layers x 10 steps; the cost is
    tl_rms_factor's own latency at M=50, K=1024 (100 KB touched), not node
    overhead, and it must be measured rather than assumed. The decoder floor is
    1.88 ms; the Pi0 measured decoder was 7.08 ms.
open_questions: []
deviations:
  - field: adarms_placement.s_residency
    was: "whole 1024-entry S in shared memory, loaded once before the mainloop"
    is: "staged per iteration, one BLOCK_K slice, alongside A and W"
    why: >
      Deadlocked. Under warp specialization a global->shared T.copy outside
      T.Pipelined lowers to a producer TMA with no matching consumer arrival;
      the kernel compiled, launched and hung. Found by a job that sat 17 minutes
      with no output after the kernel finished compiling.
  - field: "L2 nest, variants A and B"
    was: "A_s[s][i, j] *= F_local[i] * S_shared[k0 + j] for both"
    is: >
      Variant A applies only S in the mainloop and keeps F in the fp32 epilogue
      where tl_qkv_gemm_rope had it; variant B applies F and S together in the
      bf16 mainloop where tl_scaled_gate had it.
    why: >
      F is per-row and commutes with the reduction, so its placement is free.
      Following each origin kernel is the smaller diff and keeps each one's
      validated numerics for the F part.
  - field: warp_groups
    was: "all four warp_spec=True (the decorator default)"
    is: "variant C is warp_spec=False"
    why: >
      Its origin, tl_matmul_res, is built with warp_spec=False. At BLOCK_M=16
      the residual GEMM is far below one wave, where the producer warp sits idle
      and still costs warps and mbarrier traffic.
  - field: verification.reference
    was: "eval/correctness/pi05/fused_vs_unfused.py"
    is: "eval/correctness/pi05/kernel_parity.py"
    why: >
      The name was inherited from Pi0's file but the gate does not compare a
      fused path against an unfused one -- Pi0.5 v1 has one implementation per
      call site. It compares each kernel against a torch recomputation.
  - field: "variant D signature"
    was: "NUM_HEADS carried over from tl_fd_flat_split"
    is: "dropped"
    why: >
      It existed only for the state-token predicate, which is dead. Leaving it
      would keep a dead compile-time constant in the compile cache key.
---

# Pi0.5 AdaRMSNorm decoder kernels

Four variants, one design. They are specified together because they share a
single decision -- where the per-K `(1 + scale)` vector meets the A operand --
and splitting them into four specs would put that decision in front of a
reviewer four times.

## Loop nest

Variants A and B are the same kernel shape with different `BLOCK_K`, stage
depth, and epilogue, so one nest covers both; C and D are given as diffs
against their Pi0 originals.

### L1 — iteration space

```
  ada_gemm(X[M, K] bf16 row,            # decoder_x, un-normalized
           F[M]    bf16,                # rstd(X), from the separate tl_rms_factor
           S[K]    bf16,                # (1 + scale), per (step, layer)
           W[K, N] bf16 row,
           B[N]    bf16)                # shift @ W, per (step, layer)
        -> C[M, N] bf16 row

  for n0 in range(0, N, 32):                    # A 80 | B 128 tiles     parallel
    for m0 in range(0, M, 64):                  # 1 tile (M=50)          parallel
      for k0 in range(0, K, BK):                # A 8 | B 4 steps        SERIAL, contraction
        C[m0:m0+64, n0:n0+32] += ( X[m0:m0+64, k0:k0+BK] * S[k0:k0+BK] )
                                 @ W[k0:k0+BK, n0:n0+32]
                                   (64,BK) @ (BK,32) -> (64,32)
      C[m0:m0+64, n0:n0+32] = C[m0:m0+64, n0:n0+32] * F[m0:m0+64] + B[n0:n0+32]
```

The last line is the whole reason `F` can stay out of the mainloop and `S`
cannot: `F` is indexed by `m0`, the parallel axis, so it commutes with the `k0`
reduction and rides the epilogue. `S` is indexed by `k0`, the contraction axis,
so it sits inside the sum and has to meet `X` before the `@`.

### L2 — mapped to hardware

```
  grid wave, A 80 / B 128 CTAs: (n0, m0) from blockIdx, k0 stays serial per CTA
  128 threads = one warp group; TileLang assigns producer/consumer roles

  for (n0, m0) in blockIdx:                     # 1 tile per CTA
    S_shared[1024] <- S[0:1024]                 # 2048 B, once, before the mainloop
    F_local[64]    <- F[m0:m0+64]               # fragment
    B_local[32]    <- B[n0:n0+32]               # fragment
    C_acc[64, 32] = 0                           # f32 RF, A 16 / B 32 elems/thread, carried

    for k0 in range(0, 1024, BK):               # mainloop: start 0, stop 1024, step BK,
                                                #   trip A 8 (BK=128) | B 4 (BK=256)
      s, phase = (k0//BK) % D, (k0//BK)//D & 1  # stage; depth A 4 | B 3

      producer  wait empty[s] @ phase^1
                A_s[s][64, BK] <- X[m0:m0+64, k0:k0+BK]      A 16384 | B 32768 B  cp.async
                W_s[s][BK, 32] <- W[k0:k0+BK, n0:n0+32]      A  8192 | B 16384 B
                                                             # B stages W2_s as well
                full[s].arrive()

      math      wait full[s] @ phase
                A_s[s][i, j] *= F_local[i] * S_shared[k0 + j]     # (64, BK) bf16, in place
                                                                 # <-- the design decision
                C_acc += A_s[s][0:64, 0:BK] @ W_s[s][0:BK, 0:32]
                         (64,BK) @ (BK,32) -> (64,32)   f32 RF
                empty[s].arrive()

    epilogue  C_acc += B_local[j]               # bias, per N
              A: stage to C_shared, rotate (2p, 2p+1) pairs in f32, scatter Q/K/V
              B: C1 = gelu_tanh(C1) * C2        # after each branch has its own bias
              C[m0:m0+64, n0:n0+32] <- C_acc
```

Two things the nest is making explicit. `A_s[s]` is written *twice* per stage --
once by the producer's copy, once by the scale -- so the scale must land between
`full[s]` and the GEMM, not before the wait; and the `empty[s].arrive()` must
follow the GEMM, not the scale. And variant B runs two GEMMs against the same
scaled `A_s[s]`, which is exactly why its warp specialization cannot be turned
off.

### L3 — innermost body, expanded

```
      for ki in range(0, BK, 16):               # iter: start 0, stop BK, step 16,
                                                #   trip A 8 (BK=128) | B 16 (BK=256)
        wgmma.m64n32k16(
          A = A_s[s][0:64, ki:ki+16]     smem-desc, k-major   # already scaled
          B = W_s[s][ki:ki+16, 0:32]     smem-desc, k-major
          C = C_acc[64, 32]              f32 RF, 64*32/128 = 16 elems/thread
          clear = never — loop-carried, T.clear before the mainloop )
```

Variant B repeats this block against `W2_s[s]` into `C2_acc`, so 32 wgmma per
stage rather than 16.

### Variant C — diff against `tl_matmul_res`

```
  epilogue   C[m0:m0+16, n0:n0+32] = R[m0:m0+16, n0:n0+32] + C_acc * G_local[j]
                                                             ^^^^^^^^^^^^^^^^^
```

One per-N multiply, structurally identical to the `Bias_local[j]` add already in
`_matmul_bias` (base.py:186-187). No mainloop change, no new buffer beyond a
32-entry fragment. `R` and `C` alias, as they already do in Pi0.

### Variant D — diff against `tl_fd_flat_split`

```
  for ki in range(0, CHUNK_BLOCKS, 1):          # mainloop, trip 3, unchanged
    ...
    for i, j in T.Parallel(64, 64):
      jg = bz*CHUNK + ki*64 + j
-     s[i, j] = if_then_else((jg < KEYS) & ((bx*64 + i >= NUM_HEADS) | (jg <= ENC_LEN)),
-                            0.0, -inf)
+     s[i, j] = if_then_else(jg < KEYS, Mask[jg], -inf)
```

`Mask` is the `prefix_mask_bias` buffer the prefix pass already allocates and
fills (`buffers.py`, length `cache_len` = 1018): zero on valid prefix keys,
`-3e38` on prompt padding, zero across the suffix. The `< KEYS` guard stays
because it also bounds the load. The removed clause is the state-token rule, and
it is dead rather than replaced -- Pi0.5 has no state token, so all 400 flat
query rows share one mask row, which is precisely why a vector suffices.

## Schedule

Deleted with a reason: all four kernels are plain producer/consumer, TileLang
assigns the roles from `T.Pipelined`, and the L2 nest already shows the
hand-off. None of them has the interacting warp groups -- a seesaw, a ping-pong,
a hand-off through smem under a named barrier -- that would need an ordering
the loop nest cannot express.

## Why these numbers

**Every config is Pi0's, unchanged.** Not because they are known good at these
shapes -- M went 51 -> 50 and the decoder key length 819 -> 1018 -- but because
changing the tiling and the math in one step makes a numerical failure
un-bisectable. `tl_scaled_gate` is explicitly documented as tiling-dependent for
*correctness*, with BLOCK_M=32 producing garbage rather than failing
(base.py:463-467). Re-tuning is a separate step after the variants are correct,
and it must pass `correct=` on every candidate.

**`S` in shared memory rather than a fragment.** A fragment is distributed
across threads by the enclosing `T.Parallel` extent, so indexing one with the
global `k0 + j` inside a `(BLOCK_M, BLOCK_K)` loop is not obviously well-formed;
shared memory has no such question and costs 2 KB against a 34 KB headroom in
the worst case. It is also loaded once rather than staged, which keeps the
pipelined body free of a copy whose role the warp-specialization pass would
have to classify -- the failure mode `fused_norm.py:15-18` documents.

**Variant B is not built on the fused kernel.** Already decided upstream, but
the arithmetic is worth stating: recovering `tl_fused_rms_gate` needs a second
`BLOCK_M x BLOCK_K` shared buffer for the scaled copy (32 KB single-buffered at
`_FUSED_GATE`, which fits the 34 KB headroom with 2 KB to spare) or a reordering
to Pow -> scale -> gemm that serializes what is currently overlapped. Both are
v2, with the v1 measurement as the number to beat.

## Known risks

**Variant B's shared memory is the tightest at 194 KB of 228 KB.** The 34 KB
headroom is what a v2 fused variant would have to fit into, and it is 2 KB more
than the 32 KB that variant needs. Any BLOCK_K increase closes it entirely.

**Variant D's register pressure is the tightest overall at roughly 176 fp32 per
thread**, dominated by `acc_o` at 128. Inherited from Pi0 unchanged; the mask
change adds nothing, but it leaves no room for anything else in that kernel.

**Variant C is not on the Hopper tensor-core path at all.** Confirmed from the
generated CUDA, not inferred: at BLOCK_M=16 TileLang emits `tl::mma_sync` and
`ldmatrix` -- the Ampere warp-level path -- while the same kernel at BLOCK_M=64
emits wgmma descriptors and `warpgroup_*`. Nothing is wrong with that here,
since the residual GEMMs are bandwidth-bound at an arithmetic intensity of 10.7
against a ridge point of 295, and the config is Pi0-measured. But it makes
BLOCK_M=64 a cliff rather than a knob when these get re-tuned: crossing it
changes the instruction path, the register allocation and the smem requirement
all at once, so a sweep that straddles it is comparing two different kernels.

**M=50 divides none of the block sizes** (64, and 4x16 covers 50 with 14 wasted
rows). The prefix bring-up established empirically that TileLang bounds-checks
the scalar stores in `tl_rope_scatter_bf16` at a ragged M, but that was one
kernel; each variant here needs the same check rather than the inherited
assumption.
