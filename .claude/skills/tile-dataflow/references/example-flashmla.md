# Worked example — FlashMLA SM90 dense decode (split-KV)

Reverse-engineered from `csrc/sm90/decode/dense/splitkv_mla.cuh`, `traits.h`,
`config.h`, and the authors' write-up `docs/20250422-new-kernel-deep-dive.md`.
Line references are `splitkv_mla.cuh:NNN`.

Read this one for: **attention / decode**, **cooperating math warp groups**
(the "seesaw" schedule — *not* producer/consumer), **two-level pipelining**,
**online-softmax state as loop-carried data**, **smem aliasing**, and
**split-KV with a separate combine kernel**.

Contrast with `example-deepgemm.md`. Both are SM90 warp-specialized kernels and
their `warp_groups` sections have almost nothing in common — which is the point
of having that section at all.

---

```yaml
spec_version: 1
kernel: sm90_mla_decode_splitkv
status: approved
source: FlashMLA csrc/sm90/decode/dense/splitkv_mla.cuh

# ---------------------------------------------------------------- 0. problem
arch: sm90a                     # guarded on __CUDA_ARCH__ == 900 (splitkv_mla.cuh:971)
problem:
  op: "O = softmax(Q K^T * scale) V, MLA: K and V are the SAME tensor with d_k=576 and V = K[:, :512]"
  dims: {q_seq_per_hk: dynamic, kv_seqlen: dynamic, d_k: 576, d_v: 512, page: 64}
  dynamic: [q_seq_per_hk, kv_seqlen, batch]
  dtypes: {q: bf16, kv: bf16, p: bf16, acc: f32, o: bf16, lse: f32}
  layouts: {q: "row (M, 576)", kv: "paged, page=64, row (64, 576)", o: "row (M, 512)"}
  regime: >
    Decode. q_seq_per_hk = seqlen_q * (h_q / h_k); for DeepSeek-V3 decode with h_q=128,
    h_k=1, seqlen_q=1 that is 128, so exactly 2 CTAs of BLOCK_SIZE_M=64 cover one
    request's query. Long kv_seqlen, small M -- the regime that makes split-KV and a
    combine kernel necessary.
  note: "d_k != d_v is the MLA-specific fact that breaks specs written for MHA."

# ------------------------------------------------------------- 1. grid / CTA
grid:
  mode: "wave over (m, kv_head), persistent over requests within an SM part"
  ctas: "num_m_block * h_k * num_sm_parts"
  shape: "(m_block_idx, k_head_idx, partition_idx)  # splitkv_mla.cuh:972-974, 1344"
  cta_tile: {M: 64, N: 512}     # BLOCK_SIZE_M x HEAD_DIM_V -- the output tile
  rasterization: >
    A host-side scheduler (get_decoding_sched_meta.cu) splits the total KV blocks across
    num_sm_parts partitions balanced by payload, and each CTA walks
    [begin_req_idx, end_req_idx] with begin/end block indices. Load balance across
    ragged sequence lengths is the whole job of this scheduler.
  cluster:
    shape: [1, 1, 1]            # no cluster; there is no operand shared between CTAs to multicast
    multicast: none
  launch:
    threads: 256                # T::NUM_THREADS, __launch_bounds__(256, 1, 1) at splitkv_mla.cuh:960
    cta_per_sm: 1
    smem_B: 230808              # derived, see checks.smem
    max_regs_per_thread: 255
  launch_extra: "cudaLaunchKernelEx with cudaLaunchAttributeProgrammaticStreamSerialization -- PDL, so this kernel overlaps with the preceding one (splitkv_mla.cuh:1340-1352)"

# ------------------------------------------------------------- 2. mainloop
mainloop:
  axis: kv_seqlen
  step: 128                     # TWO page blocks of 64 per round -- the seesaw needs a pair
  trip_count: "(end_block_idx - start_block_idx) / 2 rounds, per request in this CTA's partition"
  tail: >
    Causal masking shrinks end_block_idx by the common mask length, and the residual
    per-row mask is applied only to the last two KV blocks via rRightBorderForQSeq
    (splitkv_mla.cuh:1050-1060). An odd block count is handled by the IS_BLK0_LAST /
    IS_BLK1_LAST template specializations of the subroutine.
  operands_per_iter:
    - {name: K0, tile: [64, 576], dtype: bf16, bytes: 73728, src: gmem, via: "TMA-2D x9, one per 64x64 sub-tile, EVICT_FIRST cache hint"}
    - {name: K1, tile: [64, 576], dtype: bf16, bytes: 73728, src: gmem, via: "TMA-2D x9, same"}
  loop_carried: >
    rO (128 f32 registers per thread, one half of the output per warp group),
    rL[2] (running softmax denominator, per thread, in registers), and
    sM (running softmax maximum, in SHARED MEMORY because both warp groups update it).
    Where each piece of the online-softmax state lives is a design decision, not an
    implementation detail: m is shared, l is not.
  per_iter_math: >
    Online softmax: row max, exp, and the rescale of rO by scale0/scale1. This is the
    CUDA-core work the seesaw exists to overlap with the other warp group's wgmma.

# ------------------------------------------------------------- 3. pipeline
pipeline:
  depth: 2                      # sK0 / sK1 -- two KV page blocks in flight
  stage_index: "explicit sK0 / sK1, not a modulo -- the two stages have different roles in the seesaw"
  phase: "cur_phase_K0, cur_phase_K1, cur_phase_Q, each flipped by hand after a full round of waits"
  prologue: >
    Q is TMA'd once per request before the loop (launch_q_copy, splitkv_mla.cuh:1024);
    K0 and K1 for the first two blocks are issued before the mainloop
    (splitkv_mla.cuh:1082-1085). Note the deliberate order for K1: tiles 4-8 are issued
    BEFORE tiles 0-3, so the tiles needed first arrive last -- the later tiles get a head
    start because they will be consumed later.
  sub_pipeline: >
    SECOND LEVEL, and the reason this kernel tolerates memory latency. Each 64x576 K
    block is nine independent 64x64 TMA copies with nine independent mbarriers
    (barriers_K0[9], barriers_K1[9]). The QK^T GEMM waits on tile i and computes on
    tile i while tile i+1 is still in flight (splitkv_mla.cuh:180-187 comment).
    Instruction-level pipelining inside a single mainloop stage.
  per_stage_bytes: 73728
  staged_buffers:
    - {name: sK0, shape: [64, 576], dtype: bf16, bytes: 73728, swizzle: "GMMA Layout_K_SW128_Atom"}
    - {name: sK1, shape: [64, 576], dtype: bf16, bytes: 73728, swizzle: "GMMA Layout_K_SW128_Atom"}
  non_staged_buffers:
    - {name: sQ,  bytes: 73728, aliases: "sP1 occupies sQ's 8th 64-wide tile",
       alias_safe_because: "Q tile 8 is hoisted into registers as rQ8 before the mainloop (retrieve_rP_from_sP, splitkv_mla.cuh:1103), so the smem copy is dead from that point on. Without this alias there is no room for sP1."}
    - {name: sP0, bytes: 8192, aliases: none, alias_safe_because: "n/a"}
    - {name: sO,  bytes: 0, aliases: "sK0/sK1", alias_safe_because: "the mainloop is finished before the epilogue stages O (splitkv_mla.cuh:991)"}
    - {name: "sM, sL_reduction_wksp, sScale0, sScale1", bytes: 1280, aliases: none, alias_safe_because: "n/a"}
    - {name: barriers, bytes: 152, aliases: none, alias_safe_because: "9+9+1 mbarriers x 8 B"}
  barriers:
    - {name: barriers_K0, kind: mbarrier-tx, count: 9, init_arrive_count: 1,
       produced_by: "elected thread, arrive_and_expect_tx(64*64*2 = 8192) per sub-tile",
       waited_by: "the warp group doing QK^T on K0, on cur_phase_K0"}
    - {name: barriers_K1, kind: mbarrier-tx, count: 9, init_arrive_count: 1,
       produced_by: same, waited_by: "the warp group doing QK^T on K1"}
    - {name: barrier_Q,  kind: mbarrier-tx, count: 1, init_arrive_count: 1,
       produced_by: "elected thread", waited_by: "both warp groups"}

# ------------------------------------------- 4. warp specialization / roles
# PATTERN: cooperating math groups, NOT producer/consumer. Both groups issue wgmma.
# They split the OUTPUT (O_L / O_R), not the ROLE, and exchange partial results
# through smem under named barriers.
warp_groups:
  - id: math0
    warps: 4
    threads: 128
    regs: "not reconfigured -- no setmaxnreg in the dense kernel; the compiler allocates up to 255 under __launch_bounds__(256,1)"
    role: "owns O_L = O[:, 0:256] and the P0 = Q K0^T softmax"
    issues: "wgmma (QK^T and PV), TMA for the next K0, st.shared to sP0, exp/max on CUDA cores"
    elected: "true for the TMA issue only"
  - id: math1
    warps: 4
    threads: 128
    regs: same
    role: "owns O_R = O[:, 256:512] and the P1 = Q K1^T softmax"
    issues: "wgmma (QK^T and PV), TMA for the next K1, st.shared to sP1, exp/max on CUDA cores"
    elected: "true for the TMA issue only"
inter_group_sync: >
  Five NamedBarriers over all 256 threads (traits.h, NamedBarriers enum), each ordering
  one hand-off in the seesaw:
    sMInitialized      - sM is visible before either group reads the running max
    sScale0Ready       - wg0's scale0 is visible to wg1 (wg1 needs it for its rescale)
    sScale1Ready       - wg1's scale1 is visible to wg0
    sP0Ready           - wg0's rescaled P0 has landed in sP0 for wg1's O_R += P0 V0R
    rO1sP0sV0RIssued   - wg1 has consumed sP0 and issued its gemm, so wg0 may proceed
  This section IS the algorithm. Deleting it does not lose a detail, it loses the kernel.

# ------------------------------------------------ 5. iters (instruction level)
math:
  - group: "math0 (on K0) and math1 (on K1)"
    stage_phase: "QK^T"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 64, K: 16}   # GMMA::ss_op_selector<bf16,bf16,f32, Shape<64,64,576>, K, K> (traits.h)
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 36                 # 576 / 16, issued as 9 sub-tiles x 4 wgmma each
    a_source: "smem-desc for tiles 0-7 (sQ); rf for tile 8 (rQ8), because sQ's tile 8 is aliased away to sP1"
    b_source: smem-desc
    acc: {location: RF, elems_per_thread: 32, dtype: f32, cleared: "at the first wgmma of the block, ScaleOut::Zero"}
    accumulate_across_iters: false      # rP is per-KV-block, consumed by softmax immediately
    after_batch: "row max -> m_new -> scale -> exp; publish scale through sScale*, publish P through sP*"
    split: >
      wg0 issues its 9 tiles in two chunks (PHASE-0 = tiles 0-3, PHASE-2 = tiles 4-8) with
      other work interleaved between them; wg1 issues all 9 in one PHASE-1 chunk. The
      asymmetry is the seesaw -- wg0's gap is where wg1's CUDA-core softmax runs.
  - group: "math0 and math1"
    stage_phase: "PV, local P (own P, in registers)"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 256, K: 16}  # GMMA::rs_op_selector<..., Shape<64,256,64>, K, MN>
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 4                  # 64 / 16
    a_source: rf                        # P is already in this group's registers
    b_source: smem-desc                 # V is sK viewed transposed (SmemLayoutV)
    acc: {location: RF, elems_per_thread: 128, dtype: f32, cleared: "no -- rO is loop-carried"}
    accumulate_across_iters: true
    after_batch: "none; the rescale happened before the gemm"
  - group: "math0 and math1"
    stage_phase: "PV, remote P (the OTHER group's P, from smem)"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 256, K: 16}  # ss variant -- A comes from sP0 / sP1
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 4
    a_source: smem-desc
    b_source: smem-desc
    acc: {location: RF, elems_per_thread: 128, dtype: f32, cleared: no}
    accumulate_across_iters: true
    after_batch: "none"
  # Per round (2 KV blocks) each warp group issues 36 + 4 + 4 = 44 wgmma.

# ------------------------------------------------------------- 6. epilogue
epilogue:
  position: after-mainloop
  math: "combine rL across the two warp groups through sL_reduction_wksp, divide rO by it, compute softmax_lse, cast f32 -> bf16"
  path: "rf -> smem (sO, aliasing sK0/sK1) -> TMA store -> gmem"
  output: {tile: [64, 512], dtype: bf16}
  split_reduction: >
    Two paths chosen per request by is_no_split. Unsplit requests write o_ptr and
    softmax_lse_ptr directly. Split requests write oaccum_ptr / softmax_lseaccum_ptr
    (f32, row stride 520 rather than 512 to dodge bank conflicts) and a SEPARATE combine
    kernel (csrc/smxx/decode/combine/combine.cu) merges the partials by LSE. That kernel
    needs its own spec.

# ------------------------------------------------------------- 7. checks
checks:
  smem: >
    sQ 73728 + sK0 73728 + sK1 73728 + sP0 8192 + sM 256 + sL_wksp 512 + sScale0 256 +
    sScale1 256 + barriers 152 = 230808 B vs 232448 B cap -> PASS, 1640 B spare (0.7%).
    This is as tight as an SM90 kernel gets, and it is why sP1 aliases sQ and sO aliases
    sK: without both aliases the kernel does not fit at all.
  threads: "128 + 128 = 256 == __launch_bounds__ first arg -> PASS. Both groups are whole warp groups, required for wgmma."
  acc_registers: >
    rO is 64*256/128 = 128 f32 registers per thread, per warp group. Across 256 threads
    that is 32768 registers = exactly HALF the SM's 65536 -> PASS, but it is the binding
    constraint of the entire design. A full 64x512 output would need the whole register
    file, which is why FA3's ping-pong (two output matrices alternating) is impossible
    here and the seesaw (one output, split between groups) was invented instead. Add
    rP (32) plus rQ8, rL, and addressing and the group is near 255 with no setmaxnreg
    headroom to give.
  mma_k: "QK^T: 36 * 16 = 576 == d_k -> PASS.  PV: 4 * 16 = 64 == page size -> PASS."
  mma_m: "64 == BLOCK_SIZE_M -> PASS. Both groups compute the SAME 64 rows over different output columns, so the usual M-split rule does not apply -- the split is along N."
  mma_n_legal: "N=64 and N=256 are legal wgmma atoms (multiples of 8, <= 256) -> PASS"
  trip_count: "scheduler-driven; end_block_idx clamped by causal common_mask_len -> PASS"
  output_coverage: "O_L (cols 0-255) by wg0 + O_R (cols 256-511) by wg1 = 512 == d_v -> PASS"
  occupancy: "smem 230808 > 232448/2 -> 1 CTA/SM, matches __launch_bounds__(_,1,1) -> PASS"
  barrier_arrivals: "all mbarriers init(1), single elected TMA producer -> PASS. NamedBarriers use all 256 threads except sMInitialized which uses 128 -> PASS. Phase bits tracked per barrier group by hand -> PASS."
  arithmetic_intensity: >
    Per CTA round: QK^T 2*64*128*576 + PV 2*64*512*128 = 17.8 MFLOP over 2*64*576*2 =
    147456 B = 121 FLOP/byte, against an H100 SXM5 bf16 dense ridge of ~989.5e12/3.35e12
    = 295 FLOP/byte -> FLAG at the CTA level. It closes at the DRAM level: with
    q_seq_per_hk=128 the two m_blocks of the same kv head read identical KV through L2,
    doubling effective intensity to ~242 FLOP/byte, which lands right at the ridge (the
    authors compute ~256 against a throttled ~258 and call it compute-bound). The
    kernel therefore sits ON the boundary by construction -- which is exactly why
    overlapping CUDA-core softmax with tensor-core GEMM is worth the whole seesaw.

# ------------------------------------------------------------- 8. handover
verification:
  reference: "torch scaled_dot_product_attention on the dequantized/unpaged tensors"
  tolerance: "bf16 accumulation tolerance; LSE compared separately"
  perf_target: "authors report up to 660 TFLOP/s and ~3 TB/s on H800 SXM5, ~80% of throttled tensor-core peak"
open_questions: []
deviations: []
```

## Loop nest

### L1 — iteration space

```
  mla_decode(Q[q_hk, dk] bf16 row-major,
             KV[kv, dk] bf16 paged page=64,      # K and V are ONE tensor: V = KV[:, 0:dv]
             blk_table[kv/64] i32, seqlens_k[batch] i32)
    -> O[q_hk, dv] bf16, lse[q_hk] f32

  dk = 576, dv = 512.  dk != dv — the MLA fact that breaks specs written for MHA.
  q_hk = seqlen_q * (h_q / h_k);  128 for DeepSeek-V3 decode (h_q=128, h_k=1, seqlen_q=1).

  for b in range(0, batch, 1):                       # per request                  parallel
    for h in range(0, h_k, 1):                       # per kv head                  parallel
      for m0 in range(0, q_hk, 64):                  # ceil(q_hk/64) tiles          parallel
        # online softmax over the kv axis; m_run and l_run are carried, not just O
        for n0 in range(0, seqlens_k[b], 64):        # ceil(seqlen_k/64) steps      SERIAL, contraction
          S    [64, 64]  =  Q[m0:m0+64, 0:dk] @ KVᵀ[n0:n0+64, 0:dk]
                            (64,576) @ (576,64) -> (64,64)
          P, m_run, l_run =  online_softmax(S, m_run, l_run)
          O[m0:m0+64, 0:dv] += P[64,64] @ KV[n0:n0+64, 0:dv]
                               (64,64) @ (64,512) -> (64,512)
        O[m0:m0+64, 0:dv] /= l_run

  contraction axes: dk (in QK^T), kv (in PV).   mainloop.axis = kv.
  the kv loop is additionally SPLIT across CTAs (grid.z), with a combine kernel merging by lse.
```

### L2 — mapped to hardware

```
  grid = (ceil(q_hk/64), h_k, num_sm_parts).  The b loop is NOT in the grid: a host-side
  scheduler slices the total kv blocks into num_sm_parts balanced partitions, and each CTA
  walks the requests and block ranges its partition covers (splitkv_mla.cuh:972-974, 1344).

  256 threads = 2 cooperating math WGs. No producer group — both issue wgmma.
  wg0 owns O_L = O[m0:m0+64, 0:256];  wg1 owns O_R = O[m0:m0+64, 256:512].   dv_h = 256

  meta = tile_scheduler_metadata[partition_idx]

  for b in range(meta.begin_req_idx, meta.end_req_idx + 1, 1):     # requests in this partition
    blk_lo = meta.begin_block_idx if b == meta.begin_req_idx else 0
    blk_hi = min(meta.end_block_idx, ceil((seqlens_k[b] - common_mask_len) / 64))
                                                     # causal shrinks the stop bound, it does not
                                                     # mask block-by-block (splitkv_mla.cuh:1050-1060)

    Q_s[64, 576] <- Q[m0:m0+64, 0:576]               # 73728 B, once per request, TMA
    rQ8[64, 64]  <- Q_s[:, 512:576]                  # sub-tile 8 hoisted to RF so sP1 can alias it
    O_h[64, 256] = 0                                 # f32 RF, 128 elems/thread, carried
    m_run[64] = -inf   (SMEM — shared by both WGs)   l_run[2] = 0  (RF, per WG)

    for n0 in range(blk_lo*64, blk_hi*64, 128):      # mainloop: step 128 = TWO page blocks,
                                                     # trip ceil((blk_hi-blk_lo)/2).
                                                     # Two, because the seesaw needs a pair of
                                                     # independent P matrices to interleave.
      K0_s[64,576] <- KV[blk_table[n0//64      ], 0:576]    73728 B, 9 x TMA of [64,64]
      K1_s[64,576] <- KV[blk_table[n0//64 + 1  ], 0:576]    73728 B, 9 x TMA, tiles 4-8 issued first
      V0_s = K0_s[:, 0:512] read as V0_sᵀ[512, 64]          # layout composition, no copy
      V1_s = K1_s[:, 0:512] read as V1_sᵀ[512, 64]

      # --- each warp group computes ONE S, then multiplies BOTH P's into its own half of O ---
      wg0:  S0[64,64] = Q_s[64,576] @ K0_sᵀ[576,64]    (64,576)@(576,64) -> (64,64)  f32 RF, 32/thread
            P0, scale0 = online_softmax_step(S0, m_run)                     -> sP0 for wg1
            O_L[64,256] += P0[64,64]  @ V0_s[64, 0:256]     (64,64)@(64,256) -> (64,256)   A from RF
            O_L[64,256] += sP1[64,64] @ V1_s[64, 0:256]     (64,64)@(64,256) -> (64,256)   A from SMEM
      wg1:  S1[64,64] = Q_s[64,576] @ K1_sᵀ[576,64]
            P1, scale1 = online_softmax_step(S1, m_run)                     -> sP1 for wg0
            O_R[64,256] += P1[64,64]  @ V1_s[64, 256:512]                                  A from RF
            O_R[64,256] += sP0[64,64] @ V0_s[64, 256:512]                                  A from SMEM
      # rescaling of O_h by scale0/scale1 is interleaved between these — see ## Schedule.

    epilogue  l_run combined across the two WGs through sL_reduction_wksp
              O_h /= l_run;  cast f32 -> bf16
              sO (aliases K0_s/K1_s) <- O_h;  TMA store
              if is_no_split: O[m0:m0+64, 256*wg : 256*wg+256] else Oacc + lse -> combine kernel
```

### L3 — innermost bodies, expanded

```
  QK^T   for t in range(0, 9, 1):                    # dk in sub-tiles of 64: 576/64 = 9
           wait barriers_K[t] @ phase                # sub-tile t landed; t+1..8 still in flight,
                                                     # which is how TMA latency is hidden here
           for ki in range(64*t, 64*t + 64, 16):     # iter: step 16 = wgmma K, 4 per sub-tile,
                                                     # 36 total over the full dk
             wgmma.m64n64k16(
               A = Q_s[0:64, ki:ki+16]      smem-desc for t in 0..7;  RF (rQ8) for t == 8
               B = K_sᵀ[ki:ki+16, 0:64]     smem-desc, k-major
               C = S[64, 64]                f32 RF, 64*64/128 = 32 elems/thread
               clear = (t == 0 and ki == 0) )
         # wg0 splits this loop: t in 0..3, then other work, then t in 4..8.
         # wg1 runs t in 0..8 in one chunk. The gap IS the seesaw.

  PV     for ki in range(0, 64, 16):                 # iter: start 0, stop kvb=64, step 16, trip 4
           wgmma.m64n256k16(
             A = P[0:64, ki:ki+16]          RF (own P, rs_op_selector)
               | sP*[0:64, ki:ki+16]        smem-desc (other WG's P, ss_op_selector)
             B = V_s[ki:ki+16, 0:256]       smem-desc, Major::MN — V is dv-major in smem,
                                            i.e. the tensor stored is V_sᵀ[dv, kvb]
             C = O_h[64, 256]               f32 RF, 64*256/128 = 128 elems/thread
             clear = never — O_h is carried across the whole mainloop )
         # four of these per round per warp group: 2 P matrices x issued by both groups.
```

`C = 128 elems/thread` is the number the whole kernel is built around: 128 x 256
threads = 32768 registers = exactly half the SM. See `checks.acc_registers`.

The same PV product written the other way round is what a dv-major accumulator
would imply:

```
      O_hᵀ[dv_h, m] += V_sᵀ[dv_h, kvb] @ P_sᵀ[kvb, m]
```

This kernel uses the **first** form — `m` is the wgmma M and `dv_h` is its N —
which is why B carries `Major::MN` and V is viewed transposed in smem. The other
form would make `dv_h` the M dimension and need a different atom and a different
smem layout. Same math, different kernel.

## Schedule

The warp groups cooperate rather than split producer from consumer, so the L2
nest cannot show the ordering — this is the algorithm. Per round, over blocks
`blk0 = n0//64` and `blk1 = blk0+1`, with `[w]` naming the warp group:

```
  [0] S0 = Q K0ᵀ                                      36 wgmma, tiles 0-3 then 4-8
  [1] S1 = Q K1ᵀ                                      36 wgmma, tiles 0-8 in one chunk
  [0] m_new0 = max(m_run, rowmax(S0));  scale0 = exp(m_new0 - m_run);  m_run = m_new0
  [0] P0 = exp(S0 - m_new0)                           CUDA cores, overlaps [1]'s wgmma
  [0]                                                 --> sScale0Ready
  [0] O_L = O_L*scale0 + P0 @ V0_L                    4 wgmma, A from RF
  [1] m_new1 = max(m_run, rowmax(S1));  scale1 = exp(m_new1 - m_run);  m_run = m_new1
  [1] P1 = exp(S1 - m_new1)
  [1]                                                 <-- sScale0Ready   --> sScale1Ready
  [1] O_R = O_R*(scale0*scale1) + P1 @ V1_R           4 wgmma, A from RF
  [0]                                                 <-- sScale1Ready
  [0] P0 *= scale1;  st.shared P0 -> sP0              --> sP0Ready
  [1]                                                 <-- sP0Ready
  [1] O_R += sP0 @ V0_R                               4 wgmma, A from SMEM
  [1] st.shared P1 -> sP1                             --> rO1sP0sV0RIssued
  [0]                                                 <-- rO1sP0sV0RIssued
  [0] O_L = O_L*scale1 + sP1 @ V1_L                   4 wgmma, A from SMEM

  TMA for the next K0 / K1 is issued the moment each buffer goes dead, inside these gaps.
```

Five NamedBarriers, all over 256 threads except `sMInitialized` (128). Reordering
any hand-off without re-deriving the whole sequence will hang.

## Why these numbers

**The seesaw exists because of one register-count fact.** A 64×512 fp32
accumulator is 32768 registers — half the SM. Two of them (FA3's ping-pong)
would need the entire register file, leaving nothing for operands. So instead of
two output matrices alternating between CUDA-core and tensor-core phases, there
is *one* output split column-wise across two warp groups, and the groups
alternate. Every named barrier in `inter_group_sync` is a consequence of that
choice. Change `cta_tile.N` or `acc.location` and the whole schedule dissolves.

**Two KV blocks per round, not one.** The seesaw needs two independent P
matrices to interleave. With one block there is nothing for the other group to
do while the first does its softmax.

**Nine TMA copies per K block, not one.** A single 73728-byte TMA has 73728
bytes of latency to hide before the first wgmma can start. Nine 8192-byte copies
with nine barriers let the GEMM start after the first eighth arrives. This is
the `sub_pipeline` field, and it is the reason a two-stage pipeline suffices
where a GEMM would want four.

**sP1 aliases sQ's 8th tile.** There is no room for another 8 KB buffer — 1640
bytes are left. Q's 8th tile is hoisted into `rQ8` before the mainloop, and the
QKᵀ for that tile switches from the `ss` wgmma variant to `rs`. A spec that
omitted the alias would be 8 KB over budget with no visible cause.

**No `setmaxnreg`.** Unlike DeepGEMM there is no lopsided producer to starve —
both groups need maximum registers. The dense kernel leaves allocation to the
compiler under `__launch_bounds__(256, 1)`. (The sibling sparse-FP8 kernel does
use 192/160/152 splits, because it *does* have a dedicated non-math group.)

## Known risks

- **smem has 1640 bytes of headroom.** Any new buffer requires finding another
  alias first.
- **Registers are the binding constraint, not smem.** `rO` alone is 128 per
  thread. The source contains an `if (start_block_idx - 16777216 < end_block_idx)`
  — a no-op guard whose only purpose is to stop NVCC from spilling
  (`splitkv_mla.cuh:1121`). Treat any change touching live ranges as a
  spill risk and check the compiler's register count, not just correctness.
- **The named-barrier order is deadlock-sensitive.** Five barriers with an
  asymmetric arrival pattern; reordering any hand-off in the seesaw without
  re-deriving the whole sequence will hang.
- **The kernel sits on the compute/memory boundary.** At smaller `h_q` it falls
  to memory-bound and the seesaw's overlap stops paying. `regime` is load-bearing
  here.
- **Split-KV needs the combine kernel.** Benchmarking `splitkv_mla` alone
  measures a partial result.
