# Worked example — FlashMLA SM90 dense decode (split-KV)

Reverse-engineered from `csrc/sm90/decode/dense/splitkv_mla.cuh`, `traits.h` and
`config.h`. Line references are `splitkv_mla.cuh:NNN` unless named otherwise.

Read this one for: **attention / decode**, **cooperating math warp groups** (the
"seesaw" — *not* producer/consumer), the only worked instance of
**`online_softmax`**, an **L3 timeline with two math groups contending for one
tensor pipe**, **`cta_per_sm: 1` forced and costed**, **software pipelining one
round deep**, **smem aliasing**, and **split-KV with a separate combine kernel**.
Contrast with `example-deepgemm.md`, whose `warp_groups` and L3 have nothing in
common with it.

**Provenance — read this before trusting a line number.** Every
`splitkv_mla.cuh:NNN` / `traits.h:NNN` below was read out of a real checkout of
`deepseek-ai/FlashMLA` at commit `15f13e5030374295491c5ce31b02d7e63a7772c6`
("Extend decode-combine num_splits buckets to 256", #199, 2026-07-28) — not
recalled, and not inferred from this file's earlier drafts.

**To re-check any of them**, the checkout is kept in this repo at
`.cache/fmla-verify`, gitignored (`.gitignore:8`) so it is never committed but
persists on disk:

    git -C .cache/fmla-verify log -1 --format=%H     # 15f13e5030374295...
    wc -l .cache/fmla-verify/csrc/sm90/decode/dense/{splitkv_mla.cuh,traits.h}
                                                     # 1355, 107

Line numbers move between revisions, so if a citation ever stops matching, check
the commit before assuming the citation is wrong.

**Not read**, and marked as such wherever it matters: the `combine.cu` half of
the pair, and the authors' write-up `docs/20250422-new-kernel-deep-dive.md` —
the two claims below that trace to the authors rather than to the code say so.
Tags are the skill's standard `[D]` / `[I]` / `TODO — needs source`.

---

```yaml
spec_version: 1
kernel: sm90_mla_decode_splitkv
status: reference               # documents a kernel that already exists: no sign-off to record,
                                # no Phase 2 to unblock, open_questions may stay non-empty
approved_by:                    # n/a at status: reference
source: "FlashMLA csrc/sm90/decode/dense/splitkv_mla.cuh @ 15f13e5"

# ---------------------------------------------------------------- 0. problem
arch: sm90a                     # guarded on __CUDA_ARCH__ == 900 (splitkv_mla.cuh:971)
problem:
  op: "O = softmax(Q K^T * scale) V, MLA: K and V are the SAME tensor with d_k=576 and V = K[:, :512]"
  dims: {q_seq_per_hk: dynamic, kv_seqlen: dynamic, d_k: 576, d_v: 512, page: 64}   # config.h:5-9
  dynamic: [q_seq_per_hk, kv_seqlen, batch]
  dtypes: {q: bf16, kv: bf16, p: bf16, acc: f32, o: bf16, lse: f32}
  layouts: {q: "row (M, 576)", kv: "paged, page=64, row (64, 576)", o: "row (M, 512)"}
  regime: >
    Decode. q_seq_per_hk = seqlen_q * (h_q / h_k); for DeepSeek-V3 decode with h_q=128,
    h_k=1, seqlen_q=1 that is 128, so exactly 2 CTAs of BLOCK_SIZE_M=64 cover one
    request's query. Long kv_seqlen, small M -- the latency-bound regime, which forces
    split-KV and makes cta_per_sm a live question (checks.residency).
  note: "d_k != d_v is the MLA-specific fact that breaks specs written for MHA."

# ------------------------------------------------------------- 1. grid / CTA
grid:
  mode: "wave over (m, kv_head), persistent over requests within an SM part"
  ctas: "num_m_block * h_k * num_sm_parts"          # splitkv_mla.cuh:1339-1344
  shape: "(m_block_idx, k_head_idx, partition_idx)  # splitkv_mla.cuh:972-974"
  cta_tile: {M: 64, N: 512}     # BLOCK_SIZE_M x HEAD_DIM_V -- the output tile
  rasterization: >
    -> persistence.scheduler. Load balance across ragged sequence lengths is the whole
    job of it; locality is not considered at all (l2_schedule).
  l2_schedule: >
    Solved offline, but for the wrong axis to call it an L2 decision: the scheduler
    optimises LOAD BALANCE, not locality. The reuse that matters comes free from the
    tiling -- at q_seq_per_hk=128 the two m_blocks of one kv head read byte-identical KV
    concurrently, the factor of 2 checks.arithmetic_intensity depends on. Nothing
    protects it, and the K TMA's EVICT_FIRST hint (splitkv_mla.cuh:47) works against it
    (open_questions).
  persistence:                  # "wave x persistent"; both halves are real
    cta_per_sm: 1              # forced twice over -- checks.residency
    grid_realises_it: "num_sm_parts is chosen so num_m_block * h_k * num_sm_parts ~= num_sms: one wave by construction, not by luck. [D]"
    scheduler: >
      STATIC AND OFFLINE (get_decoding_sched_meta.cu). Each CTA reads one precomputed
      (begin_req_idx, end_req_idx, begin_block_idx, end_block_idx) from
      tile_scheduler_metadata and walks it with index arithmetic -- no work queue and no
      atomic anywhere in the mainloop. The host balances the partitions by payload. [D]
    phase_ordering: >
      None in-kernel; the combine kernel is a separate LAUNCH, released early by
      cudaTriggerProgrammaticLaunchCompletion (splitkv_mla.cuh:1226). A grid barrier would
      have nothing to retain -- a 64x512 f32 partial per split fits in neither RF nor smem.
  cooperative: false            # no CTA waits on another, so residency is never required.
                                # The sm90 cooperative/cluster exclusivity never binds
                                # here -- this kernel uses NEITHER.
  cluster:
    shape: [1, 1, 1]            # no cluster; no operand is shared between CTAs to multicast
    multicast: none
  launch:
    threads: 256                # T::NUM_THREADS (traits.h:23), __launch_bounds__(256,1,1) at splitkv_mla.cuh:960
    cta_per_sm: 1               # forced by smem AND registers together; costed in checks.residency
    smem_B: 230808              # sizeof(SharedMemoryPlan), traits.h:71-83; see checks.smem
    max_regs_per_thread: 255
  launch_extra: "cudaLaunchKernelEx with cudaLaunchAttributeProgrammaticStreamSerialization -- PDL, so this kernel overlaps with the preceding one (splitkv_mla.cuh:1338-1352)"

# ------------------------------------------------------------- 2. mainloop
mainloop:
  axis: kv_seqlen
  step: 128                     # TWO page blocks of 64 per round -- the seesaw needs a pair
  trip_count: >
    Per request in this CTA's partition, and DIFFERENT PER WARP GROUP: wg0 stops at
    `end-2` (splitkv_mla.cuh:1139), wg1 at `end-3` (:1170), because each round ends by
    computing the NEXT round's scores and wg1 looks one block further ahead. The
    difference is absorbed by the tail specializations below.
  tail: >
    Causal masking shrinks end_block_idx by the common mask length (splitkv_mla.cuh:1053-1055),
    and the residual per-row mask is applied only to the last two KV blocks via
    rRightBorderForQSeq (splitkv_mla.cuh:1057-1060, applied at :352-356 and :429-434).
    The odd/last block counts are handled by the IS_BLK0_LAST / IS_BLK1_LAST /
    IS_BLK2_LAST template specializations of the two subroutines (splitkv_mla.cuh:1143-1147,
    1174-1182).
  operands_per_iter:
    - {name: K0, tile: [64, 576], dtype: bf16, bytes: 73728, src: gmem, via: "TMA-2D x9, one per 64x64 sub-tile, EVICT_FIRST cache hint (splitkv_mla.cuh:36-52)"}
    - {name: K1, tile: [64, 576], dtype: bf16, bytes: 73728, src: gmem, via: "TMA-2D x9, same"}
  loop_carried: [rO, m, l]      # m and l are the primitives.md contract names; the nest carries
                                # them as sM and rL, bound at their declaration in L2 and in
                                # loop_carried_where. rO is the source identifier and is what the
                                # L2/L3/L4 nest writes -- there is no second name for it.
  loop_carried_where: >
    Where each piece lives is a design decision, not an implementation detail.
    rO  128 f32 registers/thread, one HALF of the output per warp group (rO0 / rO1).
    m   SHARED MEMORY (sM), not registers, because the two groups CHAIN their max updates
        within a round (wg0 writes at :369, wg1 overwrites at :443-447). The only softmax
        state that is not thread-private, and what sScale0Ready / sScale1Ready order.
    l   rL[2] f32 in registers, a PER-LANE PARTIAL of the row sum, not the row sum -- a
        lane owns 2 accumulator rows (rP0 is ((2,2,8),1,1) = 2 rows x 16 cols under the
        wgmma m64nN C layout, traits.h:66-69) and both reductions are DEFERRED to the
        epilogue (:1185-1205). Legal because alpha is uniform along a row.
  per_iter_math: "-> non_mma: softmax (x2 per step, one per warp group, chained through sM), p_publish (x2)."

# ------------------------------------------------------------- 3. pipeline
pipeline:
  depth: 2                      # sK0 / sK1 -- two KV page blocks in flight
  stage_index: "explicit sK0 / sK1, not a modulo -- the two stages have different roles in the seesaw"
  phase: "cur_phase_K0, cur_phase_K1, cur_phase_Q, each flipped by hand once per round (splitkv_mla.cuh:241, 252)"
  depth_is_not_double_buffering: >
    THE MOST MISREADABLE FIELD HERE, in both directions. depth 2 is the seesaw PAIR --
    the two KV blocks of the SAME round -- not one buffer computing while another fills.
    But the kernel IS software-pipelined, one round deep, by a different mechanism:
    each round computes the SCORES for the next. Both subroutines open on an rP that is
    already full and close by issuing QK^T for round r+1, then `warpgroup_wait<0>`
    (splitkv_mla.cuh:816-831 wg0, :937-944 wg1, the authors' reason at :940-943); the
    prologue primes rP0 and rP1 (:1121-1122, :1155). So round r+1's K is in flight and
    its QK^T issued while round r still runs -- the opposite of what `depth: 2` suggests.
  buffer_death_is_per_half: >
    A K block does not die all at once: tiles 0-3 are V*L (wg0's), 4-7 are V*R (wg1's),
    8 is rope. Each group refills exactly what it last read (splitkv_mla.cuh:795, 822,
    926, 932). Consequences are L3 ORDERING EDGE 1. [D]
  prologue: >
    Q is TMA'd once per request before the loop (launch_q_copy, splitkv_mla.cuh:1024,
    and re-issued for the NEXT request at :1223 while this one is in its epilogue);
    K0's 9 tiles and K1's are issued before the mainloop (:1082-1085). K1 is issued
    4-8 THEN 0-3 because that is the order wg1 consumes them in -- see math.split. [D]
  sub_pipeline: >
    SECOND LEVEL, and the reason this kernel tolerates memory latency. Each 64x576 K block
    is nine independent 64x64 TMA copies with nine mbarriers (traits.h:80-81), and the
    QK^T wait is issued inside the per-tile helper (splitkv_mla.cuh:162-165), not hoisted:
    it turns "wait for 73728 B" into "wait for 8192 B".
  per_stage_bytes: 73728        # ONE stage: one 64x576 bf16 K block. depth 2 -> 147456 B staged.
  staged_buffers:               # the buffers of ONE stage; depth instantiates them as sK0 / sK1
    - {name: "sK (instantiated twice: sK0, sK1)", shape: [64, 576], dtype: bf16, bytes: 73728,
       swizzle: "GMMA::Layout_K_SW128_Atom (traits.h:51-54); read a second time as V through SmemLayoutV, a composition of the same layout (traits.h:56-59)"}
  non_staged_buffers:
    - {name: sQ,  bytes: 73728, swizzle: "GMMA::Layout_K_SW128_Atom (traits.h:46-49)",
       aliases: "sP1 occupies sQ's 8th 64-wide tile (splitkv_mla.cuh:986)",
       alias_safe_because: "Q tile 8 is hoisted into registers as rQ8 before the mainloop (retrieve_rP_from_sP, splitkv_mla.cuh:1103), so the smem copy is dead from that point on. Without this alias there is no room for sP1."}
    - {name: sP0, bytes: 8192, swizzle: "GMMA::Layout_K_SW128_Atom over [64,64] (traits.h:61-64) -- it is read back as a wgmma A descriptor, so it cannot be linear",
       aliases: none, alias_safe_because: "n/a"}
    - {name: sO,  bytes: 0, swizzle: "TWO layouts, chosen by is_no_split: bf16 Layout_K_SW128_Atom for the TMA-store path (splitkv_mla.cuh:624-627), or f32 Shape<64,512>:Stride<520,1> -- linear, padded, NOT swizzled -- for the split path (:661-664)",
       aliases: "sK0/sK1",
       alias_safe_because: "the mainloop is finished before the epilogue stages O (splitkv_mla.cuh:991). Note it must alias BOTH K buffers: the padded f32 split buffer is 64*520*4 = 133120 B, which does not fit in sK0's 73728 B alone."}
    - {name: "sM, sL_reduction_wksp, sScale0, sScale1", bytes: 1280,
       swizzle: "none -- linear f32 vectors (64, 128, 64, 64 entries; traits.h:76-79). Written by one lane per row, read as a per-quad broadcast, so a swizzle would buy nothing",
       aliases: none, alias_safe_because: "n/a"}
    - {name: barriers, bytes: 152, swizzle: "n/a -- mbarrier state", aliases: none,
       alias_safe_because: "9+9+1 mbarriers x 8 B (traits.h:80-82)"}
  barriers:
    - {name: barriers_K0, kind: mbarrier-tx, count: 9, init_arrive_count: 1,
       produced_by: "elected lane issues the TMA; the CONSUMER group calls arrive_and_expect_tx(64*64*2 = 8192) immediately before its wait (splitkv_mla.cuh:162-165)",
       waited_by: "the warp group doing QK^T on K0 (wg0), per sub-tile, on cur_phase_K0"}
    - {name: barriers_K1, kind: mbarrier-tx, count: 9, init_arrive_count: 1,
       produced_by: same, waited_by: "the warp group doing QK^T on K1 (wg1)"}
    - {name: barrier_Q,  kind: mbarrier-tx, count: 1, init_arrive_count: 1,
       produced_by: "thread 0 (splitkv_mla.cuh:699)", waited_by: "both warp groups (:1099)"}

# ------------------------------------------- 4. warp specialization / roles
# PATTERN: cooperating math groups, NOT producer/consumer. Both groups issue wgmma AND
# TMA. They split the OUTPUT (rO0 / rO1), not the ROLE, and exchange partial results
# through smem under named barriers.
warp_groups:
  - id: math0
    warps: 4
    threads: 128
    regs: "not reconfigured -- no setmaxnreg in the dense kernel; the compiler allocates up to 255 under __launch_bounds__(256,1)"
    role: "owns rO0 = O[:, 0:256] and the rP0 = sQ sK0^T softmax"
    issues: "wgmma (QK^T and PV), TMA for sub-tiles 0-3 of the next K0 and K1, st.shared to sP0, exp/max on CUDA cores"
    elected: "true for the TMA issue only (idx_in_warpgroup == 0)"
  - id: math1
    warps: 4
    threads: 128
    regs: same
    role: "owns rO1 = O[:, 256:512] and the rP1 = sQ sK1^T softmax"
    issues: "wgmma (QK^T and PV), TMA for sub-tiles 4-8 of the next K1 and K0, st.shared to sP1, exp/max on CUDA cores"
    elected: "true for the TMA issue only"
inter_group_sync: >
  Five NamedBarriers (traits.h:101-107), four of them over all 256 threads, each ordering
  one hand-off in the seesaw:
    sMInitialized      - sM is visible before wg0 reads the running max. 128 threads,
                         wg0 ONLY (splitkv_mla.cuh:1125); wg1 is ordered transitively
                         because it cannot read sM before sScale0Ready.
    sScale0Ready       - wg0's scale0 is visible to wg1 (wg1 needs it for its rescale)
    sScale1Ready       - wg1's scale1 is visible to wg0
    sP0Ready           - wg0's rescaled P has landed in sP0 for wg1's rO1 += sP0 @ sV0R
    rO1sP0sV0RIssued   - wg1 has consumed sP0 and issued its gemm, so wg0 may proceed
  This section IS the algorithm. Deleting it does not lose a detail, it loses the kernel.
  What each one prevents: the table in ## Warp-group choreography.

# ------------------------------------------------ 5. iters (instruction level)
math:
  - group: "math0 (on K0) and math1 (on K1)"
    stage_phase: "QK^T"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 64, K: 16}   # GMMA::ss_op_selector<bf16,bf16,f32, Shape<64,64,576>, K, K> (traits.h:26-29)
    contracts: d_k              # 576, the HEAD DIM -- not mainloop.axis (kv_seqlen). 36 x 16 = 576
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 36                 # 9 sub-tiles x 4 cute::gemm each (splitkv_mla.cuh:170-174)
    a_source: "smem-desc for tiles 0-7 (sQ); rf for tile 8 (rQ8), because sQ's tile 8 is aliased away to sP1 (splitkv_mla.cuh:213-218)"
    b_source: smem-desc
    acc: {name: "rP0 (wg0) / rP1 (wg1) -- the SAME registers the softmax then overwrites in place with the exponentials", location: RF, elems_per_thread: 32, dtype: f32, cleared: "at the first wgmma of the block, ScaleOut::Zero (splitkv_mla.cuh:222, 230)"}
    accumulate_across_iters: false      # rP is per-KV-block, consumed by softmax immediately
    after_batch: "-> non_mma.softmax; publish scale through sScale*, P through sP* (non_mma.p_publish)"
    split: >
      RESOLVED FROM SOURCE (splitkv_mla.cuh:220-253). wg0 issues its 9 tiles in two
      chunks -- PHASE-0 = tiles 0,1,2,3 and PHASE-2 = tiles 4,5,6,7,8 -- and wg1 issues
      all 9 in one PHASE-1 chunk, in the order 4,5,6,7,8,0,1,2,3. Neither order is
      arbitrary: each chunk boundary is the boundary between the sub-tiles wg0 TMA'd
      (0-3) and the ones wg1 TMA'd (4-8), so **every group consumes each half in the
      order that half was issued**, and the gap between wg0's two chunks holds the
      `warpgroup_wait<4>` plus wg0's own K1 TMA issue (:820-824).
      The scores being computed are the NEXT round's -- see pipeline.software_pipelined_scores.
  - group: "math0 and math1"
    stage_phase: "PV, local P (own P, in registers)"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 256, K: 16}  # GMMA::rs_op_selector<..., Shape<64,256,64>, K, MN> (traits.h:36-39)
    contracts: page             # 64, ONE page block -- half of mainloop.step, which is the seesaw pair
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 4                  # 64 / 16 (splitkv_mla.cuh:297)
    a_source: rf                        # P is already in this group's registers
    b_source: smem-desc                 # V is sK viewed transposed (SmemLayoutV)
    acc: {name: "rO (rO0 / rO1 as the subroutines' parameter names)", location: RF, elems_per_thread: 128, dtype: f32, cleared: "no -- rO is loop-carried"}
    accumulate_across_iters: true
    after_batch: "none; the rO *= alpha rescale happened BEFORE the gemm -- the primitive's hazard"
  - group: "math0 and math1"
    stage_phase: "PV, remote P (the OTHER group's P, from smem)"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 256, K: 16}  # ss variant -- A comes from sP0 / sP1 (traits.h:41-44)
    contracts: page             # 64, as above
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 4
    a_source: smem-desc
    b_source: smem-desc
    acc: {name: rO, location: RF, elems_per_thread: 128, dtype: f32, cleared: no}
    accumulate_across_iters: true
    after_batch: "none"
  # Per round (2 KV blocks) each warp group issues 36 + 4 + 4 = 44 wgmma, so the ONE
  # shared tensor pipe retires 88 -- 72 QK + 16 PV. That total is what L3 budgets.

# ------------------------------- 5b. non-MMA work (the CUDA-core column of L3)
non_mma:
  - id: softmax
    primitive: online_softmax                       # references/primitives.md -- WHAT
    mechanism: "lane-local max and sum -- the wgmma C layout keeps a row inside one lane,
                so neither reduction needs a shfl; the CHAINING between groups goes
                through sM, and that is the cost, not the reduce"
    params:
      rows: 64                  # BLOCK_SIZE_M; 2 rows per lane under the wgmma m64nN C layout [D]
      block: 64                 # ONE page block per instance. TWO instances per mainloop
                                # step of 128, one per group, CHAINED through sM.
      span: >
        THREE different spans in one primitive. reduce_max: quad -- a score row lives in
        4 lanes x 16 cols, so 16 lane-local FMAX/row then 2 __shfl_xor_sync at widths 1
        and 2 (splitkv_mla.cuh:352-360). reduce_sum: NONE in the mainloop, l stays a
        per-lane partial reduced once in the epilogue (:1185-1205). The carry m: cta,
        via sM.
      first_iter: "uniform -- splitkv_mla.cuh:331-393 has no iteration-0 specialization; sM is pre-initialised to MAX_INIT_VAL_SM and the same path runs every round."
      masked_rows: >
        clamped. Masked scores get MAX_INIT_VAL = -1e33f and sM starts at
        MAX_INIT_VAL_SM = -1e30f, under the source's stated invariant
        MAX_INIT_VAL * scale_softmax_log2 < MAX_INIT_VAL_SM (splitkv_mla.cuh:14-18) -- both
        finite, so no NaN at any seqlen_q. Backstop: rL == 0 or NaN is forced to 1.0f
        before the divide (:1211-1218).
      p_cast: "bf16, before the PV wgmma (the rs variant needs a bf16 A fragment)"
    where: mainloop.per_iter
    kind: "reduction + elementwise"
    over: "row, block=64 scores"
    loop_carried: [m, l]
    dtype: >
      f32 scores and reductions, f32 ex2.approx, ONE rounding f32->bf16 on P, f32 rO
      accumulation, f32 divide. RESOLVED: wg0's `P0 *= scale1` multiplies the RETAINED
      f32 rP0 and rounds once (`rPb(i) = (InputT)(rP0(i)*scale_factor)`,
      splitkv_mla.cuh:538-543), so P0 carries one rounding, not two -- a
      numerics-for-registers trade, made deliberately: it holds 32 f32 registers of rP0
      live across the whole hand-off on a kernel whose registers already bind
      (checks.acc_registers).
    cost: >
      per group per round per thread: 32 FMAX + 4 shfl.bfly + 32 FFMA + 34 ex2.approx +
      34 FADD/FMUL for l + 32 cvt.pack, plus the alpha rescale of rO, which dominates at
      128 FMUL each. wg0 rescales TWICE (scale0 at :372-377, then scale1 at :561-568) =
      256 FMUL; wg1 rescales ONCE with the folded scale0*scale1 (:464-470) = 128 FMUL.
      ~500 warp-instructions for wg0, ~320 for wg1, +~100 issue cycles each for ex2 on
      the MUFU at 1/4 rate. Both groups share the same four schedulers, so per round that
      is ~1.0k scheduler cycles. [I on the rates, [D] on the op counts]
    touches: "S/rP and rO (RF, wgmma acc fragment), sM (smem 64 f32), sScale0/sScale1 (smem)"
    on_critical_path: >
      yes, twice. (a) the primitive's own alpha hazard. (b) an edge the primitive does
      NOT have: wg1's alpha depends on wg0's scale0 via sScale0Ready, so the two
      instances are serialised against each other -- L3 edge 5, and the reason only one
      of the two is covered.

  - id: p_publish
    where: mainloop.per_iter
    kind: "smem store, cross-warp-group hand-off"
    over: "P[64, 64] bf16, 8192 B"
    span: warpgroup
    primitive: none             # bespoke; no contract in references/primitives.md
    mechanism: "stmatrix.sync.aligned.m8n8.x4.shared.b16 -- Copy_Atom<SM90_U32x4_STSM_N> (splitkv_mla.cuh:489-492)"
    loop_carried: []
    dtype: "bf16; the cast itself is softmax.p_cast"
    cost: "64 B/thread as 4 x stmatrix.x4 (16 B/thread each). The acc-fragment gather is done by the tiled-copy retile, not by hand. [D]"
    touches: "sP0 (wg0) / sP1 (wg1, aliasing sQ tile 8); read back as a wgmma A descriptor"
    on_critical_path: "yes -- sP0Ready gates wg1's third wgmma batch, rO1sP0sV0RIssued gates wg0's."

  - id: lse_and_normalise
    where: epilogue
    kind: "reduction + elementwise + cast"
    over: "l across the quad then across both groups; then rO[64, 256] per group"
    span: "quad (2 x shfl.bfly, splitkv_mla.cuh:1186-1189) then cta (sL_reduction_wksp, :1191-1205)"
    primitive: none
    mechanism: "shfl.bfly then a one-level smem exchange"
    loop_carried: []
    dtype: "f32 reduce and divide, one rounding f32->bf16 on O (unsplit) or none (split, f32 partials); lse = log2(l) + m*scale_log2 in f32"
    cost: "128 FMUL/thread for the divide plus the reduction; once per request, not per round"
    touches: "rL, sL_reduction_wksp, sM, rO -> sO (aliases sK0/sK1)"
    on_critical_path: "no -- after the mainloop, nothing in flight"

  - id: split_combine
    where: "SEPARATE KERNEL -- csrc/smxx/decode/combine/combine.cu, NOT READ"
    kind: reduction
    over: "num_splits partials of (acc_o[64,512] f32, lse[64] f32)"
    span: cta
    primitive: split_reduce     # references/primitives.md, the split variant
    mechanism: "a second, tiny online pass over the splits, merging by LSE"
    loop_carried: []
    dtype: "f32 partials in, one rounding f32->bf16 on the final O"
    cost: "TODO -- needs the combine kernel's source; it needs its own spec"
    touches: "oaccum_ptr / softmax_lseaccum_ptr, both f32, gmem row stride HEAD_DIM_V = 512 (splitkv_mla.cuh:1245-1248)"
    on_critical_path: "n/a -- separate launch, but part of the shipped latency (checks.acceptance)"
    note: "separate entry per primitives.md, not part of online_softmax. Whether the host scheduler can emit an empty split: TODO -- needs source."

# ------------------------------------------------------------- 6. epilogue
epilogue:
  position: after-mainloop
  math: "-> non_mma.lse_and_normalise"
  path: >
    Two paths through sOutputBuf, both detailed per-thread in L4: UNSPLIT is
    rf -> bf16 -> swizzled smem -> TMA store (splitkv_mla.cuh:622-658), SPLIT is
    rf -> f32 -> padded smem -> SM90_BULK_COPY_S2G (:659-684).
  output: {tile: [64, 512], dtype: "bf16 unsplit / f32 partial when split"}
  split_reduction: >
    Separate combine kernel, chosen per request by is_no_split (splitkv_mla.cuh:1230-1253):
    unsplit writes gO / gSoftmaxLse, split writes f32 gOAccum / gSoftmaxLseAccum for
    combine.cu to merge by LSE (non_mma.split_combine). That kernel needs its own spec.

# ------------------------------------------------------------- 7. checks
l4_accesses: accesses-flashmla.yaml   # scripts/tv_check.py computes L4's table from it
checks:
  smem: >
    depth 2 x per_stage 73728 = 147456 staged, plus 83352 non-staged (sQ 73728 + sP0 8192
    + sM 256 + sL_wksp 512 + sScale0/1 512 + barriers 152) = 230808 B vs the 232448 B cap
    -> PASS, 1640 B spare (0.7%). As tight as an SM90 kernel gets, and why sP1 aliases sQ
    and sO aliases sK: without both aliases it does not fit at all. [traits.h:71-83]
  threads: "128 + 128 = 256 == __launch_bounds__ first arg -> PASS. Both groups are whole warp groups, required for wgmma."
  acc_registers: >
    rO is 64*256/128 = 128 f32 registers per thread, per warp group. Across 256 threads
    that is 32768 registers = exactly HALF the SM's 65536 -> PASS, but it is the binding
    constraint of the entire design (## Why these numbers). Add rP0 (32 f32, held live
    across the hand-off so its rescale rounds once) plus rQ8, rL and addressing, and the
    group is near 255 with no setmaxnreg headroom to give.
  register_budget: "256 threads x 255 regs x 1 CTA/SM = 65280 <= 65536 -> PASS, and it is why cta_per_sm cannot be 2 (checks.residency)."
  mma_k: >
    QK^T: 36 * 16 = 576 == d_k -> PASS. The standard form
    (count_per_stage * inst.K == mainloop.step) does not apply here: QK^T contracts over
    d_k, not over mainloop.axis.
    PV: 4 * 16 = 64 == page size -> PASS, and this one does contract along kv.
  mma_m: "64 == BLOCK_SIZE_M -> PASS. Both groups compute the SAME 64 rows over different output columns, so the usual M-split rule does not apply -- the split is along N."
  mma_n_legal: "N=64 and N=256 are legal wgmma atoms (multiples of 8, <= 256) -> PASS"
  trip_count: "scheduler-driven, and different per warp group by one block (mainloop.trip_count); end_block_idx clamped by causal common_mask_len -> PASS"
  output_coverage: "rO0 (cols 0-255) by wg0 + rO1 (cols 256-511) by wg1 = 512 == d_v -> PASS"
  occupancy: "smem 230808 > 232448/2 -> 1 CTA/SM, matches __launch_bounds__(_,1,1) -> PASS"
  barrier_arrivals: "all mbarriers init(1); the elected lane issues the TMA and the CONSUMER group calls arrive_and_expect_tx before its wait -> PASS. NamedBarriers use all 256 threads except sMInitialized, which uses 128 and is wg0-internal -> PASS. Phase bits tracked per barrier group by hand -> PASS."
  arithmetic_intensity: >
    Per CTA round: 72 QK x 131072 + 16 PV x 524288 = 17.8 MFLOP over 2*64*576*2 =
    147456 B = 121 FLOP/byte, against an H100 SXM5 bf16 ridge of 989.5e12/3.35e12 = 295
    -> FLAG at the CTA level. It closes at the DRAM level: at q_seq_per_hk=128 the two
    m_blocks of one kv head read identical KV through L2, doubling it to ~242, right at
    the ridge. The kernel sits ON the boundary by construction -- which is why
    overlapping CUDA-core softmax with tensor-core GEMM is worth the whole seesaw.
    (The authors' ~256 against a throttled ~258 is inherited from their write-up, NOT
    read in this pass.) The 17.8 MFLOP is the same number L3 budgets the tensor pipe
    with; if the two disagree, one has dropped a warp group.

  residency: >
    cta_per_sm = 1, FORCED, not chosen -- unreachable on EITHER budget alone.
      smem       two CTAs need <= 116224 B each, but the irreducible floor is sQ 73728 +
                 ONE K block 73728 = 147456 B, 27% over even with the seesaw deleted and
                 depth cut to 1. sQ cannot shrink: QK^T needs all 576 channels at once.
      registers  rO alone is 32768 = HALF the register file, so two CTAs would spend all
                 65536 on accumulators before one operand fragment.
    d_k = 576 is what forces it; on a d_k=128 MHA kernel neither would bind and the same
    design could have run 2 CTAs/SM.
    WHAT IT COSTS: five NamedBarriers per round each stop all 256 threads with no second
    CTA to fill them. The seesaw covers four (L3 t2-t5) and not the fifth (L3 bubble A) --
    the one thing a second CTA would have bought here.

  concurrency: >
    FLAG, not PASS -- filled in and it fails. Two bubbles, one structural (A, the
    round-boundary drain) and one unmeasured (B, wg1's first QK sub-tile wait). Tensor
    ~4.2k cycles/round, CUDA ~24% of it, copy engine four issue points. -> L3 BUBBLE CHECK
    for all of it.
  vectorisation: >
    PASS on every touch in L4's table: TMA is not per-thread; sQ/sK wgmma reads are
    GMMA-canonical; both sP and sO stores are stmatrix.x4, the widest legal store from a
    wgmma acc fragment; the split sO store is 0-way only because of the stride-520
    padding, which L4 derives.
  addressing: >
    PASS -- counted in L4. The one that could have failed does not: the dependent
    block_table_ptr load is hoisted to the top of each subroutine (splitkv_mla.cuh:769-771,
    881-882), a full round ahead of the TMA that consumes it, so it is off the issue path.
    Per wgmma iter: nothing.
  tile_order: "-> grid.l2_schedule. SOLVED offline, but for load balance; the L2 reuse is a consequence of the tiling, not of the schedule. PASS on solved-not-defaulted, FLAG on the EVICT_FIRST tension."
  persistence: "persistent over requests inside a partition; grid ~= num_sms x 1 by the scheduler's construction, so no shortfall. cooperative false -- no CTA waits on another. No semaphore. PASS."
  non_mma_accounting: >
    softmax and p_publish appear in L3's CUDA columns with their costs; lse_and_normalise
    and split_combine are outside the timeline. softmax.loop_carried [m, l] both appear in
    mainloop.loop_carried [rO, m, l]. Both critical-path entries say what the copy engine
    does during them -- it issues the next round's sub-tiles from t3 onward, because
    buffers die per half-block. PASS.
  rounding_contract: >
    PASS on the mainloop -- one rounding to bf16 on P and one on O, everything else f32
    (non_mma.softmax.dtype has where each lands). The parity consequence is the SPLIT
    path: f32 partials plus a second LSE merge, so split and unsplit are NOT bitwise
    identical for the same input, and a parity test must pin is_no_split.
  traceability: >
    L1 -> L2 -> YAML names: q_hk/m0 -> cta_tile.M=64; dk -> sQ, sK columns; dv -> the
    rO0/rO1 split at dv_h=256; kv/n0 -> mainloop.axis and step 128.
    L2/L3/L4 use the SOURCE's identifiers throughout, so every buffer name in the nest
    greps against splitkv_mla.cuh. Two bindings need stating because L1 is hardware-free
    and the YAML follows primitives.md: L1's output `O` is `rO` (instances rO0, rO1) from
    L2 down, and L1's carried `m`/`l` -- the names mainloop.loop_carried and
    non_mma.softmax.loop_carried use -- are `sM` and `rL[2]` in the nest, bound at their
    L2 declaration and in mainloop.loop_carried_where. Nothing else is renamed.
    The `@` rule has ONE named exception: the shared inner name of PV is `kv` ==
    mainloop.axis as required, but QK^T's is `dk`, which is not the mainloop axis at all.
    Same exception as checks.mma_k, and it is the MLA fact -- QK^T contracts over a
    576-channel axis that no loop in the mainloop walks. PASS with the exception named.
  loop_bounds: >
    L1: three parallel ranges plus `for n0 in range(0, seqlens_k[b], 64)`. L2: the
    mainloop is `range(blk_lo*64, blk_hi*64, 128)` -- step 128 == mainloop.step, and its
    trip count is the per-group `(end - start - 2 or 3)/2` of mainloop.trip_count. L4:
    QK^T is 9 sub-tiles x `range(64t, 64t+64, 16)` = 36 == math.count_per_stage; PV is
    `range(0, 64, 16)` = 4 == math.count_per_stage. PASS, with the caveat that the L4 QK
    loop's sub-tile order is per phase, not 0..8 -- wg1's is 4,5,6,7,8,0,1,2,3 (math.split).

  floor: >
    TODO -- needs measurement; no Phase 0 run exists for this kernel here, so this file
    carries NO floor number. Structural input to one: the kernel reads d_k x 2 B = 1152 B
    per KV token per kv head, and a floor needs that against measurement 2's `a + MB/b`
    plus measurement 1's launch cost counted TWICE (splitkv + combine).
  reference: >
    n/a -- not measured here. The authors' 660 TFLOP/s is this implementation measuring
    itself, not an independent kernel to calibrate a floor model against.
  acceptance: >
    The (splitkv_mla + combine) PAIR, at the shipped batch and seqlen distribution, in the
    graph it ships in. splitkv_mla alone measures a partial result and reads ~n_splits
    times too fast on requests that split.
  falsifiability: >
    "the seesaw hides the softmax"   -> replace the softmax with a constant scale. The
      seesaw covers wg1's instance (under wg0's first PV batch) but NOT wg0's, which is
      bubble A. So predicted drop is ~8-10% of round time, not the ~24% the CUDA column
      is worth. A drop near 24% means nothing was overlapping; a drop near zero means
      bubble A is not real and this timeline is wrong.
    "the sub_pipeline pays"          -> collapse to one 73728 B TMA and one barrier per
      block. If round time is unchanged, the latency was already hidden.
    "buffers die per half-block"     -> move wg0's tiles-0-3 TMA (splitkv_mla.cuh:795)
      down to the end of the subroutine. If round time is unchanged, the early issue was
      not buying cover and edge 1 is over-stated.
    "cta_per_sm 1 costs bubble A"    -> not falsifiable here, since 2 is unreachable.
      Falsifiable on a smaller-d_k variant that fits 2 CTAs/SM.
    "on the compute/memory boundary" -> dram__bytes_read above 1152 B per token per kv
      head means the two m_blocks miss each other in L2 and the intensity argument is
      wrong by 2x.

# ------------------------------------------------------------- 8. handover
verification:
  reference: "torch scaled_dot_product_attention on the dequantized/unpaged tensors"
  tolerance: "bf16 accumulation tolerance; LSE compared separately; is_no_split must be pinned (checks.rounding_contract)"
  perf_target: "authors report up to 660 TFLOP/s compute-bound and ~3000 GB/s memory-bound on H800 SXM5 with CUDA 12.8 (README.md:24, 35). The ~80%-of-throttled-peak reading is inherited, not recomputed here."
open_questions:
  - "grid.l2_schedule: the K TMA's EVICT_FIRST hint (splitkv_mla.cuh:47) works against the sibling m_block's L2 reuse that checks.arithmetic_intensity depends on. Either the two m_blocks are close enough in time that the hint never fires, or the intensity argument is optimistic."
  - "split_combine: can the host scheduler emit an empty split? combine.cu was not read; primitives.md flags it as the split variant's hazard."
  - "epilogue split path: the stride-520 padding is justified below for the SMEM staging buffer, where it is worth 4x. Whether the authors also had the combine kernel's gmem read pattern in mind is not recoverable without combine.cu."
  - "checks.floor and every cycle figure in L3: [I], derived from published H100 peaks and the op counts in non_mma. Nothing here was measured."
deviations: []
```

## Loop nest

### L1 — iteration space

```
  mla_decode(Q[q_hk, dk] bf16 row-major,
             KV[kv, dk] bf16 paged page=64,      # K and V are ONE tensor: V = KV[:, 0:dv]
             block_table[kv/64] i32, seqlens_k[batch] i32)
    -> O[q_hk, dv] bf16, lse[q_hk] f32

  dk = 576, dv = 512.  dk != dv — the MLA fact that breaks specs written for MHA.
  q_hk = seqlen_q * (h_q / h_k);  128 for DeepSeek-V3 decode (h_q=128, h_k=1, seqlen_q=1).

  for b in range(0, batch, 1):                       # per request                  parallel
    for h in range(0, h_k, 1):                       # per kv head                  parallel
      for m0 in range(0, q_hk, 64):                  # ceil(q_hk/64) tiles          parallel
        m[64] = -1e30;  l[64] = 0                    # online_softmax state, carried with O
        for n0 in range(0, seqlens_k[b], 64):        # ceil(seqlen_k/64) steps      SERIAL, contraction
          S    [64, 64]  =  Q[m0:m0+64, 0:dk] @ KVᵀ[n0:n0+64, 0:dk]
                            (64,576) @ (576,64) -> (64,64)
          m, l = carry(online_softmax(S, m, l))      # the only carried softmax state;
          P, alpha = the same call's per-iteration outputs, NOT carried
          O[m0:m0+64, 0:dv] *= alpha                 # THE HAZARD: strictly before the MMA
          O[m0:m0+64, 0:dv] += P[64,64] @ KV[n0:n0+64, 0:dv]     # O is carried: = rO
                               (64,64) @ (64,512) -> (64,512)
        O[m0:m0+64, 0:dv] /= l

  contraction axes: dk (in QK^T), kv (in PV).   mainloop.axis = kv.  (checks.traceability
  names dk as the one `@` whose inner axis is deliberately not the mainloop axis.)
  the kv loop is additionally SPLIT across CTAs (grid.z), with a combine kernel merging by lse.
```

### L2 — mapped to hardware

```
  grid = (ceil(q_hk/64), h_k, num_sm_parts).  The b loop is NOT in the grid: a host-side
  scheduler slices the total kv blocks into num_sm_parts balanced partitions, and each CTA
  walks the requests and block ranges its partition covers (splitkv_mla.cuh:972-974, 1344).
  num_sm_parts is sized so the grid is one wave at 1 CTA/SM (grid.persistence).

  256 threads = 2 cooperating math WGs. No producer group — both issue wgmma AND TMA.
  Every buffer below is the SOURCE's identifier. rO is one variable (:1088) that wg0 and wg1
  each own an instance of; the subroutines name them rO0 / rO1, and they hold
  O[m0:m0+64, 0:256] and O[m0:m0+64, 256:512] respectively.   dv_h = 256

  sched_meta = tile_scheduler_metadata[partition_idx]   # a table read, never an atomic

  for b in range(sched_meta.begin_req_idx, sched_meta.end_req_idx + 1, 1):   # this partition
    blk_lo = sched_meta.begin_block_idx if b == sched_meta.begin_req_idx else 0
    blk_hi = min(sched_meta.end_block_idx, ceil((seqlen_k - common_mask_len) / 64))
                                                     # causal shrinks the stop bound, it does not
                                                     # mask block-by-block (splitkv_mla.cuh:1053-1060)

    sQ[64, 576]  <- Q[m0:m0+64, 0:576]               # 73728 B, once per request, TMA
    rQ8[64, 64]  <- sQ[:, 512:576]                   # sub-tile 8 hoisted to RF so sP1 can alias it
    rO[64, 256] = 0                                  # f32 RF, 128 elems/thread, LOOP-CARRIED
    sM[64] = -1e30      # the carried `m`: SMEM, because both WGs chain their updates through it
    rL[2]  = 0          # the carried `l`: RF, per-LANE PARTIAL — reduced only in the epilogue

    rP0 = sQ @ sK0ᵀ ; rP1 = sQ @ sK1ᵀ                # PROLOGUE: round 0's scores, computed
                                                     # before the loop (splitkv_mla.cuh:1121, 1155)

    for n0 in range(blk_lo*64, blk_hi*64, 128):      # mainloop: step 128 = TWO page blocks,
                                                     # trip ~(blk_hi-blk_lo)/2, and one shorter
                                                     # for wg0 than wg1 (mainloop.trip_count).
                                                     # Two blocks, because the seesaw needs a pair
                                                     # of independent P matrices to interleave.
      # --- each warp group softmaxes the scores it computed LAST round, then multiplies
      #     BOTH P's into its own half of rO, then computes NEXT round's scores ---
      #     rP0/rP1 arrive holding SCORES and leave holding the f32 exponentials, IN PLACE;
      #     rPb / rP1b are the bf16 casts the wgmma consumes (splitkv_mla.cuh:331-393)
      wg0:  rPb, sScale0 = online_softmax(rP0, carry sM)                -> sP0 for wg1
            rO0[64,256] = rO0*scale0 + rPb[64,64] @ sV0L[64,0:256] (64,64)@(64,256) A from RF
            rO0[64,256] = rO0*scale1 + sP1[64,64] @ sV1L[64,0:256]                  A from SMEM
      wg1:  rP1b, sScale1 = online_softmax(rP1, carry sM)               -> sP1 for wg0
            rO1[64,256] = rO1*(scale0*scale1) + rP1b[64,64] @ sV1R[64,0:256]        A from RF
            rO1[64,256] +=                      sP0[64,64]  @ sV0R[64,0:256]        A from SMEM

      sK0[64,576] <- KV[block_table_ptr[n0//64 + 2], 0:576]  73728 B, 9 x TMA of [64,64]:
      sK1[64,576] <- KV[block_table_ptr[n0//64 + 3], 0:576]  tiles 0-3 by wg0, 4-8 by wg1,
                                                             each as soon as its own reader retires
      sV0L, sV0R = get_half_V(sK0, 0), get_half_V(sK0, 1)   # sKᵀ[512, 64] split at column 256;
      sV1L, sV1R = get_half_V(sK1, 0), get_half_V(sK1, 1)   # layout composition, no copy
      rP0 = sQ @ sK0ᵀ (wg0, two chunks) ;  rP1 = sQ @ sK1ᵀ (wg1, one chunk)
                                                             # NEXT round's scores — the pipelining
      # the two softmax instances are CHAINED, not independent: scale1 rescales both wg0's
      # already-multiplied rO and wg0's rP0. See ## Warp-group choreography.

    epilogue  rL reduced across the quad then across both WGs (sL_reduction_wksp);
              rO /= rL; lse = log2(rL) + sM*scale_log2; -> sOutputBuf -> gmem.  §6.
```

### L3 — schedule

Four columns, not three: the two math groups issue into **one shared tensor
pipe**, so it needs its own column labelled by owner, and each group needs its
own CUDA column. The round is **asymmetric in which round's scores its QK belong
to** — each group's 36 compute the *next* round's — but symmetric in count, so
the pipe's per-round budget is the `math` section's 72 QK + 16 PV either way.
What to check by eye: the tensor column is busy t1-t9 and **empty at t0**.
Cycle counts are `[I]`; row order is program order from the two subroutines
(`splitkv_mla.cuh:742-832`, `854-945`).

```
  one steady-state round, blocks blk0 (in sK0) and blk1 (in sK1).  [w] = warp group.
  entering the round, rP0 and rP1 already hold this round's scores. Buffer names are the
  source's; sV*L / sV*R are the get_half_V views of sK0 / sK1 (splitkv_mla.cuh:717-722).
  sc0 / sc1 abbreviate the per-row scale0 / scale1 that live in sScale0 / sScale1.

    copy engine (TMA)       wg0 CUDA cores          wg1 CUDA cores          tensor cores (SHARED)
    ----------------------- ----------------------- ----------------------- -----------------------
 t0 --                      [0] softmax(rP0):       [1] <-- sScale0Ready    -- EMPTY (bubble A)
                              32 FMAX + 4 shfl,                                both groups drained
                              m->sM, sc0->sScale0                              their QK at t9 of
                              rO0 *= sc0 (128                                  the previous round
                              FMUL <-- ALPHA),
                              rP0 = ex2 -> rPb
                              --> sScale0Ready
 t1 --                      [0] issue PV, then      [1] softmax(rP1):       [0] rO0 += rPb @ sV0L
                              wait<0> on it           m, sc1->sScale1,          4 wgmma m64n256k16
                                                      rO1 *= sc0*sc1            A from RF
                                                      (128 FMUL <-- ALPHA)
                                                      --> sScale1Ready
 t2 --                      [0] <-- sScale1Ready    [1] stmatrix -> sP1     [1] rO1 += rP1b @ sV1R
                                                                                4 wgmma, A from RF
 t3 sK0 <- blk r+2, t0-3    [0] rPb = bf16(rP0      --                      [1]  "
      4 x TMA-2D, 8192 B          * sScale1)
      elected lane of wg0     stmatrix -> sP0
      sV0L DEAD: wg0's t1     --> sP0Ready
      batch retired at t2
 t4 --                      [0] <-- rO1sP0          [1] <-- sP0Ready        [1] rO1 += sP0 @ sV0R
                                    sV0RIssued        --> rO1sP0sV0RIssued      4 wgmma, A from SMEM
                                                                                LAST read of sV0R
 t5 sK1 <- blk r+3, t4-8    [0] rO0 *= sScale1      [1] wait<1>             [0] rO0 += sP1 @ sV1L
      5 x TMA, wg1's lane     128 FMUL <-- ALPHA      (own PV retires)          4 wgmma, A from SMEM
      sV1R DEAD                                                                 LAST read of sV1L
 t6 sK0 <- blk r+2, t4-8    --                      [1] wait<0>             [0] rP0[r+2] tiles 0-3
      5 x TMA, wg1's lane                                                      16 wgmma, waits
      sV0R DEAD                                                                barriers_K0[0..3]
 t7 sK1 <- blk r+3, t0-3    [0] wait<4> (retires    --                      [0]  "
      4 x TMA, wg0's lane         the t5 batch)
      sV1L DEAD
 t8 --                      --                      --                      [1] rP1[r+3], tiles
                                                                               4,5,6,7,8,0,1,2,3
                                                                               36 wgmma; the FIRST
                                                                               waits barriers_K1[4]
                                                                               issued only at t5
                                                                               <-- bubble B risk
 t9 --                      [0] wait<0>             [1] wait<0>             [0] rP0[r+2] tiles 4-8
                                                                               20 wgmma
```

```
  ORDERING EDGES

  1. WHAT GATES THE NEXT COPY IS BUFFER DEATH, AND A K BLOCK DIES IN HALVES. Tiles 0-3 of
     a block are V*L and are read only by wg0; tiles 4-7 are V*R and only by wg1; tile 8
     is rope, read only by QK^T. So each group refills exactly what it last read, the
     moment its own `warpgroup_wait` retires that read: wg0 at t3 and t7, wg1 at t5 and
     t6 (splitkv_mla.cuh:795, 822, 926, 932). The copy engine gets work from the MIDDLE
     of the round, not only its tail.

  2. CONSEQUENCE: the cover for each sub-tile is the distance from its issue row to its
     consume row. barriers_K0[0] is issued t3 and consumed t6 -- ~8 PV wgmma plus three
     CUDA stretches of cover. barriers_K1[4] is issued t5 and consumed t8 -- covered only
     by wg0's 16 QK at t6-t7. That asymmetry is bubble B.

  3. EACH HALF IS CONSUMED IN THE ORDER IT WAS ISSUED. wg0 consumes 0-3 then 4-8; wg1
     consumes 4-8 then 0-3 (splitkv_mla.cuh:220-253). Both match their issue order, in
     the prologue (:1082-1085) and in the steady state. There is no "needed first,
     arrives last" anywhere.

  4. THE ALPHA HAZARD FIRES THREE TIMES PER ROUND (t0, t1, t5), not once: wg0 rescales
     TWICE because a wgmma is issued between scale0 and scale1, wg1 ONCE with the folded
     scale0*scale1. 256 vs 128 FMUL/thread/round.

  5. A CROSS-GROUP SOFTMAX EDGE THE PRIMITIVE DOES NOT HAVE. wg1's alpha depends on wg0's
     scale0 (sScale0Ready), and wg0's rP0 must be rescaled by wg1's scale1 (sScale1Ready)
     before publication. The two `online_softmax` instances are serialised against each
     other. This is the price of putting `m` in smem, and it is what makes t0 empty.

  6. THE ONLY MEMORY SERIALISATION is barriers_K*[t]. Everything else is a register or
     smem hazard between the two warp groups.

  BUBBLE CHECK  (per steady-state round; cycle figures [I], from published H100 peaks and
                 the op counts in non_mma, NOT measured — see checks.floor)

  the rate, stated once   H100 SXM5 bf16 dense: 989.5e12 / 132 SM / 1.755 GHz = ~4270
                FLOP/cycle/SM, so wgmma.m64n64k16 (131072 FLOP) is ~31 cycles and
                m64n256k16 (524288 FLOP) is ~123. Both columns below use it; per
                references/schedule-l3.md the criterion is the RATIO, which survives the rate
                being 2x wrong --
                except that the CUDA column is counted in instructions and does not scale
                with it, so halve the tensor rate and the CUDA share doubles.

  tensor cores  ~4.2k cycles busy per round: 72 QK x ~31 = ~2.2k, 16 PV x ~123 = ~2.0k.
                That is 17.8 MFLOP, the same figure checks.arithmetic_intensity computes
                from the other direction. ~2.4 us at 1.755 GHz.
                Idle at t0 (bubble A) and at risk at t8 (bubble B):
                  A  the round-boundary drain. Both groups end t9 with `warpgroup_wait<0>`,
                     so nothing is in flight when wg0 starts its softmax, and wg1 cannot
                     help -- it is blocked on sScale0Ready. Costs one wg0 softmax,
                     ~0.4k cycles [I], ~9% of the round. STRUCTURAL, not a tuning miss:
                     it follows from `m` living in smem (edge 5). This is what a second
                     resident CTA would have covered (checks.residency).
                  B  wg1's QK at t8 blocks on barriers_K1[4], issued at t5. Cover is
                     wg0's 16 QK at t6-t7, ~0.5k cycles ~ 0.28 us, against a 0.6-0.8 us
                     [I] HBM round trip -- so up to ~0.3-0.5 us is exposed on a DRAM
                     miss and ~nothing on an L2 hit. The sibling m_block's L2 reuse
                     (checks.arithmetic_intensity) is what decides which, and the
                     EVICT_FIRST hint is what puts it in doubt (open_questions).
  CUDA cores    ~1.0k of ~4.2k scheduler cycles across both groups = ~24%. Both groups'
                warps share the same four schedulers, so the two columns add rather than
                run in parallel. This is the falsifiable number (checks.falsifiability),
                and note what it is NOT: the seesaw hides wg1's ~320 instructions under
                wg0's t1 batch, but wg0's ~500 are bubble A, exposed.
  copy engine   busy at t3, t5, t6, t7 -- four issue points, 18 descriptors, 147456 B per
                round. Per BYTE it is the column with no slack: 147456 B against
                ~25.4 GB/s of DRAM per SM is ~5.8 us versus ~2.4 us of tensor time, and
                only the sibling m_block's L2 hits bring it to ~2.9 us. The engine idles
                while the memory system does not — which is what
                checks.arithmetic_intensity is measuring, and why bubble B is the one to
                measure first.
```

### L4 — instructions and threads

```
  QK^T   for t in <the phase's sub-tile order>:      # wg0: 0,1,2,3 then 4,5,6,7,8
                                                     # wg1: 4,5,6,7,8,0,1,2,3   (math.split)
           arrive_and_expect_tx(8192); wait barriers_K[t] @ phase
                                                     # sub-tile t landed; the others still in
                                                     # flight — how TMA latency is hidden here
           for ki in range(64*t, 64*t + 64, 16):     # iter: step 16 = wgmma K, 4 per sub-tile,
                                                     # 36 total over the full dk
             wgmma.m64n64k16(
               A = sQ[0:64, ki:ki+16]       smem-desc for t in 0..7;  RF (rQ8) for t == 8
               B = sK0ᵀ|sK1ᵀ[ki:ki+16, 0:64]  smem-desc, k-major
               C = rP0 | rP1 [64, 64]       f32 RF, 64*64/128 = 32 elems/thread
               clear = (first tile of the phase-0/1 chunk) )   # ScaleOut::Zero, splitkv_mla.cuh:222

  PV     for ki in range(0, 64, 16):                 # iter: start 0, stop kvb=64, step 16, trip 4
           wgmma.m64n256k16(
             A = rPb | rP1b[0:64, ki:ki+16]  RF (own P, rs_op_selector)
               | sP0 | sP1[0:64, ki:ki+16]  smem-desc (other WG's P, ss_op_selector)
             B = sV*[ki:ki+16, 0:256]       smem-desc, Major::MN — V is dv-major in smem,
                                            i.e. the tensor stored is sVᵀ[dv_h, kvb]
             C = rO [64, 256]               f32 RF, 64*256/128 = 128 elems/thread
             clear = never — rO is carried across the whole mainloop )
         # four per round per warp group: 2 P matrices x issued by both groups.

  PER-THREAD ACCESS, one round.  [D] unless marked.
  touch                     bits/thread   transactions             coalescing / conflicts
  ------------------------- ------------- ------------------------ ---------------------------
  KV gmem -> sK0/sK1        n/a           1 elected lane issues     TMA: no per-thread access
    TMA-2D x9 per block                   each descriptor, 8192 B   to state. 18 per round,
    (Q -> sQ same, once                   each; wg0 issues 0-3,     split between the two
     per request)                         wg1 issues 4-8            groups (L3 edge 1).
  sQ, sK read by wgmma      n/a           descriptor read           Layout_K_SW128_Atom
                                                                    -> 0-way. NOTE sK is read
                                                                    a SECOND time as V via
                                                                    SmemLayoutV (Major::MN):
                                                                    two patterns over one
                                                                    buffer, and SmemLayoutV is
                                                                    a COMPOSITION of the same
                                                                    atom (traits.h:56-59), so
                                                                    both are conflict-free. A
                                                                    hand-rolled swizzle would
                                                                    break one.
  sQ tile 8 -> rQ8          512 b         4 x LDSM.M8N8.X4          prologue only, per request
    (8192 B / 128 thr)                    (SM75_U32x4_LDSM_N)       — off the mainloop
  sP0 / sP1 publish         512 b         4 x stmatrix.m8n8.x4.b16  Widest legal store from a
    (8192 B / 128 thr)                    (SM90_U32x4_STSM_N)       wgmma acc fragment; the
                                                                    gather is the tiled copy's
                                                                    retile. On the sP0Ready path.
  sM read (running max)     32 b          1 wavefront, ideal 1     4 lanes own one row,
                                            8 distinct words, bcast 4x  computed [D]
  rO -> sOutputBuf, UNSPLIT 2048 b        16 x stmatrix.x4          into the bf16
    (32768 B / 128 thr)                                             Layout_K_SW128_Atom buffer
                                                                    -> 0-way, then TMA store.
  rO -> sOutputBuf, SPLIT   4096 b        64 x STS.64 (float2)      SMEM, f32, Shape<64,512>
    (65536 B / 128 thr)                                             : Stride<520,1>. See below.

  the stride-520 buffer, since the arithmetic is the interesting part
    It is a SHARED-MEMORY staging buffer (splitkv_mla.cuh:661-664), not the gmem tensor:
    gOAccum's row stride is HEAD_DIM_V = 512 with no padding (:1245-1248). Bank arithmetic
    applies to the smem buffer and only to it.
    A 64-bit STS splits a warp into 2 phases of 16 lanes. Lane l writes words
    (l/4)*520 + (l%4)*2 + C, and 520 mod 32 = 8, so its start bank is
    ((l/4)*8 + (l%4)*2 + C) mod 32 — 16 distinct EVEN banks over the phase, each lane
    covering {b, b+1}: all 32 banks exactly once -> 0-way.
    At stride 512 the row term is 0 mod 32, the 16 lanes collapse onto 4 start banks
    -> 4-way. The +8 words of padding is worth exactly that 4x, and nothing more; it is
    not the 32-way column-walk story a "row stride" invites.
    Read side: SM90_BULK_COPY_S2G, one 2048 B row per thread (:681) — a TMA store, so no
    per-thread smem read to analyse.

  addressing
    hoisted, per request  TMA descriptors (host-built CUtensorMap), smem bases, rQ8, and
                          scale_log2 = scale * log2(e) folded once [I]
    per round             2 x LDG.32 of block_table_ptr for r+2 and r+3, issued at the
                          TOP of each subroutine (:769-771, :881-882) — a full round ahead
                          of the TMAs that consume them, so the dependent load is off the
                          issue path. Plus 2 x 9 descriptor coordinate updates.
    per wgmma iter        none — descriptors built once, the ki offset is an immediate
```

`C = 128 elems/thread` is the number the whole kernel is built around: 128 × 256
threads = 32768 registers = exactly half the SM (`checks.acc_registers`).

The same PV written `rOᵀ[dv_h, m] += sVᵀ[dv_h, kvb] @ sPᵀ[kvb, m]` is the same
math and a different kernel: `dv_h` would become the wgmma M and need a different
atom and smem layout. This kernel uses the form above — `m` is M, `dv_h` is N —
which is why B carries `Major::MN`.

## Warp-group choreography

L3 showed one round across the engines. This is the barrier structure itself,
which the timeline references but does not carry.

| Barrier | Released by | Waited by | Breaks how, if reordered or dropped |
|---|---|---|---|
| `sMInitialized` | wg0's 128 threads, once (`:1125`) | wg0 only | wg0 reads `sM` before it is initialised; wg1 is ordered transitively through `sScale0Ready` |
| `sScale0Ready` | wg0, after `sScale0` (t0) | wg1 (t0) | `rO1` is rescaled against a stale max |
| `sScale1Ready` | wg1, after `sScale1` (t1) | wg0 (t2) | wg0 publishes `sP0` at the wrong max level; wg1's `sP0 @ sV0R` is silently wrong |
| `sP0Ready` | wg0, after `stmatrix -> sP0` (t3) | wg1 (t4) | wg1 reads a half-written `sP0` |
| `rO1sP0sV0RIssued` | wg1, after issuing `sP0 @ sV0R` (t4) | wg0 (t4) | wg0's next round overwrites `sP0` under a live wgmma |

`sScale1Ready` is the one that surprises: it makes wg0 wait on wg1 *inside the
same round*, so the seesaw is a **chain**, not two pipelines sharing smem. That
is the cost of putting `m` in shared memory, and bubble A is the bill.

## Why these numbers

Arguments the YAML cannot carry. Anything a `checks` block already states —
the smem total, the nine TMA copies, where each rounding lands — is not
repeated here.

**The seesaw exists because of the one number in `checks.acc_registers`.** Two
output accumulators — FA3's ping-pong — would need the entire register file,
leaving nothing for operands. So instead of two output matrices alternating
between CUDA-core and tensor-core phases, there is *one* output split
column-wise across two warp groups, and the groups alternate. Every named
barrier in `inter_group_sync` is a consequence. Change `cta_tile.N` or
`acc.location` and the whole schedule dissolves.

**Why the seesaw covers four barriers and not the fifth.** `checks.residency`
records the count; the reason is direction. Four are hand-offs where one group
waits on the *other*, so holding the groups out of phase leaves wgmma in flight
across each. `sScale0Ready` at the round boundary is the one where the chain
runs the wrong way: wg1 cannot start until wg0's softmax is done, and wg0's
softmax cannot start until its own scores land. Nothing inside the CTA can be
out of phase with that, which is why bubble A is structural rather than a tuning
miss — a consequence of `m` living in smem, priced in cycles by L3.

**No `setmaxnreg`, unlike DeepGEMM.** There is no lopsided producer to starve —
both groups need maximum registers. (The sibling sparse-FP8 kernel does use
192/160/152 splits, because it *does* have a dedicated non-math group.)

## Known risks

- **smem headroom is 1640 B** — a new buffer needs a new alias (`checks.smem`).
- **Registers bind before smem** (`checks.acc_registers`), and the source proves
  the author hit it: the no-op `if (start_block_idx - 16777216 < end_block_idx)`
  at `:1121` exists only to stop NVCC spilling. Check the compiler's register
  count on any change touching live ranges, not just correctness.
- **The named-barrier order is deadlock-sensitive** — five barriers, asymmetric
  arrivals; read the choreography table before touching one.
- **The kernel sits on the compute/memory boundary** (`checks.arithmetic_intensity`).
  At smaller `h_q` it falls to memory-bound and the seesaw's overlap stops paying.
- **Bubble B is the untested claim** — `checks.falsifiability`, first item to measure.
- **`is_no_split` must be pinned in any parity test** (`checks.rounding_contract`).
- **Split-KV needs the combine kernel** — `checks.acceptance`.
- **No performance number here was measured** — `checks.floor`, open_questions.
