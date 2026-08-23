# Worked example — DeepGEMM SM90 FP8 GEMM (1D1D)

Reverse-engineered from `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`
plus `mma/sm90.cuh`, `scheduler/gemm.cuh`, `csrc/jit_kernels/heuristics/sm90.hpp`
and `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp`, at `DeepGEMM @ 26f2661`.
Line references are `1d1d.cuh:NNN` unless named otherwise.

Read this one for: **GEMM**, **producer/consumer warp specialization**, the
**clean L3** (copy engine never idle), **persistent grids**, **TMA multicast**,
and the **drain-and-promote** accumulator pattern that fine-grained quantization
forces.

Contrast with `example-flashmla.md`, whose warp groups cooperate rather than
split producer from consumer.

---

```yaml
spec_version: 1
kernel: sm90_fp8_gemm_1d1d
status: reference               # reverse-engineered; open_questions records what the SOURCE does not settle
approved_by:                    # n/a for status: reference
source: deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh @ 26f2661

# ---------------------------------------------------------------- 0. problem
arch: sm90a                     # 1d1d.cuh:49 guards on __CUDA_ARCH__ >= 900; wgmma + TMA require the `a` suffix
problem:
  op: "D += sum_k (A_k @ B_k^T) * sfa_k * sfb_k, fp8 e4m3 operands, per-128-channel scales applied every K-block"
  dims: {M: dynamic, N: static, K: static}
  dynamic: [M]                  # SHAPE_M==0 -> runtime shape_m; N and K baked in by the JIT (1d1d.cuh:64-66)
  dtypes: {a: e4m3, b: e4m3, sfa: f32, sfb: f32, acc: f32, d: f32}
  layouts: {a: "k-major (M,K)", b: "k-major (N,K)", d: "row (M,N)"}
  regime: >
    Throughput; enough tiles to fill 132 persistent CTAs. Worked here at BLOCK_M=128,
    BLOCK_N=128 -- see open_questions.block_n: this fork's heuristic will not SELECT
    BLOCK_N=128 for the fp32-D path, though the kernel template is generic in it.

# ------------------------------------------------------------- 1. grid / CTA
grid:
  mode: persistent              # 1d1d.cuh:179,256 -- while (scheduler.get_next_block(...))
  ctas: 132                     # kNumSMs; launch_config.num_sms (sm90.hpp:256-259), LaunchArgs at sm90_fp8_gemm_1d1d.hpp:127
  shape: "(kNumSMs,) linear; scheduler maps CTA -> (m_block_idx, n_block_idx)"
  cta_tile: {M: 128, N: 128}
  rasterization: >
    Grouped, not row-major. kNum1DBlocksPerGroup = get_num_1d_blocks_per_group() picks
    8 or 16 by minimising `cand*BLOCK_N + ceil(132/cand)*BLOCK_M` (gemm.cuh:14-26); at
    128x128 both score 3200 and the tie goes to 8. get_swizzled_block_idx (gemm.cuh:137-173)
    then walks a panel of 8 primary blocks against every secondary block, so 8 consecutive
    CTAs sweep 8 B-tiles against one shared A-tile (kIsTMAMulticastOnA). Panel working set
    per K-block = 8*BLOCK_N + ceil(132/8)*BLOCK_M operand rows; DRAM order is a sequential
    sweep of the secondary operand slab within a panel. [I]
  l2_schedule: "defaulted to the runtime swizzle above. DeepGEMM is a JIT library over arbitrary (M,N,K), so there is no static graph to solve against"
  persistence:
    cta_per_sm: 1              # forced by smem AND registers -- see checks.residency, not by __launch_bounds__
    grid_realises_it: "yes -- grid = 132 = 132 SMs x 1"
    scheduler: >
      Static index arithmetic, no atomic and no work queue: block_idx = (++current_iter)*kNumSMs
      + blockIdx.x, then get_swizzled_block_idx (gemm.cuh:235,332). Every CTA walks a fixed
      stride-132 subsequence of the tile list, so tiles per CTA = ceil(num_blocks/132).
    phase_ordering: >
      None inside the kernel -- CTAs never wait on each other. Across kernels: PDL.
      cudaGridDependencySynchronize() at 1d1d.cuh:158 sits before the producer/math split,
      so the first TMA waits on the previous kernel's completion signal, not on a launch boundary.
  cooperative: false            # no CTA blocks on another CTA, and it would exclude the cluster below
  cluster:
    shape: [2, 1, 1]            # the LAUNCH cluster dim is (get_cluster_size(),1,1) (jit/handle.hpp:97);
                                # cluster_m/cluster_n are a logical split, swept in {1,2} (sm90.hpp:91-98)
    multicast: >
      A, and one TMA fills both CTAs of the cluster. kIsTMAMulticastOnA = (cluster_n > 1)
      (sm90_fp8_gemm_1d1d.hpp:61), and the heuristic lands on cluster_n=2, cluster_m=1: candidates
      are ranked by num_bytes_l2_ab = K*(BLOCK_M/cluster_n + BLOCK_N/cluster_m) (sm90.hpp:286),
      which halves whichever tile dim it divides. At a square 128x128 tile (1,2) and (2,1) tie at
      K*192, and the tie goes to the earlier candidate -- the list is built cluster_m-outer
      (sm90.hpp:91-92) and the comparator is a strict `<` seeded with candidate 0 (sm90.hpp:307,
      common.hpp:20-24). So A is the halved operand: K*(128/2 + 128) = 24576 B, the figure L3's
      copy column runs on, and the panel groups on N so 8 CTAs share one A tile (see rasterization).
      Mirror case: BLOCK_N > BLOCK_M makes (2,1) strictly cheaper, multicast moves to B, and the
      scheduler's panel groups on M instead (gemm.cuh:19-21,141-142). Every byte count below
      assumes the A case.
  launch:
    threads: 384                # kNumTMAThreads(128) + kNumMathThreads(256), sm90.hpp:256-257
    cta_per_sm: 1               # see checks.residency
    smem_B: 200960              # derived, see checks.smem
    max_regs_per_thread: 255

# ------------------------------------------------------------- 2. mainloop
mainloop:
  axis: K
  step: 128                     # BLOCK_K; forced -- DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling") at 1d1d.cuh:52
  trip_count: "ceil_div(shape_k, 128)"   # 1d1d.cuh:187,261
  tail: >
    ceil_div with no predication and no mask, and that is correct rather than merely tuned-around.
    The last K block is issued at k_idx = (n_k-1)*128 even when that runs past shape_k
    (1d1d.cuh:218,225). TMA does not over-read: the A/B tensor maps carry the true K extent as
    their gmem inner dim (runtime_utils.hpp:201), and the shared memory corresponding to the
    out-of-bounds part of a tile read from global is ZERO-FILLED -- CUDA C++ Programming Guide,
    TMA, "Negative indices and out of bounds". Zero e4m3 operands contribute nothing to the wgmma
    accumulator, so the ragged block adds an exact zero. The scales are never ragged: the SF
    tensor map's outer extent is itself ceil_div(shape_k, 128) (runtime_utils.hpp:263), the same
    n_k the loop counts, so sf_k_idx is always in bounds.
    That this is intended and not luck is visible in the host asserts: the Normal entry point
    constrains only C/D and majorness (sm90_fp8_gemm_1d1d.hpp:85-86) and says nothing about K,
    while the k-grouped entry point in the SAME file asserts every group's K is 128-aligned
    (:166) because group starts must align, and siblings that genuinely require it assert
    k % block_k == 0 outright (sm90_tf32_hc_prenorm_gemm.hpp:82). A ragged K is a supported
    shape here, not an untested one.
  operands_per_iter:
    - {name: SFA, tile: [128],      dtype: f32,  bytes: 512,   src: gmem, via: TMA-2D}   # issued first, 1d1d.cuh:229
    - {name: SFB, tile: [128],      dtype: f32,  bytes: 512,   src: gmem, via: TMA-2D}
    - {name: A,   tile: [128, 128], dtype: e4m3, bytes: 16384, src: gmem, via: TMA-2D}
    - {name: B,   tile: [128, 128], dtype: e4m3, bytes: 16384, src: gmem, via: TMA-2D}
  loop_carried: "[final_accum] -- 64 f32 in RF per thread. NOT the wgmma accumulator; see non_mma.dequant_promote."
  per_iter_math: "non_mma: scale_stage, then dequant_promote"

# ------------------------------------------------------------- 3. pipeline
pipeline:
  depth: 4                      # kNumStages = (232448 - 65792)/33792 = 4 (sm90.hpp:234-236); heuristics also
                                # reject <3, and <4 when block_m*block_n < 128*192 (sm90.hpp:134)
  stage_index: "iter_idx % kNumStages"          # get_pipeline(), 1d1d.cuh:165
  phase: "(iter_idx / kNumStages) & 1"          # same lambda; consumer waits on `phase`, producer on `phase ^ 1`
  prologue: "none explicit -- the producer runs ahead bounded only by empty_barriers, and iter_idx is declared outside the scheduler loop and never reset (1d1d.cuh:168), so the pipeline carries across tiles. The ramp therefore costs 3 stages of the FIRST tile only. [I]"
  per_stage_bytes: 33792        # 16384 + 16384 + align(512,128) + align(512,128), sm90.hpp:213-220
  staged_buffers:
    - {name: smem_a,   shape: [128, 128], dtype: e4m3, bytes: 16384, swizzle: 128B}   # kSwizzleAMode = get_swizzle_mode(BLOCK_K=128, 1) = 128 (sm90.hpp:168, utils.hpp:13)
    - {name: smem_b,   shape: [128, 128], dtype: e4m3, bytes: 16384, swizzle: 128B}
    - {name: smem_sfa, shape: [128],      dtype: f32,  bytes: 512,   swizzle: none}
    - {name: smem_sfb, shape: [128],      dtype: f32,  bytes: 512,   swizzle: "none, padded to 128B for TMA alignment"}
  non_staged_buffers:
    - {name: smem_d, bytes: 65536, swizzle: "NONE -- swizzle_cd_mode = 0 for fp32 D (sm90.hpp:173-174); see checks.vectorisation",
       aliases: none, alias_safe_because: "n/a -- allocated ahead of the staged region (1d1d.cuh:103-110)"}
    - {name: barriers, bytes: 256, aliases: none,
       alias_safe_because: "n/a -- host reserves kNumMaxStages*8*2 = 256 B (sm90.hpp:210); the device uses 2*depth*8 = 64 B of it"}
  barriers:
    - {name: full,  kind: mbarrier-tx, count: 4, init_arrive_count: 1,
       produced_by: "TMA elected thread, arrive_and_expect_tx(33792) after issuing the 4 copies (1d1d.cuh:233)",
       waited_by: "both math warp groups, on `phase`"}
    - {name: empty, kind: mbarrier, count: 4, init_arrive_count: 8,
       produced_by: "one arrival per math warp -- kNumTMAMulticast * kNumMathThreads / 32 = 1 * 8 (1d1d.cuh:140)",
       waited_by: "TMA warp, on `phase ^ 1`, before reusing the stage"}
  # Under multicast the empty count becomes 2*8=16 (each math warp arrives once per CTA of the
  # cluster, 1d1d.cuh:266-273) and the producer needs one extra round of empty waits after the
  # last tile so the distributed barriers can be torn down safely (1d1d.cuh:237-244).

# ------------------------------------------- 4. warp specialization / roles
warp_groups:
  - id: producer
    warps: 4
    threads: 128
    regs: 24                    # warpgroup_reg_dealloc<kNumTMARegisters>, 1d1d.cuh:154,172
    role: "issue TMA loads for every stage; own the persistent scheduler walk"
    issues: "cp.async.bulk.tensor (x4) + mbarrier.arrive.expect_tx; mbarrier wait on empty"
    elected: true               # warp_idx == kNumMathThreads/32 && cute::elect_one_sync() -- ONE thread of 128 does the work
  - id: math0
    warps: 4
    threads: 128
    regs: 240                   # warpgroup_reg_alloc<kNumMathRegisters>, 1d1d.cuh:155,248
    role: "rows 0-63 of the tile"
    issues: "18 ld.shared, wgmma.mma_async x4, 128 f32 ALU ops, then st.shared + TMA store"
    elected: false
  - id: math1
    warps: 4
    threads: 128
    regs: 240
    role: "rows 64-127 of the tile"
    issues: same as math0
    elected: false
inter_group_sync: >
  Only the full/empty mbarrier pair couples producer to math. There is NO math0<->math1
  barrier anywhere in the mainloop -- each owns a disjoint 64-row slice of the accumulator
  tile. NamedBarrier::sync(128, math_wg_idx) appears only in the epilogue, scoped to one
  warp group, to order that group's st.shared against its own TMA store (1d1d.cuh:326,337).
  Consequence for L3: the two math groups stagger by one wgmma batch because the tensor
  cores serialise them, not because anything orders them.

# ------------------------------------------------ 5. iters (instruction level)
math:
  - group: "math0 and math1 (identical, disjoint row ranges)"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 128, K: 32}
    contracts: K=128            # BLOCK_K, and it IS mainloop.axis here -- the plain case    # FP8MMA in mma/sm90.cuh:26-28; FP8MMASelector<128> -> MMA_64x128x32_F32E4M3E4M3_SS_TN (mma/sm90.cuh:51)
    dtype: "e4m3 x e4m3 -> f32"
    count_per_stage: 4                    # BLOCK_K / WGMMA::K = 128 / 32 (1d1d.cuh:297)
    a_source: smem-desc                   # make_smem_desc(smem_a + math_wg_idx*64*BLOCK_K + k*32, layout_type=1 = B128, mma/sm90.cuh:260-272)
    b_source: smem-desc
    acc: {name: accum, location: RF, elems_per_thread: 64, dtype: f32, cleared: "at iter 0 of every stage"}
    accumulate_across_iters: false
    after_batch: >
      warpgroup_commit_batch() + warpgroup_wait<0>(), then empty_barrier_arrive(stage)
      (1d1d.cuh:309), THEN the scaled promotion into final_accum -- see L3's ordering edges.
  # acc.cleared: WGMMA::wgmma(desc_a, desc_b, accum, k) passes the loop index as
  # `scale_d`, so iter 0 uses ScaleOut::Zero and iters 1-3 accumulate (1d1d.cuh:297-300,
  # mma/sm90.cuh:19). kNumAccum = M*N/128 = 64*128/128 = 64 f32 registers per thread.

# ------------------------------- 5b. non-MMA work (the CUDA-core column of L3)
non_mma:
  - id: scale_stage
    where: "mainloop.per_iter -- after full[s], BEFORE warpgroup_arrive"
    kind: "staging, smem -> RF; no arithmetic"
    over: "sfa rows [64w, 64w+64) and sfb cols [0,128), one K-block"
    span: lane
    primitive: none             # bespoke; no contract in references/primitives.md
    mechanism: "ld.shared.b32 x2 (scale_a_0/1) + ld.shared.b64 x16 (scales_b[i] as float2)"
    loop_carried: []
    dtype: "f32, no rounding -- a copy"
    cost: "18 LDS/thread, LSU (1d1d.cuh:283-289). Negligible against the 128-op promote. [I]"
    touches: "smem_sfa[s] at r_0 = 16*warp_idx + lane/4 and r_1 = r_0+8; smem_sfb[s] at i*8 + 2*(lane%4) (1d1d.cuh:252-253,283-289). The addresses are cited; the conclusion that each broadcasts within a quad to 1 transaction and 0-way conflict is [I]."
    on_critical_path: >
      Not for the MMA, but a correctness edge: it must precede warpgroup_arrive
      (1d1d.cuh:281-289, comment at :282) so the scales are in registers before empty[s]
      is released. See L3.
  - id: dequant_promote
    where: "mainloop.per_iter -- after warpgroup_wait<0> and after empty[s].arrive()"
    kind: elementwise
    over: "the 64 accumulator elems/thread (the whole 64x128 slice), every K-block"
    span: lane
    primitive: none
    mechanism: "register-local FMUL + FFMA"
    loop_carried: [final_accum]
    dtype: >
      f32 end to end. `accum[]` is the wgmma f32 output and is never narrowed; sfa*sfb
      rounds once, the FFMA into final_accum rounds once -- two roundings per element per
      K-block, K/128 times.
    cost: >
      64 FMUL + 64 FFMA = 128 f32 instructions/thread (1d1d.cuh:313-320). Per CTA-stage:
      2 WG * 128 thr * 128 = 32768 instructions at 128 fp32 lanes/SM/cycle = 256 cycles. The
      instruction count is [D]; the 128 lanes/cycle and therefore the 256 are [I].
    touches: "accum[64], final_accum[64], scale_a_0/1, scales_b[16] -- all RF. No smem touch at all."
    on_critical_path: "no -- registers only, the stage buffer is already released, and the other math WG's batch covers it. See L3."
  - id: d_reduce_add
    where: epilogue
    kind: "elementwise accumulate into gmem"
    over: "64 x 128 f32 per math warp group, once per output tile"
    span: cta
    primitive: none
    mechanism: "SM90_TMA_REDUCE_ADD_2D -- the copy engine's reduction unit, not the CUDA cores"
    loop_carried: []
    dtype: "f32 add performed in the TMA unit"
    cost: "0 thread-instructions; 1 issue from 1 elected thread per warp group (1d1d.cuh:340-345)"
    touches: "smem_d[64,128] f32, UNSWIZZLED, row stride 512 B -> gmem D. The st.shared that fills it is the kernel's one bank-conflict site; see L4."
    on_critical_path: "no -- after the mainloop, and the next tile's reuse of smem_d is gated by tma_store_wait<0> (1d1d.cuh:324-325)"

# ------------------------------------------------------------- 6. epilogue
epilogue:
  position: after-mainloop
  math: "none beyond the per-stage promotion already folded into final_accum"
  path: "rf -> st.shared into smem_d -> tma_store_fence -> SM90_TMA_REDUCE_ADD_2D -> gmem"
  output: {tile: [64, 128], dtype: f32}   # store_block_m = wgmma_m = 64: each math WG stores its own slab (sm90.hpp:164)
  split_reduction: >
    reduce-add at the TMA store. The host requires c.has_value() (sm90_fp8_gemm_1d1d.hpp:85),
    so D is always accumulated into, which is what lets partial results from a split land
    in place. Upstream DeepGEMM uses a plain TMA store.

# ------------------------------------------------------------- 7. checks
l4_accesses: accesses-deepgemm.yaml   # scripts/tv_check.py computes L4's table from it
checks:
  smem: >
    Host formula (sm90.hpp:208-238): smem_extra = smem_cd 65536 + smem_barriers 256 = 65792;
    smem_per_stage = 33792; depth = (232448 - 65792)/33792 = 4; total = 65792 + 4*33792 =
    200960 B of the 232448 B cap -> PASS, 31488 B spare. depth 5 needs 234752 -> would NOT
    fit. The 64 KB fp32 epilogue buffer is the term that caps depth, and 1d1d.cuh:56 asserts
    C/D is fp32, so the bf16-D lever that would free 32 KB is closed in this kernel.
  threads: "128 + 128 + 128 = 384 == __launch_bounds__ first arg -> PASS. All groups are 128-thread multiples, required for wgmma."
  acc_registers: >
    128*128/256 = 64 f32 accumulator elems/thread == WGMMA::kNumAccum -> PASS. But
    drain-and-promote needs a SECOND 64-register array (final_accum) plus 16 float2 of B
    scales and 2 A scales: ~162 registers of live state before addressing. This is why the
    math groups take 240 and why sm90.hpp:72-73 caps BLOCK_N at 160 "register spills".
  register_budget: "128*24 + 256*240 = 64512 <= 65536 per SM -> PASS with 1024 spare. (The KGroupedContiguous variant uses 40/232 and lands on exactly 64512 too -- the split is chosen to saturate the SM either way.) All four values are multiples of 8, as setmaxnreg requires."
  mma_k: "4 iters * 32 == 128 == mainloop.step -> PASS"
  mma_m: "64 * 2 math groups == 128 == cta_tile.M -> PASS. Enforced in source: DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2)) at 1d1d.cuh:258."
  mma_n_legal: "N=128 is a legal wgmma atom (multiple of 8, <= 256) and is in FP8MMASelector -> PASS"
  trip_count: "ceil_div(K,128)*128 >= K -> PASS. The overhang is named in mainloop.tail: no predication, TMA zero-fill."
  output_coverage: >
    The scheduler enumerates every (m_block, n_block) exactly once -> PASS. M is dynamic, so the
    last m-block is ragged, and nothing masks it: the same TMA edge semantics as mainloop.tail
    cover it in both directions -- the A rows past shape_m come in zero-filled, and the
    SM90_TMA_REDUCE_ADD_2D of those rows is out of bounds on the store side and is dropped. So
    the ragged rows compute zeros that are never written -> PASS, by TMA clipping, not by
    predication.
  occupancy: >
    smem 200960 > 232448/2 and registers 64512 > 65536/2, so either limit alone forces 1 CTA/SM
    -> PASS. __launch_bounds__(384, 1) (1d1d.cuh:39) is CONSISTENT WITH this but is not evidence
    for it: the second argument is a minimum-blocks-per-SM hint that constrains the register
    allocation, and it cannot cap occupancy (see example-phase0.md:120-123). Asking for >= 1 block
    per SM is the weakest form of that hint and rules nothing out; the smem and register
    arithmetic is the whole argument.
  barrier_arrivals: "full=1 (single elected TMA thread) -> PASS. empty=8 = one per math warp; becomes 16 under 2-CTA multicast -> PASS. Phase rule stated -> PASS."
  arithmetic_intensity: >
    Per CTA tile: 2*128*128*K FLOP over (128+128)*K bytes = 128 FLOP/byte. H100 SXM5 fp8
    DENSE ridge is 1979e12 / 3.35e12 = 591 FLOP/byte, so the tile is ~4.6x below ridge and
    cannot saturate the fp8 tensor cores from DRAM on its own -> FLAG. Two other fields
    carry it: cluster multicast halves A's DRAM traffic, and the grouped rasterization keeps a
    panel of 8 CTAs on the same A tile in L2. Both ridge terms are [I] -- published peaks, not
    measured on this machine.
  floor: >
    n/a -- no Phase 0 measurements exist for this checkout, so there is no floor to compare
    against. WOULD NEED MEASUREMENT: empty-kernel cost at 132 CTAs / 200960 B smem, cold
    streaming bandwidth vs size, bandwidth vs CTA count.
  reference: >
    tests/test_fp8_fp4.py:53 benches deep_gemm.cublaslt_gemm_nt on the same shape alongside
    the kernel. DeepGEMM is not fused, so this is a plain single-call cuBLAS FP8 comparison,
    not a composition. WOULD NEED MEASUREMENT: no number is recorded in the source.
  acceptance: >
    NOT NAMED by the source, and the two candidates differ: tests/test_fp8_fp4.py measures
    with bench_kineto on an isolated call, while the kernel ships inside a captured graph
    behind PDL (1d1d.cuh:158), where grid ramp and the dependent-launch overlap both change.
    A spec reusing this design has to pick one.
  falsifiability: >
    depth 4 beats 3 -> build with kNumStages=3 and re-bench. Multicast + rasterization carry
    the sub-ridge tile -> lts__t_sector_hit_rate and dram__bytes_read against the panel's
    unique-bytes prediction. Early empty-release pays -> move empty_barrier_arrive below the
    promote loop and re-bench. The 8-way smem_d conflict ->
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st. None was run here.
  concurrency: >
    Copy engine idle 0 cycles in steady state; tensor cores idle only in the first tile's
    prologue; CUDA cores ~50% idle and not the bottleneck -> PASS. See L3. Which column is empty
    is structural and follows from the ordering edges; every idle FRACTION here is [I], resting on
    the modelled cycles in open_questions.engine_cycles.
  vectorisation: >
    Every touch at L4 states bits/thread. sfa 32 b, sfb 64 b, smem_d store 64 b -- all
    conflict-free EXCEPT the smem_d store, which is 8-way conflicted at BLOCK_N=128 because
    swizzle_cd_mode is 0 for fp32 D and the 512 B row stride is an exact multiple of the 128 B
    bank cycle -> FLAG, computed at 8 wavefronts against an ideal of 2 (4x) from
    references/accesses-deepgemm.yaml. Widths and conflict counts are both [D] now; only the
    ncu counter's naming is left open (open_questions.bank_ways).
    The source dodges it by tile size, not by swizzle: sm90.hpp:59-67
    builds the BLOCK_N candidate list so that BLOCK_N % 32 != 0 always, commented "Avoid
    bank conflicts for 1D1D kernel FP32 output".
  addressing: >
    r_0, r_1, col_idx and math_wg_idx are computed once before the scheduler loop
    (1d1d.cuh:251-253) and held in registers, with __shfl_sync used to force uniform
    registers. Per mainloop iteration the only arithmetic is get_pipeline: one AND and one
    shift (depth 4 is a power of two), plus one IMAD per staged-buffer base. With
    #pragma unroll kNumPipelineUnrolls = kNumStages (1d1d.cuh:151,217,275) the stage index
    becomes a compile-time constant in the unrolled body and all of it folds away, leaving
    only the phase bit -> PASS. The unroll is paid for in registers: the KGroupedContiguous
    variant sets kNumPipelineUnrolls=0 and drops the math groups from 240 to 232.
  tile_order: "grid.rasterization carries the panel width and the tie-break that produced 8 -> PASS. grid.l2_schedule says defaulted, with the reason (JIT over arbitrary shapes) -> PASS."
  residency: >
    cta_per_sm = 1, and BOTH limits bind: 200960 B > 232448/2 and 64512 regs > 65536/2. The
    2-or-4 alternative is not merely unchosen, it is unreachable -- get_pipeline_config
    (sm90.hpp:234-236) sets depth to whatever fills smem, so shrinking the tile grows the
    stage count and smem stays near the cap. BLOCK_M=64 would drop to 1 math WG
    (sm90.hpp:257) at 128*24 + 128*240 = 33792 regs, still over 65536/2, and depth would
    rise to 7 for 212224 B. 1 is correct HERE because the regime is throughput at large M:
    each CTA has K/128 stages of independent work per tile and depth-4 lookahead across
    them, so the per-stage mbarrier wait is covered by the pipeline rather than by a second
    CTA. That argument fails at small M -- see example-flashmla.md. -> PASS.
  persistence: "grid 132 == SM_count * cta_per_sm(1) -> PASS. cooperative false and no CTA waits on another -> PASS. No semaphore, so nothing to self-reset under graph replay -> n/a. cluster [2,1,1] present, which is legal only because cooperative is false (sm90 excludes the pair) -> PASS."
  non_mma_accounting: >
    All three non_mma entries appear in L3: scale_stage and dequant_promote in the CUDA-core
    column with their costs, d_reduce_add in the copy-engine column of the epilogue (not a
    steady-state stage). dequant_promote.loop_carried [final_accum] matches
    mainloop.loop_carried -> PASS. No entry is on_critical_path: yes.
  rounding_contract: >
    dequant_promote.dtype pins both roundings (sfa*sfb, then the FFMA), and accum is never
    narrowed. The test reference is (a.float() @ b.float().T + c) computed from the
    PRE-quantization bf16 tensors (tests/generators.py:314), so the tolerance absorbs
    quantisation error as well as accumulation order -> PASS, with the caveat that this
    reference cannot detect a promotion-order bug on its own.
  traceability: >
    Walked L4 -> L2 -> L1. PASS.
    L4 -> L2: ki's 128 is mainloop.step and its 32 is inst_shape.K, trip 4 = count_per_stage;
    smem_a[64w:64w+64] and accum[64,128] take their 64 from inst_shape.M, their 128 from
    inst_shape.N = cta_tile.N; 64 elems/thread = acc.elems_per_thread = checks.acc_registers;
    the sfb x16 and smem_d x32 counts are kNumAccum/4 = 16, also acc.elems_per_thread; r_0's
    stride of 16 rows is inst_shape.M / 4 warps. Two L4 numbers that look like codegen detail
    and are NOT -- both trace to an L2 field, which is the point of the check: the descriptor's
    SBO 1024 is the 128 B swizzle atom x 8 rows, i.e. staged_buffers.smem_a.swizzle; and the
    512 B row stride behind the 8-way conflict is non_staged_buffers.smem_d at cta_tile.N=128,
    so the only cure for it is a different BLOCK_N, which is exactly what sm90.hpp:59-67 does.
    L2 -> L1: m0/n0/k0 and their 128s are cta_tile.M/N and mainloop.step; n_k is
    mainloop.trip_count; s and phase are pipeline.stage_index and pipeline.phase; 64w is the
    math0/math1 row split.
    Contraction axis: the shared inner name at L1 is k0 = K = mainloop.axis. At L2 both operand
    slices read 0:128 and the axis is positional (smem_a's second index, smem_b ᵀ's first) -- annotated
    in the nest because at a square tile it is not visible by eye. At L4 it is ki on both -> K.
    Names: the nest uses the SOURCE identifiers throughout -- smem_a/smem_b/smem_sfa/smem_sfb are
    staged_buffers, accum is math.acc.name, final_accum is mainloop.loop_carried. The one pair
    without a source symbol at tile granularity is scale_a/scale_b, annotated in the nest as the
    tile view of per-thread scale_a_0/1 and scales_b[16].
  loop_bounds: >
    Six loops. L1 m0 range(0,M,128) trip ceil(M/128), n0 range(0,N,128) trip ceil(N/128), k0
    range(0,K,128) trip ceil(K/128) == mainloop.trip_count. L2's tile walk is not a range and
    says so, carrying trip ceil(num_blocks/132) to match grid.persistence.scheduler. L2's k0 is
    range(0, 128*n_k, 128) trip n_k == mainloop.trip_count. L4's ki is range(0, 128, 32) trip 4
    == math.count_per_stage == checks.mma_k. -> PASS.
    The one bound worth writing out rather than abbreviating: L2's stop is 128*n_k, NOT K. They
    differ by up to 127 whenever K % 128 != 0, and that gap is the whole of mainloop.tail. A nest
    that wrote `range(0, K, 128)` at L2 would have the same trip count and would hide it.

# ------------------------------------------------------------- 8. handover
verification:
  reference: "tests/generators.py:314 -- ref_d = (a.float() @ b.float().t() + c), from the bf16 tensors BEFORE fp8 quantization, not from the dequantized fp8 inputs"
  tolerance: "calc_diff < 0.001, where calc_diff = 1 - cosine similarity computed in float64 (deep_gemm/testing/numeric.py:5-11, QuantConfig.max_diff at tests/generators.py:65-70)"
  perf_target: "TFLOP/s from bench_kineto, printed beside cublaslt_gemm_nt on the same shape (tests/test_fp8_fp4.py:51-53). No number is baked into the source."
open_questions:
  - id: block_n
    q: >
      This checkout's heuristic cannot SELECT BLOCK_N=128 for this kernel. sm90.hpp:59-76
      builds the candidate list as {16} + {24, 40, 56, ..., 152} whenever kernel_type is
      1D1D and cd_dtype is fp32 -- and 1d1d.cuh:56 asserts cd_dtype is fp32 -- so no
      candidate is a multiple of 32. The template is generic in BLOCK_N and compiles at 128;
      the spec above is worked there because the arithmetic is legible. Reproducing what this
      fork actually launches means re-deriving at a reachable BLOCK_N: 120 gives kNumAccum=60,
      per_stage=32768, smem_cd=61440, depth=5.
  - id: bank_ways
    q: >
      SETTLED as a design number, open only as a counter name. The smem_d store is computed
      at 8 wavefronts against an ideal of 2 -- a 4x serialisation -- by enumerating the warp
      in references/accesses-deepgemm.yaml, so the tile-order argument does not rest on an
      inference any more. What remains is a labelling question: a 64-bit st.shared is
      processed in two 16-lane phases, so ncu may report 4-way per phase rather than 8-way
      over one. The ratio to conflict-free is 4x under either reading, so nothing in this
      spec turns on it. l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st would name it.
  - id: engine_cycles
    q: >
      INFERRED: L3's cycle figures come from DeepGEMM's own cost model (l1 128 B/cycle/SM,
      l2 min(64*num_sms, 8e6/1.3e3) B/cycle, sm90.hpp:277-278) plus a 4096 fp8 MAC/cycle/SM
      tensor-core assumption. They are a balance argument, not a measurement.
deviations: []
```

## Loop nest

### L1 — iteration space

```
  gemm(A[M,K] e4m3 k-major, B[N,K] e4m3 k-major,
       SFA[M, K/128] f32, SFB[N, K/128] f32, C[M,N] f32) -> D[M,N] f32

  # M is dynamic (SHAPE_M==0 -> runtime shape_m); N and K are baked in by the JIT.
  D[:, :] = C[:, :]                            # D is accumulated into, not written
  for m0 in range(0, M, 128):                  # ceil(M/128) tiles     parallel
    for n0 in range(0, N, 128):                # ceil(N/128) tiles     parallel
      for k0 in range(0, K, 128):              # ceil(K/128) steps     SERIAL, contraction axis
        D[m0:m0+128, n0:n0+128] += SFA[m0:m0+128, k0//128] * SFB[n0:n0+128, k0//128]
                                   * ( A[m0:m0+128, k0:k0+128] @ Bᵀ[k0:k0+128, n0:n0+128] )
                                       (128,128) @ (128,128) -> (128,128)

  # step 128 on k0 is forced, not chosen: DG_STATIC_ASSERT(BLOCK_K == 128) because the
  # scale granularity is per-128-channel (1d1d.cuh:52). At K % 128 != 0 the last slice here
  # is short, and it is the ONE place L1 and L2 differ: L2 reads a full 128 and takes the
  # overhang as zeros. See mainloop.tail.
  # SFA/SFB have ceil(K/128) columns, so their last column is not short.
```

### L2 — mapped to hardware

```
  grid: 132 persistent CTAs. The (m0, n0) loops are flattened and walked by
  sched::Scheduler in panels of 8 primary blocks, so a panel's CTAs share an operand tile
  in L2 and a cluster's two CTAs share a multicast TMA. k0 stays serial inside each CTA.

  384 threads = 1 producer WG (128) + 2 math WGs (128 each).
  math WG w in {0,1} owns rows [64w, 64w+64) of the 128x128 tile.

  cudaGridDependencySynchronize()               # PDL: wait for the producing kernel, 1d1d.cuh:158

  for (m0, n0) in scheduler(cta_id):           # not a range: stride-132 walk of the flattened
                                               # (m0,n0) list, trip ceil(num_blocks/132) per CTA,
                                               # 1d1d.cuh:179,256
    final_accum[64, 128] = 0                   # mainloop.loop_carried, f32 RF, 64 elems/thread
    n_k = ceil_div(K, 128)                      # mainloop.trip_count, 1d1d.cuh:187,261

    for k0 in range(0, 128*n_k, 128):          # mainloop: start 0, stop 128*n_k, step 128, trip n_k.
                                               # stop is 128*n_k and NOT K -- the last block runs
                                               # past K when K % 128 != 0; see mainloop.tail
      s, phase = (k0//128) % 4, (k0//128)//4 & 1        # depth 4, 1d1d.cuh:165

      producer  (1 elected thread of 128, 24 regs)
                wait empty[s] @ phase^1
                smem_sfa[s][128]     <- SFA[m0:m0+128, k0//128]         512 B  TMA   # issued
                smem_sfb[s][128]     <- SFB[n0:n0+128, k0//128]         512 B  TMA   # in this
                smem_a  [s][128,128] <- A  [m0:m0+128, k0:k0+128]     16384 B  TMA   # order,
                smem_b  [s][128,128] <- B  [n0:n0+128, k0:k0+128]     16384 B  TMA   # :229-232
                full[s].arrive_and_expect_tx(33792)                # 1d1d.cuh:233
                # under 2-CTA multicast smem_a is the one filled for both CTAs at once

      math WG w (128 threads, 240 regs)
                wait full[s] @ phase
                scale_a[64] = smem_sfa[s][64w:64w+64]              # non_mma.scale_stage; the tile
                scale_b[128] = smem_sfb[s][0:128]                  # view of scale_a_0/1, scales_b[16].
                                                                   # MUST precede warpgroup_arrive
                accum[64,128] = smem_a[s][64w:64w+64, 0:128] @ smem_b[s]ᵀ[0:128, 0:128]
                                (64,128) @ (128,128) -> (64,128)   f32 RF, stage-local
                                # the shared inner slice is smem_a's SECOND index and smem_b ᵀ's FIRST:
                                # both are K, extent mainloop.step = BLOCK_K = 128
                empty[s].arrive()                                  # release early, 1d1d.cuh:309
                final_accum += scale_a * scale_b * accum           # non_mma.dequant_promote, :313-320

    epilogue  (per math WG, 64 rows each)
              tma_store_wait<0>                                    # last tile's store, :324-325
              final_accum -> smem_d[64w:64w+64, 0:128] via st.shared   # 65536 B buffer, not staged
              tma_store_fence; NamedBarrier::sync(128, w)          # 1d1d.cuh:326,337
              D[m0+64w : m0+64w+64, n0:n0+128] += SM90_TMA_REDUCE_ADD_2D(smem_d)
```

### L3 — schedule

```
  engine timeline, steady state. The math columns consume stage s; the copy column fills
  s', whichever buffer was released most recently. The producer floats 1-3 stages ahead
  because the two columns are nearly balanced (527 vs 512 cyc below) and depth 4 absorbs
  the drift -- [I]: the BOUND is structural (empty[] pins it to 1..depth-1), where inside that
  range it settles is a consequence of the modelled cycles. The two math WGs run the same
  program with NO barrier between them; the tensor cores serialise their wgmma batches, and
  that is the only thing that staggers them. One batch = 4 wgmma = 256 cyc [I], so t1..t4 is
  one 512-cyc stage.

    copy engine (TMA)               CUDA cores (LSU/ALU)              tensor cores (WGMMA)
    ------------------------------- --------------------------------- -------------------------
 t0 producer: wait empty[s'] @ ph^1 WG0: promote(s-1), 128 f32 ops    WG1: batch of stage s-1
    1 elected thread of 128         WG1: warpgroup_wait<0>, blocked
 t1 tma::copy x4 -> smem[s']        WG0: wait full[s], 18 LDS,        WG0: batch of stage s
    sfa,sfb,A,B  33792 B  :229-232       warpgroup_arrive, issue 4
 t2 full[s'].arrive_and_            WG1: wait<0> returns;             WG0:  "
    expect_tx(33792)  :233               empty[s-1].arrive()  :309;
                                         promote(s-1)  :313-320
 t3 wait empty[s'+1] @ ph           WG1: wait full[s], 18 LDS,        WG1: batch of stage s
                                         warpgroup_arrive, issue 4
 t4 tma::copy x4 -> smem[s'+1]      WG0: wait<0> returns;             WG1:  "
                                         empty[s].arrive()  :309;
                                         promote(s), 128 f32 ops

  ORDERING EDGES, and which are real
    empty[s].arrive() sits ABOVE the promote (:309 before :313), so the copy engine refills
      stage s during the 128 promote ops instead of after them. It is SAFE only because
      scale_stage already pulled sfa/sfb into registers before warpgroup_arrive -- the source
      says so at :282 ("all shared memory read must be prior to `warpgroup_arrive`").
      Release-early and read-scales-early are one decision, not two.
    empty[s] needs 8 warp arrivals (4 warps x 2 WGs), so the stage reopens only once BOTH
      groups' wgmma(s) have retired. That much is structural; WHERE the two arrivals land --
      WG0 at t4, WG1 as its own batch drains just after, both still before either group's
      promote -- follows from the modelled cycles. [I]
    the promote (CUDA cores) overlaps the OTHER group's wgmma (tensor cores): different
      units, disjoint registers, and nothing orders them -- the overlap is emergent, not
      enforced. With a single math WG, warpgroup_wait<0> would put the promote strictly
      between two batches and the tensor cores would idle for its whole duration, a third
      of the stage.
    the ONE true serialisation is full[s] -- bytes must land before wgmma reads smem.

  BUBBLE CHECK   (cycles per CTA-stage. EVERY number below is [I] -- DeepGEMM's own L1/L2 model
                  plus a tensor-core assumption, see open_questions.engine_cycles. The criterion
                  is the 527:512 RATIO, which survives both being 20% wrong; the absolutes do not.)
    copy engine idle    0. Work per stage: 32768 B of A/B into smem at 128 B/cyc/SM = 256
                        cyc of L1 write, and 24576 B out of L2 with multicast at 46.6
                        B/cyc/SM = 527 cyc (sm90.hpp:277-278,286). 527 marginally exceeds
                        the tensor cores' 512, so it never runs far enough ahead to reach an
                        empty[] wait; the residual wait lands on the math side as full[s].
                        Nothing on the CUDA-core column sits between the copy and tensor
                        columns, because empty[] is released before the promote.
    tensor cores idle   prologue of the FIRST tile only (stages 0..2); the persistent loop
                        carries the pipeline across tiles. Work: 8 wgmma x 64 cyc = 512 cyc,
                        and t1..t4 above is exactly those 512 cycles back to back.
    CUDA cores idle     ~50%. Work: 2 x 128 cyc of promote inside a 512-cyc stage. Fine --
                        they are not the bottleneck, and each group's promote is covered by
                        the other group's batch.

    The binding column is the copy engine's L2 read (527) against the tensor cores (512):
    within 3%, which is what arithmetic_intensity already flagged from the other direction.
    This tile is balanced ONLY because multicast halves A's share of that L2 read and the panel
    keeps both operand slabs L2-resident to begin with. Drop multicast and the copy column goes
    to 32768/46.6 = 703 cyc against the same 512; drop the panel and the same bytes arrive from
    DRAM instead of L2. Either inverts the timeline.
```

### L4 — instructions and threads

```
      for ki in range(0, 128, 32):             # iter: start 0, stop BLOCK_K=128, step 32, trip 4
        wgmma.m64n128k32(                      # FP8MMASelector<128> -> MMA_64x128x32_F32E4M3E4M3_SS_TN
          A = smem_a[s][64w:64w+64, ki:ki+32]     smem-desc, k-major, layout_type=1 (B128), SBO 1024
          B = smem_b[s][ki:ki+32, 0:128]ᵀ         smem-desc, k-major, layout_type=1 (B128)
          C = accum[64, 128]                     f32 RF, kNumAccum = 64*128/128 = 64 elems/thread
          clear = (ki == 0) )                  # loop index passed as scale_d: iter 0 uses
                                               # ScaleOut::Zero, iters 1-3 accumulate (:297-300)
      warpgroup_commit_batch(); warpgroup_wait<0>()

  PER-THREAD ACCESS, every gmem/smem touch in a steady-state stage.
  Widths, counts and addresses are [D]; every CONFLICT-WAY and TRANSACTION count is [I].
    smem_a, smem_b fill    NO per-thread access -- the copy engine writes smem. 1 elected thread
                     issues 1 cp.async.bulk.tensor per operand, which is why the producer
                     needs only 24 registers.
    smem A/B read    NO per-thread ld.shared either -- wgmma reads smem through a matrix
                     descriptor. 128 B swizzle atom, base 1024 B aligned (:93). [I] on the
                     descriptor read itself; the atom is verified conflict-free for the
                     ldmatrix-shaped gather in scripts/tests/known_answers.yaml.
    sfa load         32 b/thread (ld.shared.b32) x2 (:283-284). 8 distinct words per warp,
                     broadcast 4x -> 1 wavefront, ideal 1 -> 1x. [D]
    sfb load         64 b/thread (ld.shared.b64, float2) x16 (:288-289). 8 distinct words per
                     warp, broadcast 8x -> 1 wavefront, ideal 1 -> 1x. [D]
    smem_d store     64 b/thread (st.shared.b64) x32 (:332-334), EPILOGUE ONLY, once per output
                     tile. smem_d is unswizzled (swizzle_cd_mode = 0 for fp32 D) and its row
                     stride is BLOCK_N*4 = 512 B = 4 exact bank cycles, so the row index drops
                     out of the bank index entirely: 64 distinct words land on 8 banks.
                     8 WAVEFRONTS against an ideal of 2 -> 4x SERIALISATION. [D]
                     sm90.hpp:59-67 keeps BLOCK_N off every multiple of 32 for exactly this
                     reason -- an L4 symptom whose only cure is the L2 number BLOCK_N.
    D store          NO per-thread access -- 1 elected thread per math WG issues one
                     SM90_TMA_REDUCE_ADD_2D over its own 64 x 128 slab (:340-345).
    addressing       HOISTED to before the scheduler loop (:251-253): math_wg_idx, row_idx,
                     col_idx, r_0, r_1 -- __shfl_sync forces them into uniform registers.
                     PER-ITERATION: get_pipeline = 1 AND + 1 shift (depth 4 is a power of
                     two) and 1 IMAD per staged base. All of it folds to nothing under
                     #pragma unroll kNumPipelineUnrolls = 4, which makes the stage index a
                     compile-time constant and leaves only the phase bit. Cost of the
                     unroll: 8 registers per math thread (240 vs 232 unrolled/not).
```

## Warp-group choreography

Deleted: the two math groups have no barrier between them at all, so there is no
hand-off pattern to draw. The one thing their coexistence buys — the tensor cores
staying busy through a promote — is in L3's ordering edges. Compare
`example-flashmla.md`, where this section carries the algorithm.

## Why these numbers

**depth 4, not 5.** The 64 KB fp32 epilogue buffer is a fixed cost that does not
scale with depth, so it eats two stages' worth of smem before the pipeline gets
any (arithmetic in `checks.smem`). What makes 4 the *tuned* number rather than
merely the largest that fits: the heuristic's floor of 4 stages for tiles under
128×192 (`sm90.hpp:134`) and the smem ceiling of 4 meet exactly here, so the
config is on the boundary in both directions.

**1 CTA/SM.** Argued with its arithmetic in `checks.residency`, and not repeated
here. The one thing to carry away: it is a consequence of the throughput regime,
and the same reasoning inverts at small M — see `example-flashmla.md`.

**Two math warp groups, not one.** wgmma's M is fixed at 64, so a 128-row tile
needs two warp groups and the source asserts it (`1d1d.cuh:258`). It also keeps
the accumulator at 64 registers per thread instead of 128 — one warp group on a
128×128 tile would need 128 accumulator registers plus 128 more for `final_accum`
and would spill. It is also what keeps the tensor cores busy through a promote —
see L3's third ordering edge.

**Drain-and-promote instead of a running wgmma accumulator.** Per-128-channel
scales change every K-block, so products from different K-blocks cannot be summed
in the tensor-core accumulator. The extra cost — one more 64-register array and
128 f32 instructions per stage — is placed *after* the empty-barrier arrival, and
is only safe there because the scales were staged into registers before
`warpgroup_arrive`.

**24 / 240 register split.** The producer does nothing but compute addresses and
issue TMA from one elected thread, so it is starved down to 24 registers to fund
the math groups at 240 — a split sized to consume the SM's register file almost
exactly at 1 CTA/SM (`checks.register_budget`).

## Known risks

- **smem is on the boundary.** Any added buffer forces depth 3, which the
  heuristics reject. A bf16 D would buy 32 KB, but `1d1d.cuh:56` asserts fp32,
  so that lever is closed in this kernel.
- **The unroll factor is a register decision.** The `KGroupedContiguous` variant
  sets `kNumPipelineUnrolls = 0` and drops the math groups to 232 registers.
  Changing it moves both `checks.register_budget` and `checks.addressing` — the
  per-iteration stage arithmetic comes back when the unroll goes away.
- **The tile is below the fp8 ridge point** and L3 shows the copy engine's L2
  read as the binding column at 527 cycles against 512 of wgmma. Performance
  depends on multicast and rasterization, not on the mainloop. Profile
  `lts__t_sector_hit_rate`, not just tensor-core utilization.
- **The BLOCK_N=128 config worked above is not selectable in this checkout.**
  See `open_questions.block_n`. Anyone reusing these numbers as a starting point
  for a real launch must re-derive them at a candidate BLOCK_N.
- **Barrier teardown under multicast.** The extra round of empty waits after the
  last tile is required, not defensive.
