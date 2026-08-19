# Worked example — DeepGEMM SM90 FP8 GEMM (1D1D)

Reverse-engineered from `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`
plus `mma/sm90.cuh` and `jit_kernels/heuristics/sm90.hpp`. Cited to that source;
line references are `1d1d.cuh:NNN` unless named otherwise.

Read this one for: **GEMM**, **producer/consumer warp specialization**,
**persistent grids**, **TMA multicast**, and the **drain-and-promote**
accumulator pattern that fine-grained quantization forces.

Contrast with `example-flashmla.md`, whose warp groups cooperate rather than
split producer from consumer.

---

```yaml
spec_version: 1
kernel: sm90_fp8_gemm_1d1d
status: approved
source: deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh

# ---------------------------------------------------------------- 0. problem
arch: sm90a                     # 1d1d.cuh:49 guards on __CUDA_ARCH__ >= 900; wgmma + TMA require the `a` suffix
problem:
  op: "D = sum_k (A_k @ B_k^T) * sfa_k * sfb_k, fp8 e4m3 operands, per-128-channel scales applied every K-block"
  dims: {M: dynamic, N: static, K: static}
  dynamic: [M]                  # SHAPE_M==0 -> runtime shape_m; N and K baked in by the JIT (1d1d.cuh:64-66)
  dtypes: {a: e4m3, b: e4m3, sfa: f32, sfb: f32, acc: f32, d: f32}
  layouts: {a: "k-major (M,K)", b: "k-major (N,K)", d: "row (M,N)"}
  regime: "throughput; enough tiles to fill 132 persistent CTAs. Chosen config below is BLOCK_M=128, BLOCK_N=128."

# ------------------------------------------------------------- 1. grid / CTA
grid:
  mode: persistent              # 1d1d.cuh:179,256 -- while (scheduler.get_next_block(...))
  ctas: 132                     # kNumSMs; one CTA per SM, tiles come from the scheduler
  shape: "(kNumSMs,) linear; scheduler maps CTA -> (m_block_idx, n_block_idx)"
  cta_tile: {M: 128, N: 128}
  rasterization: "sched::Scheduler<...> (scheduler/gemm.cuh) -- grouped rasterization so neighbouring CTAs share an operand tile in L2, and so a cluster's CTAs are multicast-compatible"
  cluster:
    shape: [2, 1, 1]            # kNumTMAMulticast=2; heuristics sweep cluster_m/cluster_n in {1,2} (sm90.hpp:91-98)
    multicast: "A or B, selected by kIsTMAMulticastOnA; one TMA load fills both CTAs of the cluster"
  launch:
    threads: 384                # kNumTMAThreads(128) + kNumMathThreads(256), __launch_bounds__(384, 1) at 1d1d.cuh:39
    cta_per_sm: 1               # second arg of __launch_bounds__
    smem_B: 200768              # derived, see checks.smem
    max_regs_per_thread: 255

# ------------------------------------------------------------- 2. mainloop
mainloop:
  axis: K
  step: 128                     # BLOCK_K; forced -- DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling") at 1d1d.cuh:52
  trip_count: "ceil_div(shape_k, 128)"   # 1d1d.cuh:187,261
  tail: "none needed in the tuned shapes (K % 128 == 0); ceil_div would over-read otherwise"
  operands_per_iter:
    - {name: A,   tile: [128, 128], dtype: e4m3, bytes: 16384, src: gmem, via: TMA-2D}
    - {name: B,   tile: [128, 128], dtype: e4m3, bytes: 16384, src: gmem, via: TMA-2D}
    - {name: SFA, tile: [128],      dtype: f32,  bytes: 512,   src: gmem, via: TMA-2D}
    - {name: SFB, tile: [128],      dtype: f32,  bytes: 512,   src: gmem, via: TMA-2D}
  loop_carried: "final_accum[64] in RF per thread (f32). NOT the wgmma accumulator -- see per_iter_math."
  per_iter_math: >
    Drain and promote. The wgmma accumulator `accum` is cleared at the start of every
    stage and holds only that K-block's partial product; after the batch completes it is
    scaled by scale_a * scale_b and added into `final_accum` (1d1d.cuh:313-320). This is
    what "1D1D" means and it is the structural consequence of per-128-channel scaling:
    the scales change every K-block, so the tensor-core accumulator cannot be allowed to
    run across blocks. Cost is a second 64-register accumulator plus 64 CUDA-core FMAs
    per stage, which overlap with the next stage's wgmma.

# ------------------------------------------------------------- 3. pipeline
pipeline:
  depth: 4                      # kNumStages; heuristics require >=3, and >=4 when block_m*block_n < 128*192 (sm90.hpp:134)
  stage_index: "iter_idx % kNumStages"          # get_pipeline(), 1d1d.cuh:165
  phase: "(iter_idx / kNumStages) & 1"          # same lambda; consumer waits on `phase`, producer on `phase ^ 1`
  prologue: "none explicit -- the producer runs ahead freely, bounded only by empty_barriers"
  per_stage_bytes: 33792        # 16384 + 16384 + 512 + align(512,128)
  staged_buffers:
    - {name: smem_a,   shape: [128, 128], dtype: e4m3, bytes: 16384, swizzle: 128B}   # kSwizzleAMode
    - {name: smem_b,   shape: [128, 128], dtype: e4m3, bytes: 16384, swizzle: 128B}
    - {name: smem_sfa, shape: [128],      dtype: f32,  bytes: 512,   swizzle: none}
    - {name: smem_sfb, shape: [128],      dtype: f32,  bytes: 512,   swizzle: "none, padded to 128B for TMA alignment"}
  non_staged_buffers:
    - {name: smem_d, bytes: 65536, aliases: none, alias_safe_because: "n/a -- allocated ahead of the staged region (1d1d.cuh:70)"}
    - {name: barriers, bytes: 64, aliases: none, alias_safe_because: "n/a -- 2 * depth * 8B mbarriers"}
  barriers:
    - {name: full,  kind: mbarrier-tx, count: 4, init_arrive_count: 1,
       produced_by: "TMA elected thread, arrive_and_expect_tx(33792) after issuing the 4 copies (1d1d.cuh:233)",
       waited_by: "both math warp groups, on `phase`"}
    - {name: empty, kind: mbarrier, count: 4, init_arrive_count: 8,
       produced_by: "one arrival per math warp -- kNumTMAMulticast * kNumMathThreads / 32 = 1 * 8 (1d1d.cuh:140)",
       waited_by: "TMA warp, on `phase ^ 1`, before reusing the stage"}
  # Under multicast the empty count becomes 2*8=16 and the producer needs one extra
  # round of empty waits after the last tile so the distributed barriers can be torn
  # down safely (1d1d.cuh:237-246). Classic forgotten-last-iteration case.

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
    issues: "wgmma.mma_async x4 per stage, then CUDA-core promote, then st.shared + TMA store"
    elected: false
  - id: math1
    warps: 4
    threads: 128
    regs: 240
    role: "rows 64-127 of the tile"
    issues: same as math0
    elected: false
inter_group_sync: >
  Only the full/empty mbarrier pair couples producer to math -- there is no direct
  math0<->math1 dependency in the mainloop, because each owns a disjoint 64-row slice
  of the same accumulator tile. NamedBarrier::sync(128, math_wg_idx) appears only in
  the epilogue, to order each group's own st.shared against its TMA store
  (1d1d.cuh:326,337). This is the defining property of producer/consumer
  specialization: the math groups never talk to each other.

# ------------------------------------------------ 5. iters (instruction level)
math:
  - group: "math0 and math1 (identical, disjoint row ranges)"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 128, K: 32}    # FP8MMA in mma/sm90.cuh:26-28; FP8MMASelector<128> -> MMA_64x128x32_F32E4M3E4M3_SS_TN
    dtype: "e4m3 x e4m3 -> f32"
    count_per_stage: 4                    # BLOCK_K / WGMMA::K = 128 / 32 (1d1d.cuh:297)
    a_source: smem-desc                   # make_smem_desc(smem_a + math_wg_idx*64*BLOCK_K + k*32)
    b_source: smem-desc
    acc: {location: RF, elems_per_thread: 64, dtype: f32, cleared: "at iter 0 of every stage"}
    accumulate_across_iters: false
    after_batch: >
      warpgroup_commit_batch() + warpgroup_wait<0>(), then empty_barrier_arrive(stage)
      to release the buffer as early as possible, then the scaled promotion into
      final_accum. Releasing before the promotion is deliberate -- the producer can
      start refilling the stage while the CUDA cores do the 64 FMAs.
  # acc.cleared: WGMMA::wgmma(desc_a, desc_b, accum, k) passes the loop index as
  # `scale_d`, so iter 0 uses ScaleOut::Zero and iters 1-3 accumulate (1d1d.cuh:297-300).
  # kNumAccum = M*N/128 = 64*128/128 = 64 f32 registers per thread.

# ------------------------------------------------------------- 6. epilogue
epilogue:
  position: after-mainloop
  math: "none beyond the per-stage promotion already folded into final_accum"
  path: "rf -> st.shared into smem_d -> tma_store_fence -> SM90_TMA_REDUCE_ADD_2D -> gmem"
  output: {tile: [128, 128], dtype: f32}
  split_reduction: "reduce-add at the TMA store (this fork). Upstream DeepGEMM uses a plain TMA store; the reduce-add variant lets partial results from a split accumulate in place."

# ------------------------------------------------------------- 7. checks
checks:
  smem: "4*33792 + 65536 + 64 = 200768 B vs 232448 B cap -> PASS, 31680 B spare. depth 5 would need 234576 B -> would NOT fit. The 64 KB fp32 epilogue buffer is what caps the depth; a bf16 D would free 32 KB and buy the fifth stage."
  threads: "128 + 128 + 128 = 384 == __launch_bounds__ first arg -> PASS. All groups are 128-thread multiples, required for wgmma."
  acc_registers: "128*128/256 = 64 f32 accumulator elems/thread == WGMMA::kNumAccum -> PASS. But the drain-and-promote pattern needs a SECOND 64-register array (final_accum) plus 16 float2 of B scales: ~160 registers of live state before addressing. This is why the math groups take 240."
  register_budget: "128*24 + 256*240 = 64512 <= 65536 per SM -> PASS with 1024 spare. (The KGroupedContiguous variant uses 40/232 and lands on exactly 64512 too -- the split is chosen to saturate the SM either way.) All four values are multiples of 8, as setmaxnreg requires."
  mma_k: "4 iters * 32 == 128 == mainloop.step -> PASS"
  mma_m: "64 * 2 math groups == 128 == cta_tile.M -> PASS. Enforced in source: DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2))."
  mma_n_legal: "N=128 is a legal wgmma atom (multiple of 8, <= 256) -> PASS"
  trip_count: "ceil_div(K,128)*128 >= K -> PASS"
  output_coverage: "scheduler enumerates every (m_block, n_block) exactly once -> PASS"
  occupancy: "smem 200768 > 232448/2, registers 64512 > 65536/2 -> 1 CTA/SM, matches __launch_bounds__(_, 1) -> PASS"
  barrier_arrivals: "full=1 (single elected TMA thread) -> PASS. empty=8 = one per math warp; becomes 16 under 2-CTA multicast -> PASS. Phase rule stated -> PASS."
  arithmetic_intensity: >
    Per CTA tile: 2*128*128*K FLOP over (128+128)*K bytes = 128 FLOP/byte. H100 SXM5
    fp8 dense ridge is ~1979e12 / 3.35e12 = 591 FLOP/byte, so the tile is ~4.6x below
    ridge and CANNOT saturate the fp8 tensor cores from DRAM on its own -> FLAG.
    This is not a defect, it is why two other fields exist: cluster multicast halves one
    operand's DRAM traffic, and the scheduler's rasterization order makes neighbouring
    CTAs hit the same operand tile in L2. Any change to `cluster` or `rasterization`
    must be re-justified against this number.

# ------------------------------------------------------------- 8. handover
verification:
  reference: "torch fp32 matmul of the dequantized inputs"
  tolerance: "relative error consistent with e4m3 rounding; DeepGEMM's tests/ use per-element relative tolerance"
  perf_target: "TFLOP/s vs cuBLAS fp8 at the same shapes"
open_questions: []
deviations: []
```

## Loop nest

### L1 — iteration space

```
  gemm(A[M,K] e4m3 k-major, B[N,K] e4m3 k-major,
       SFA[M, K/128] f32, SFB[N, K/128] f32) -> D[M,N] f32

  # M is dynamic (SHAPE_M==0 -> runtime shape_m); N and K are baked in by the JIT.
  for m0 in range(0, M, 128):                  # ceil(M/128) tiles     parallel
    for n0 in range(0, N, 128):                # ceil(N/128) tiles     parallel
      for k0 in range(0, K, 128):              # ceil(K/128) steps     SERIAL, contraction axis
        D[m0:m0+128, n0:n0+128] += SFA[m0:m0+128, k0//128] * SFB[n0:n0+128, k0//128]
                                   * ( A[m0:m0+128, k0:k0+128] @ Bᵀ[k0:k0+128, n0:n0+128] )
                                       (128,128) @ (128,128) -> (128,128)

  # step 128 on k0 is forced, not chosen: DG_STATIC_ASSERT(BLOCK_K == 128) because the
  # scale granularity is per-128-channel (1d1d.cuh:52). K % 128 == 0 in the tuned shapes,
  # so there is no k tail.
```

### L2 — mapped to hardware

```
  grid: 132 persistent CTAs. The (m0, n0) loops are flattened and walked by
  sched::Scheduler, which also groups tiles so a cluster's two CTAs share an operand
  and so neighbours hit the same tile in L2. k0 stays serial inside each CTA.

  384 threads = 1 producer WG (128) + 2 math WGs (128 each).
  math WG w in {0,1} owns rows [64w, 64w+64) of the 128x128 tile.

  for (m0, n0) in scheduler(cta_id):           # while(get_next_block(...)), 1d1d.cuh:179,256
    D_acc[64, 128] = 0                         # final_accum, f32 RF, 64 elems/thread, carried
    n_k = ceil_div(K, 128)                      # 1d1d.cuh:187,261

    for k0 in range(0, 128*n_k, 128):          # mainloop: start 0, stop K, step 128, trip n_k
      s, phase = (k0//128) % 4, (k0//128)//4 & 1        # depth 4, 1d1d.cuh:165

      producer  (1 elected thread of 128, 24 regs)
                wait empty[s] @ phase^1
                A_s  [s][128,128] <- A  [m0:m0+128, k0:k0+128]     16384 B  TMA, 128B swizzle
                B_s  [s][128,128] <- B  [n0:n0+128, k0:k0+128]     16384 B  TMA, 128B swizzle
                sfa_s[s][128]     <- SFA[m0:m0+128, k0//128]         512 B  TMA
                sfb_s[s][128]     <- SFB[n0:n0+128, k0//128]         512 B  TMA
                full[s].arrive_and_expect_tx(33792)                # 1d1d.cuh:233
                # under 2-CTA multicast one of A_s / B_s is filled for both CTAs at once

      math WG w (128 threads, 240 regs)
                wait full[s] @ phase
                scale_a[64] = sfa_s[s][64w:64w+64]                 # read BEFORE warpgroup_arrive
                scale_b[128] = sfb_s[s][0:128]
                C_s[64,128] = A_s[s][64w:64w+64, 0:128] @ B_s[s]ᵀ[0:128, 0:128]
                                (64,128) @ (128,128) -> (64,128)   f32 RF, stage-local
                empty[s].arrive()                                  # release early, 1d1d.cuh:309
                D_acc += scale_a * scale_b * C_s                   # 64 FMAs, 1d1d.cuh:313-320

    epilogue  (per math WG)
              D_acc -> smem_d[64w:64w+64, 0:128] via st.shared     # 65536 B buffer, not staged
              tma_store_fence; NamedBarrier::sync(128, w)          # 1d1d.cuh:326,337
              D[m0+64w : m0+64w+64, n0:n0+128] <- SM90_TMA_REDUCE_ADD_2D(smem_d)
```

### L3 — innermost body, expanded

```
      for ki in range(0, 128, 32):             # iter: start 0, stop BLOCK_K=128, step 32, trip 4
        wgmma.m64n128k32(                      # FP8MMASelector<128> -> MMA_64x128x32_F32E4M3E4M3_SS_TN
          A = A_s[s][64w:64w+64, ki:ki+32]     smem-desc, k-major
          B = B_s[s][ki:ki+32, 0:128]ᵀ         smem-desc, k-major
          C = C_s[64, 128]                     f32 RF, kNumAccum = 64*128/128 = 64 elems/thread
          clear = (ki == 0) )                  # loop index passed as scale_d: iter 0 uses
                                               # ScaleOut::Zero, iters 1-3 accumulate (1d1d.cuh:297-300)
      warpgroup_commit_batch(); warpgroup_wait<0>()
```

The `=` on `C_s` beside the `+=` on `D_acc` is the entire 1D1D story: the
tensor-core accumulator is stage-local because the scales change every K block,
so a second fp32 accumulator carries the mainloop. Both live in RF — 64 + 64
registers per thread before operands — which is why the math groups need 240.

## Schedule

Deleted: the warp split is plain producer/consumer and the two math groups never
talk to each other, so the L2 nest already shows every ordering that exists. The
only cross-group synchronisation is the full/empty mbarrier pair, which is on the
`wait` / `arrive` lines above. Compare `example-flashmla.md`, where this section
carries the algorithm.

## Why these numbers

**depth 4, not 5.** The 64 KB fp32 epilogue buffer is a fixed cost that does not
scale with depth, so it eats two stages' worth of smem before the pipeline gets
any. 4 × 33792 + 65536 = 200768 fits; 5 stages needs 234576 and does not. The
heuristic's floor of 4 stages for tiles under 128×192 (`sm90.hpp:134`) and the
smem ceiling of 4 meet exactly here — the config is on the boundary in both
directions, which is why it is the tuned one.

**Two math warp groups, not one.** wgmma's M is fixed at 64. A 128-row tile
therefore needs two warp groups, and the source asserts it. This is also what
keeps the accumulator at 64 registers per thread instead of 128 — a single warp
group on a 128×128 tile would need 128 accumulator registers plus 128 more for
`final_accum`, and would spill.

**Drain-and-promote instead of a running wgmma accumulator.** Per-128-channel
scales change every K-block, so the products from different K-blocks cannot be
summed in the tensor-core accumulator. The extra cost — one more 64-register
array and 64 FMAs per stage — is deliberately placed after the empty-barrier
arrival so the CUDA-core work overlaps the producer's next TMA and the next
stage's wgmma.

**24 / 240 register split.** The producer does nothing but compute addresses and
issue TMA from one elected thread, so it is starved down to 24 registers to fund
the math groups at 240. The totals land on 64512 of 65536 — the split is sized
to consume the SM's register file almost exactly at 1 CTA/SM.

## Known risks

- **smem is on the boundary.** Any added buffer forces depth 3, which the
  heuristics reject. Changing D to bf16 is the lever that buys headroom.
- **Register pressure is the reason for `#pragma unroll kNumPipelineUnrolls`.**
  The `KGroupedContiguous` variant disables unrolling (`kNumPipelineUnrolls = 0`)
  and drops the math groups to 232 registers, because unrolling the pipeline
  costs live registers. A spec that changes the unroll factor must revisit
  `checks.register_budget`.
- **The tile is below the fp8 ridge point.** Performance depends on multicast and
  L2 rasterization, not on the mainloop alone. Profile L2 hit rate, not just
  tensor-core utilization.
- **Barrier teardown under multicast.** The extra round of empty waits after the
  last tile is required, not defensive.
