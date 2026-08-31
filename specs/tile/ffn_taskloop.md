---
spec_version: 1
kernel: ffn_taskloop
status: review                   # rev 2: activation transpose + BK restored to 128
approved_by: revollllt (rev 1, in conversation, "没问题，我们先完成当前 ffn 阶段")
approved_at: 2026-08-24          # rev 1 only; rev 2 awaits sign-off

# ------------------------------------------------- 0a. toolchain (Phase 0)
toolchain:
  device: H100 SXM5, acd_u cluster (see server-usage)
  compiler: "nvcc 13.1 -arch=sm_90a (module cuda/13.1, gcc/13.3); harness venv
    torch 2.13.0+cu130 -- NOTE: PLAN 4.9's 23.35 us composition figure predates
    this venv (torch 2.11), so the acceptance harness re-measures the
    composition in the same process rather than citing it"
  harness: standalone layer-step bench, same process, cold, CUDA-graph timing (benchmark-kernel)
  clocks: unpinnable; ~6% noise floor -- comparisons side by side in one process only
  measured:
    launch_floor: "see machine-limits tag launch.lat.dev.ramp (grid ramp, per launch,
      not removed by a CUDA graph)"
    bandwidth_fit: "see machine-limits tags launch.lat.dev.ramp + ld.bw.dev.dram; the fitted
      form `t_us = ramp + MB / ld.bw.dev.dram` is what every floor below uses"
    bandwidth_vs_ctas: "see machine-limits tag ld.ctas.dev.knee. NOTE this kernel is
      transaction-bound, not bandwidth-bound, on the DR side -- so ld.ctas.dev.knee is
      the wrong curve to reason from there; tma.issue.warp is."
    best_known_impl: "the composition this kernel replaces: tl_ada_scaled_gate 14.25 us
      + tl_matmul_gated_res(ffn_down) 9.10 us = 23.35 us, isolated cold (PLAN.md 4.9)"
    barrier_costs: "cluster barriers unused here. gmem counter RTT (fence.release
      + red.add -> ld.acquire poll), MEASURED 2026-08-24 job 541290, 40 concurrent
      CTA pairs x 200 iters via %globaltimer: median 640 ns, p95 736, max 960.
      Far under the 2 us rework threshold -- the DR unlock schedule stands, and
      the counter [I] figures in L3 are retired to [D]"
    tma_delivery: "CITED, not restated: machine-limits/constants/
      sm90-h100-sxm5.yaml is the source of truth (probes/tma_ring.{cu,py},
      jobs 545906 cold-DRAM / 546955 L2-resident). A number in two places
      drifts, so this spec names tags and consequences only.
      tma.issue.warp   -- one producer warp sustains ~1 TMA / 270 ns, near-
                     independent of frame size. A copy column costs
                     `txns_per_warp x 270 ns` with `txns_per_warp = K_per_CTA
                     / BK`, so BYTES PER TMA IS A FIRST-CLASS TERM here.
      tma.stages.warp.knee   -- ring depth saturates at 4; retires rev 1's depth 2.
      tma.bw.cta.geom    -- box stride is ~free, so the pre-blocking argument in
                     problem.layouts is about FRAME SIZE, not coalescing.
      tma.bytes.txn.max -- boxDim[0] x elem <= swizzle width; a bigger frame comes
                     from more box rows (<= 256), never a longer run. THIS is
                     what caps the activation descriptors and drives rev 2.
      tma.bw.dev.l2      -- the 270 ns is source-independent, so the extra L2 traffic
                     an M-split would cause is nearly free -- and, conversely,
                     MORE CTAs DOES NOT MOVE A COPY FLOOR (every CTA still
                     walks the full K).
      tma.bw.cta.warps   -- the 270 ns is PER WARP, so splitting a ring across more
                     producer warps is one of only three levers that divides a
                     copy floor (the others: bigger BK, split-K).
      ld.bw.dev.dram     -- 2.77 TB/s, the other half of
                     `max(txns_per_warp x 270 ns, bytes / ld.bw.dev.dram)`.
      Floors below are computed with machine-limits/scripts/frontier.py."
  measured_0b:
    occupancy: "1 CTA/SM at the rev-2 budget (132,224 B smem), and that is the
      intended operating point: the grid is 132 = 1 x SM count, so extra
      residency is unreachable anyway (h100-cluster-placement-limits). ncu on
      the shipped BK=64 build confirms the pattern -- theoretical 3 CTAs/SM at
      66 KB, achieved 7.7%, because 132 CTAs is 0.33 waves."
  job_ids: "PLAN.md 4.7-4.9 (e2e, profile, roofline); example-phase0.md job set"

# ---------------------------------------------------------------- 0. problem
arch: sm90a
problem:
  op: >
    One persistent task-loop kernel executing a static task table with two task
    types, fusing the decoder FFN chain.
    GU (n-tile of gated FFN):  hidden[0:50, n:n+32] =
      gelu_tanh(((x*F*S) @ W_gate)[:, n:n+32] + b1[n:n+32])
      * (((x*F*S) @ W_up)[:, n:n+32] + b2[n:n+32]).
    DR (n-tile of down+residual):  out[0:50, n:n+32] =
      x_res[0:50, n:n+32] + (hidden @ W_down)[:, n:n+32] * g[n:n+32].
    F (per-row rstd) is an INPUT -- rms stays outside this kernel (control
    variables; it joins as a task type in the full-layer extension).
  dims: {M: 50, D: 1024, FF: 4096}
  dynamic: []                    # everything static; task table baked offline
  dtypes: {a: bf16, b: bf16, acc: f32, d: bf16}
  layouts:
    # rev 2: BOTH ACTIVATIONS ARE STORED TRANSPOSED, M-major. Rationale is
    # toolchain.measured.tma_descriptor_legality: a TMA box row is capped at the
    # swizzle width (128 B = 64 bf16), so frame size can only grow through
    # boxDim[1] (<= 256). Row-major x/hidden put the SHORT axis (M_PAD=64) in
    # the boxDim[1] position, capping every activation frame at 64 x 64 x 2 =
    # 8192 B no matter what BK says. Transposed, boxDim[0] = M_PAD = 64 elems =
    # exactly 128 B and boxDim[1] = BK is free to 256 -- frames to 32 KB, and
    # box rows become ADJACENT (stride 128 B == run 128 B), i.e. one contiguous
    # run, the same shape the pre-blocked weights already have.
    x: "col(D,M_PAD) = x^T, M-major, M_PAD=64 (pad rows 50..63 zeroed).
      Host input: the planner transposes offline, free."
    w_gate: "row(D,FF) k-major, PRE-BLOCKED offline to (FF/32, D, 32) and
      interleaved with w_up as [W1_tile(32) | W2_tile(32)] -- one 128 B box row"
    w_up: "row(D,FF) k-major, interleaved into w_gate's slab (above)"
    w_down: "row(FF,D) k-major, PRE-BLOCKED offline to (D/32, FF, 32), 64 B row (SW64)"
    hidden: "col(FF,M_PAD) = hidden^T, M-major, gmem, zero-init. WRITTEN
      transposed by GU's epilogue and READ transposed by DR's A_h ring, so the
      cap above is lifted on both sides. Cost is GU's store: the wgmma C
      fragment holds (row, col) and (row, col+1) adjacent, which are 64 elems
      apart in hidden^T, so the bf16x2 vector store degrades to scalar strided.
      Priced and accepted: hidden is 512 KB, so even 4x the store sectors is
      ~2 MB against a 25 MB read side."
    out: "row(M,D), padded to 64 -- NOT transposed. DR's acc is (64 tokens,
      32 D-cols) and its epilogue writes out[row][n+col] coalesced; leaving out
      row-major keeps that epilogue and the residual aliasing unchanged."
  regime: >
    latency, M=50 decode; memory-bound by construction (intensity ~47 FLOP/B
    vs ridge 295). Prototype scope: validate the task-loop framework
    (static schedule + counters + ring continuity + decoupled prefetch) on the
    smallest closed op chain, per owner decision 2026-08-24. Growth path:
    full decoder layer, then 10x18 layer-steps in one launch.

# ------------------------------------------------------------- 1. grid / CTA
grid:
  mode: persistent               # task-loop: each CTA walks a static task list
  ctas: "132 fixed worker grid, one CTA/SM. 160 tasks (128 GU + 32 DR) are
    dealt as: CTA c in [0,128) takes GU tile c in slot 0; CTA d in [0,32) takes
    DR tile d in slot 1, gated on its counters; CTAs 128..131 idle (sentinel
    row, type=-1). rev 1 specced 80; the worker-queue revision measured 2.2x
    better GU (see deviations). NOTE per toolchain.tma_delivery(3): CTA count
    does NOT move a copy floor -- every CTA still walks the full K -- so this is
    an occupancy/bandwidth-headroom choice, not a latency lever."
  shape: "(cta_id,) -> planner task list; no blockIdx arithmetic beyond the table"
  cta_tile: {M: 64, N: 32}       # per task; M pads 50 (tail: predication on stores... see mainloop.tail)
  rasterization: >
    Static task table (planner v0). GU CTA c in [0,64) owns tasks {n=c, n=64+c},
    in that order: the first wave completes hidden columns [0:2048) before the
    second wave starts, so DR's counters unlock left to right and DR's gated
    ring starts draining while GU is still streaming W slabs. DR CTA d in
    [0,16) owns tasks {n=2d, n=2d+1}. Weight-slab read order = task order =
    sequential over W_gate/W_up, then W_down interleaved from t=0 by DR's
    dep-free ring. L2 argument: x (0.1 MB) and hidden (0.5 MB) are L2-resident;
    weights are streamed once, no reuse to schedule for.
  l2_schedule: "solved: task order chosen for counter-unlock order (above), not
    defaulted. The planner owns this table; it is index arithmetic, no atomics."
  persistence:
    cta_per_sm: 1                # realized: 80 CTAs on 132 SMs. Capacity is 3 (smem 66.5 KB)
    grid_realises_it: "no -- 80 < 132 by construction; prototype is task-bound.
      Named, accepted: the mechanism under test is counters+ring, not machine fill."
    scheduler: "static per-CTA task list from planner v0; a CTA reads its next
      task descriptor (type, n, buffer offsets, counter ids) from a const table"
    phase_ordering: "32 gmem u32 counters, one per 128-col slice of hidden.
      GU release: st.global of its C tile -> fence (release semantics) ->
      red.global.add(counter[n//4], 1). DR acquire: producer warp polls
      ld.global.acquire until counter[s] >= 4 before issuing the A_h TMA for
      k-slice s. Host zeroes counters before launch (standalone harness);
      graph-replay self-reset is a full-decoder-scope item, recorded in risks."
  cooperative: "false -- BUT this design has cross-CTA waits (DR spins on GU),
    so co-residency is a CORRECTNESS requirement, satisfied here because
    grid 80 < 132 SMs at 66.5 KB (occupancy 3/SM). Any grid growth must re-check
    grid <= SMs x occupancy or the wait is a hang. (residency.md's rule, applied.)"
  # cluster: deleted -- dependencies span arbitrary CTAs (GU->DR counters), and
  # the shared-A multicast idea is a v2 axis; nothing here needs DSMEM.
  launch:
    threads: 160                 # 1 producer warp + 1 math warp group
    cta_per_sm: 1
    smem_B: 132224               # rev 2: 4 x 33,024 GU frames + 128 B barriers.
                                 # DR pool aliases inside (98,304 of 132,096).
                                 # 1 CTA/SM, which is the operating point anyway
                                 # at a 132-CTA grid; 100,224 B of the 232,448 B
                                 # cap stay spare, and BK=256 for DR (frames
                                 # 32 KB + 16 KB, 196,608 B) would still fit --
                                 # but it needs COUNTER_K 128 -> 256, so it is a
                                 # separate rev, not part of this one.
    max_regs_per_thread: 240     # no setmaxnreg: lone producer warp makes it CTA-illegal (residency.md)

# ------------------------------------------------------------- 2. mainloop
# Primary task type GU; DR mirrors it with its own extents under `taskgraph`.
mainloop:
  axis: D                        # per GU task; DR contracts FF -- see taskgraph.dr
  step: 128
  trip_count: 8                  # 1024 / 128
  tail: "none-needed (D % 128 == 0; FF % 128 == 0). M tail: tile M=64 covers
    M=50 with padded, zeroed rows 50..63; stores write the padded buffers and
    rows >= 50 are never read downstream"
  operands_per_iter:
    - {name: A_s,  tile: [64, 128], dtype: bf16, bytes: 16384, src: gmem, via: TMA-2D}
    - {name: W1_s, tile: [128, 32], dtype: bf16, bytes: 8192,  src: gmem, via: TMA-2D}
    - {name: W2_s, tile: [128, 32], dtype: bf16, bytes: 8192,  src: gmem, via: TMA-2D}
    - {name: S_s,  tile: [128],     dtype: bf16, bytes: 256,   src: gmem, via: TMA-1D}
  loop_carried: "C1, C2 (f32 RF, per task, cleared at task start); across tasks:
    the task cursor and the ring state (frames do NOT drain at task switch)"
  per_iter_math: "a_scale (non_mma) -- the one computation between a frame
    landing and its wgmma"

# ------------------------------------------------------------- 3. pipeline
pipeline:
  depth: 4                       # rev 2, MEASURED. rev 1's 2 priced fills at
                                 # bandwidth and was wrong; job 545906 sweeps
                                 # depth 2/4/8/16 at 770/972/967/956 GB/s --
                                 # depth 2 is latency-bound (2 x 340 ns ~ the
                                 # ~680 ns TMA latency), 4 covers it, and past 4
                                 # the per-warp 270 ns issue rate binds instead.
                                 # 4 is the saturation point, not a guess.
  stage_index: "stage = iter % 4, CONTINUOUS ACROSS TASKS: at a task switch the
    producer issues the next task's iter-0 frame into the frame the last task
    just released -- the ring never drains. This is the mechanism under test."
  phase: "(global_iter / 4) & 1, where global_iter counts across the CTA's whole
    task list (8 per GU task, 32 per DR task at BK=128)"
  prologue: "producer fills the ring before math starts; 4 stages deep,
    ~132 KB in flight per CTA at t=0"
  per_stage_bytes: 33024
  staged_buffers:
    - {name: A_s,  shape: [64, 128], dtype: bf16, bytes: 16384, swizzle: 128}
    - {name: W1_s, shape: [128, 32], dtype: bf16, bytes: 8192,  swizzle: 64}
    - {name: W2_s, shape: [128, 32], dtype: bf16, bytes: 8192,  swizzle: 64}
    - {name: S_s,  shape: [128],     dtype: bf16, bytes: 256,   swizzle: none}
  non_staged_buffers:
    - {name: barriers, bytes: 128, swizzle: none, aliases: none,
       alias_safe_because: n/a}   # GU 4 + DR 12 mbarriers x 8 B, rounded up
    - {name: dr_pool, bytes: 0, swizzle: "64 (W_d) / 128 (A_h)",
       aliases: "the GU staged frames -- at BK=128/depth 4 the W_d ring is
       4 x 8192 = 32768 B and the A_h ring 4 x 16384 = 65536 B, so 98304 B
       inside the 132096 B GU staged pool (4 x 33024), one union allocation",
       alias_safe_because: "per-CTA task lists are single-type in planner v0, so
       a CTA's pool is only ever laid out one way; bytes: 0 because the storage
       belongs to the staged frames"}
  barriers:
    - {name: full,  kind: mbarrier-tx, count: 2, init_arrive_count: 1,
       produced_by: "producer elected thread, arrive_and_expect_tx(33024) (GU)
       / split tx for W_d and A_h sub-frames (DR, see taskgraph)",
       waited_by: "math WG @ phase"}
    - {name: empty, kind: mbarrier, count: 2, init_arrive_count: 1,
       produced_by: "one elected math lane, after wgmma.wait_group retires the
       last batch reading the frame -- NOT at logical consumption (schedule-l3.md)",
       waited_by: "producer @ phase^1"}

# ------------------------------------------- 4. warp specialization / roles
warp_groups:
  - id: producer
    warps: 1
    threads: 32
    regs: 240                    # NOT reconfigured -- setmaxnreg needs whole warp
                                 # groups and a lone warp forfeits it for the CTA
                                 # (residency.md); 240 is the uniform launch-bound cap
    role: "task fetch, counter acquire (DR), TMA issue, tx arrive"
    issues: "ld const task descriptor; ld.global.acquire counter poll (DR);
      cp.async.bulk.tensor 2D/1D; mbarrier.arrive.expect_tx"
    elected: true
  - id: math0
    warps: 4
    threads: 128
    regs: 240                    # uniform, see producer note
    role: "a_scale, wgmma, epilogue, counter release"
    issues: "ld.shared/st.shared b128 (a_scale); wgmma.mma_async m64n32k16;
      st.global.b32 epilogue; red.release.global.add (GU)"
    elected: false
inter_group_sync: "full/empty mbarrier pairs per frame (above). No named
  barriers between math groups -- there is only one. Cross-CTA sync is the gmem
  counter protocol in grid.persistence.phase_ordering, deliberately NOT a
  barrier: producers never wait on consumers."

# ------------------------------------------------ 5. iters (instruction level)
math:
  - group: math0
    stage_phase: "GU stage, first batch"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 32, K: 16}
    contracts: D=128             # BLOCK_K slice, and D IS mainloop.axis -- the plain case
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 8           # 128 / 16
    a_source: smem-desc          # A_s AFTER a_scale rewrote it in place
    a_major: MN                  # rev 2: A_s is x^T, so M is the contiguous
                                 # smem mode. SmemLayoutA moves from
                                 # Layout_K_SW128_Atom to Layout_MN_SW128_Atom
                                 # and MmaAtom from SS<Major::K, Major::MN> to
                                 # SS<Major::MN, Major::MN>. CuTe supports an
                                 # MN-major A operand (mma_sm90_gmma.hpp:134,
                                 # `GMMA::Major tnspA` is a template parameter).
    b_major: MN                  # unchanged
    b_source: smem-desc
    acc: {name: C1, location: RF, elems_per_thread: 16, dtype: f32, cleared: "ScaleOut::Zero at task iter 0"}
    accumulate_across_iters: "yes, within one task; task-local, never across tasks"
    after_batch: "commit with the W2 batch below; wait_group<1> keeps one stage's
      pair in flight; empty[s-1].arrive fires on retirement, BEFORE the epilogue"
  - group: math0
    stage_phase: "GU stage, second batch, same A_s frame"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 32, K: 16}
    contracts: D=128             # same slice; the second batch reads the same A_s fill
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 8
    a_source: smem-desc
    b_source: smem-desc
    acc: {name: C2, location: RF, elems_per_thread: 16, dtype: f32, cleared: "ScaleOut::Zero at task iter 0"}
    accumulate_across_iters: "yes, within one task"
    after_batch: "shared commit/wait with C1's batch -- the two GEMMs read one
      A_s fill once, which is the reason W1 and W2 share a frame"
  - group: math0
    stage_phase: "DR stage (DR-CTAs only)"
    unit: wgmma.mma_async
    inst_shape: {M: 64, N: 32, K: 16}
    contracts: FF=128            # BLOCK_K slice of FF, the DR task's mainloop axis (taskgraph.dr)
    dtype: "bf16 x bf16 -> f32"
    count_per_stage: 8
    a_source: smem-desc          # A_h (hidden slice), counter-gated
    b_source: smem-desc          # W_d, dep-free ring
    acc: {name: C_dr, location: RF, elems_per_thread: 16, dtype: f32, cleared: "ScaleOut::Zero at task iter 0"}
    accumulate_across_iters: "yes, within one task (trip 32)"
    after_batch: "wait_group<1>; empty_a[s-1] and empty_w[s-1] arrive on retirement"

# ------------------------------- 5b. non-MMA work (the CUDA-core column of L3)
non_mma:
  - id: a_scale
    primitive: none
    where: mainloop.per_iter
    kind: elementwise
    over: "A_s tile, 64 x 128"
    span: lane
    mechanism: "ld.shared.b128 / HMUL2 / st.shared.b128 in place, 8 inst x vec 8 per thread"
    loop_carried: []
    dtype: "bf16 (F*S product and the A multiply both bf16 -- matches
      tl_ada_scaled_gate bit-for-bit; rounding lands in bf16 BEFORE the f32 GEMM)"
    cost: "64 elems/thread: 8 ld.shared.b128 + 64 HMUL2 + 8 st.shared.b128, math0"
    touches: "A_s (smem, swizzle 128, RW -- same buffer wgmma reads via desc;
      needs wgmma fence, and the bank check is the l4 row that matters)"
    on_critical_path: "yes -- sits between full[s] and the stage's wgmma. L3
      shows the copy engine filling frame s^1 during it; with depth 2 that is
      the entire overlap story"
  - id: gelu_gate
    primitive: none
    where: epilogue
    kind: elementwise
    over: "C1/C2 fragments, 64 x 32 per task"
    span: lane
    mechanism: "f32 FADD bias x2, gelu_tanh f32, FMUL gate, cvt to bf16"
    loop_carried: []
    dtype: "f32 throughout, cast bf16 at store (matches today's epilogue)"
    cost: "~16 FADD x2 + 16 gelu_tanh (~8 f32 ops each) + 16 FMUL + 16 cvt per thread, math0"
    touches: "RF only; st.global.b32 out (l4 row gu_c_store)"
    on_critical_path: "no for the copy engine -- producer is already issuing the
      next task's frames; yes for the counter chain: DR's unlock waits on it"
  - id: counter_release
    primitive: none
    where: epilogue
    kind: elementwise
    over: "1 atomic per GU task"
    span: cta
    mechanism: "fence.release.gpu then red.global.add(counter[n//4], 1), one elected lane"
    loop_carried: []
    dtype: u32
    cost: "1 fence + 1 red per task, math0 elected lane; latency [I] until the probe"
    touches: "counters[32] u32 gmem"
    on_critical_path: "no for GU; it IS DR's critical input"
  - id: counter_acquire
    primitive: none
    where: mainloop.per_iter
    kind: elementwise
    over: "1 poll loop per DR k-slice"
    span: warp
    mechanism: "producer elected lane polls ld.global.acquire counter[s] >= 4;
      backoff nanosleep"
    loop_carried: []
    dtype: u32
    cost: "[I] ~0.5-1 us worst first-slice wait, then amortized zero once GU runs
      ahead; UNMEASURED -- the probe task pins it, and it is this spec's biggest
      unknown"
    touches: "counters[32] u32 gmem (read)"
    on_critical_path: "yes for DR iter 0..3; the schedule exists to make it not
      so afterwards (GU wave 1 completes columns [0:2048) first)"
  - id: dr_gate_res
    primitive: none
    where: epilogue
    kind: elementwise
    over: "C_dr fragment, 64 x 32 per task"
    span: lane
    mechanism: "f32 FMUL gate + f32 FADD residual (R read bf16 -> f32), cvt bf16"
    loop_carried: []
    dtype: "f32 gate and residual add, bf16 store (matches tl_matmul_gated_res)"
    cost: "16 FMUL + 16 FADD + 16 cvt + 16 ld.global.b32 (R) per thread, math0"
    touches: "out gmem RW rows (l4 rows dr_r_load / dr_c_store)"
    on_critical_path: "no"

# ------------------------------------------------------------- 6. epilogue
epilogue:
  position: "per task (fused-per-iter at task granularity)"
  math: "GU: +b1/+b2 f32, gelu_tanh f32, gate multiply f32, cast bf16.
    DR: gate multiply f32, residual add f32, cast bf16"
  path: "rf -> st.global.b32 (no smem staging; matches today's kernels, and the
    64x32 f32->bf16 fragment store is already 1x sectors -- see L4)"
  output: {tile: "GU: hidden[0:64, n:n+32]; DR: out[0:64, n:n+32]", dtype: bf16}
  split_reduction: none

# --------------------------------------- 5c. taskgraph (schema extension)
# One kernel, many op instances: the fields above describe the primary task
# type; this block carries the rest. See spec-schema.md "Task-graph extension".
taskgraph:
  queue: "static per-CTA lists, const memory; descriptor = {type, n, counter_id}"
  dispatch: "branch-once-per-CTA: planner v0 lists are single-type, so the
    kernel reads the first descriptor and enters one of two monomorphic loops.
    Each op body is a __forceinline__ device function with one calling
    convention: body(desc, smem_pool, ring_state, counters). Register file is
    the UNION of all reachable bodies regardless of per-CTA typing (allocation
    is per-kernel, compile-time): here max(GU ~70, DR ~55) -- benign. The union
    becomes binding when fd_split joins (acc_o alone is 128 f32/thread), and
    setmaxnreg cannot help across op types, only across warp roles -- named in
    Known risks. Growth path for mixed-type lists: page-granular smem pool
    (HazyResearch-style acquire/release) so ring continuity survives a type
    switch; and 6 of the 8 decoder task types are one GEMM body + epilogue
    functor + ring policy (data-driven within the family, opcode only across
    the 3 families gemm/attention/reduce), which is what keeps the union small."
  planner: "hardware/nvidia/h100/pi05/planner (v0: hand-enumerated, emits the
    table + counter init + TRUNCATED lists for bisection -- persistent-kernel
    bugs hang, so run-to-task-k + compare is the debug loop from day 1)"
  task_types:
    - id: GU
      count: 128                 # FF / 32
      op: "gated FFN n-tile (problem.op)"
      mainloop: {axis: D, step: 128, trip_count: 8}
      frame_bytes: 33024
      ring: {depth: 2, gated_by: "nothing -- x and W are dep-free inputs"}
      releases: "counter[n // 4] += 1 (4 GU tasks fill one 128-col hidden slice)"
    - id: DR
      count: 32                  # D / 32
      op: "down + gated residual n-tile (problem.op)"
      mainloop: {axis: FF, step: 128, trip_count: 32}
      frame_bytes: "8192 (W_d) + 16384 (A_h), SPLIT RINGS"
      ring: {depth: "W_d: 4 (dep-free -- weights run ahead; the decoupled
        prefetch this whole design exists to express), A_h: 2 (gated by
        counter[s] >= 4 per k-slice s)"}
      releases: nothing
  # rev 2: DERIVED, NOT HARDCODED, and owned by ONE side. rev 1 shipped
  # `COUNTER_K = 128` as a kernel constexpr (ffn_taskloop.cu:55) AND `N_COUNTERS
  # = 32` / `COUNTER_ARRIVE = 4` as planner literals (taskloop.py:47-48) -- two
  # independent copies of one protocol, so any BK/BN retune desynchronised them
  # silently. That is the schedule leaking into the interpreter, which
  # megakernel-taskgraph principle 1 exists to prevent, and its planner contract
  # already names counter count/init as a PLANNER OUTPUT.
  counters:
    granularity_K: "COUNTER_K = tunable, constrained to lcm-compatible values:
      COUNTER_K % BK == 0 (a DR stage maps to exactly one counter) and
      COUNTER_K % BN == 0 (a whole number of GU tiles fills a slice) and
      FF % COUNTER_K == 0. At BK=128/BN=32 the admissible set is {128, 256,
      512, ...}; rev 2 keeps 128 so COUNTER_K/BK == 1 and the DR poll maps
      1:1 onto stages."
    count: "FF / COUNTER_K            # 32 at COUNTER_K=128"
    arrive_each: "COUNTER_K / BN      # 4 at COUNTER_K=128, BN=32"
    stages_per_counter: "COUNTER_K / BK  # 1 at BK=128; the DR poll's divisor"
    dtype: u32
    owner: "the planner computes all four from the tile shape and emits them
      with the task table; the kernel takes arrive_each and stages_per_counter
      as LAUNCH ARGUMENTS and keeps no constexpr copy. A second definition of
      any of these is the defect to grep for."
    reset: "host memset before launch (standalone); self-reset under graph
      replay is full-decoder scope"
  schedule: "grid.rasterization; single-type per CTA in v0"

# ------------------------------------------------------------- 7. checks
l4_accesses: accesses-ffn-taskloop.yaml
checks:                          # budget.py --sms 132: PASS 15, MANUAL 8, FAIL 0, SKIP 0
  smem: "66176 B = 2 x 33024 + 128 barriers (dr_pool 65536 aliases inside) /
    232448 B cap, 166272 B spare -- PASS; capacity 3 CTA/SM"
  threads: "32 + 128 = 160 == launch 160 -- PASS"
  acc_registers: "16 f32/thread per acc (64x32/128); GU carries C1+C2 = 32 -- PASS"
  register_budget: "160 x 240 x 1 = 38400 <= 65536 -- PASS. Full-decoder scope at
    2 CTA/SM inherits a 200-reg cap (65536/320, granularity 8) -- Known risks"
  mma_k: "8 x 16 = 128 == D=128 / FF=128 per-stage slices (contracts inline,
    deepgemm convention) -- PASS"
  mma_m: "64 x 1 math group == cta_tile.M 64 -- PASS"
  mma_n_legal: "N=32, multiple of 8 <= 256 -- PASS"
  trip_count: "GU 8 x 128 = 1024 = D; DR 32 x 128 = 4096 = FF; tails none-needed;
    M pad policy stated in mainloop.tail -- PASS"
  output_coverage: "GU 128 tasks x 32 = 4096 = FF; DR 32 x 32 = 1024 = D; pad
    rows 50..63 written, never read -- PASS (hand)"
  occupancy: "claimed 1 CTA/SM (grid-bound at 80 CTAs); capacity smem 3, regs 1
    at 240 -- PASS; measured_0b query still owed before Phase 2"
  barrier_arrivals: "full: tx 33024 B, arrive 1 elected producer; empty: arrive 1
    elected math lane ON WGMMA RETIREMENT (wait_group<1>); DR splits into
    full_w/empty_w (depth 4) + full_a/empty_a (depth 2); phase (giter/depth)&1.
    gmem counters are NOT barriers -- protocol in grid.persistence -- PASS (hand)"
  arithmetic_intensity: "16 FLOP/B per tile, ~47 whole-op vs ridge 295 --
    memory-bound BY DESIGN (regime); tensor-core idle is correct -- MANUAL accepted"
  floor: "1.85 + 26.2 MB / 2.77 = 11.3 us (meas 2 fit); target 15 us >= floor -- PASS"
  reference: "composition replaced: 23.35 us isolated cold / 19.4 us in-graph
    (PLAN 4.9) -- calibration and slack only, never a bound"
  acceptance: "THE measurement: standalone layer-step harness, same process,
    cold, CUDA-graph timing (benchmark-kernel), fused launch vs the 2-kernel
    composition -- MANUAL (named)"
  falsifiability: "continuity claim -> nsys TMA-gap inspection; counter cost ->
    RTT probe (open task); L2-slope risk -> prototype ncu L2 counters; target ->
    the acceptance harness -- each claim in Why-these-numbers names its refuter"
  concurrency: "L3 filled; memory-bound INVERSION: the copy column's continuity
    is the criterion, tensor-core idle named and accepted -- MANUAL for review"
  vectorisation: "tv_check ALL PASS (10 accesses); 3 rows pinned at 2x sector
    over-fetch via expect -- named exception: direct-RF bf16 stores, ~1 MB total
    vs 25 MB weights, smem staging not worth it"
  addressing: "descriptor read once per task; gmem bases once per task in the
    producer; per-iteration arithmetic = giter%depth ring index (1 IADD) -- PASS (hand)"
  non_mma_accounting: "5 entries, each in L3's CUDA column with cost;
    loop_carried consistent (budget.py loop_carried PASS)"
  rounding_contract: "bf16 in-loop F*S*A (bit-matches tl_ada_scaled_gate), f32
    acc/bias/gelu/gate/residual, bf16 stores; verification.reference mirrors it -- MANUAL"
  residency: "1 CTA/SM, grid-bound; latency hidden by TMA ring depth + 80
    concurrent streams (>4x BDP), not by warps -- argued in Why these numbers;
    the 2-or-4 alternative belongs to the machine-sized full-decoder scope"
  persistence: "grid 80 < 132 -- shortfall named (task-bound prototype);
    cross-CTA waits co-resident BECAUSE of the shortfall; counters host-reset
    (standalone), self-reset deferred -- MANUAL"
  tile_order: "solved, not defaulted: counter-unlock order for DR; weights
    streamed sequentially per slab; x/hidden L2-resident -- MANUAL"
  traceability: "L4 slices -> L2 frames -> L1 loops; buffer names are the
    source's (W1/W2/Wd, hidden, x_res) -- MANUAL"
  loop_bounds: "every range states start/stop/step with trip in comment -- PASS"

# ------------------------------------------------------------- 8. handover
verification:
  reference: "torch chain mirroring adarms.py exactly (bf16 in-loop F*S scale,
    f32 acc, gelu_tanh f32, f32 gate/residual), transplanted inputs, per
    kernel_parity conventions; truncated task lists localize a failure to a task"
  tolerance: "cosine > 0.999 per call site (PLAN ground rule, relaxed 2026-08-21)"
  perf_target: "<= 15 us median, acceptance harness below (floor 11.3, composition 23.35)"
open_questions: []
deviations:
  - field: backend
    spec: "raw CUDA + inline PTX"
    actual: "raw CUDA structure (task loop, hand-placed mbarrier rings, counter
      protocol, dispatch, epilogues, TMA issue PTX) with CuTe arch primitives
      for the wgmma path: TiledMMA/cute::gemm over canonical
      Layout_K_SW128_Atom / Layout_MN_SW64_Atom smem layouts"
    reason: "hand-encoding gmma descriptors and swizzle-atom tilings is exactly
      the confidently-wrong-by-hand class the L4 checker exists to remove; the
      canonical layouts are the same objects tv_check's model asserts"
    impact: "none on the specced dataflow; adds a CUTLASS include dependency
      (CUTLASS_DIR, v4.5.1) to the build"
  - field: "rev-1 drift, RECORDED RETROACTIVELY (rev 2)"
    spec: "mainloop.step 128, pipeline.depth 2, grid.ctas 80"
    actual: "the shipped kernel ran BK=64, depth 4, 132 CTAs -- none recorded"
    reason: "BK 128->64 was taken to cut per-CTA smem (job 543200); it doubled
      GU stages 8->16 and DR stages 32->64 and cost 1.6x on DR-only (25.42 ->
      40.87 us, jobs 541962 vs 543200). job 545906 explains why: the copy column
      is `txns_per_warp x 270 ns`, so halving BK doubles it. Depth 4 and the
      132-CTA worker queue were both measured wins and are now spec values."
    impact: "rev 2 RESTORES step: 128, i.e. returns to the approved rev-1
      number; depth 4 and ctas 132 are promoted into the spec above"
  - field: warp_groups.threads
    spec: "160 (1 producer warp + 1 math warpgroup)"
    actual: "224 -- GU splits the producer into two warps (A ring, W ring) and
      DR into two (W_d ring, A_h ring); warp 6 is reserved and idle"
    reason: "one producer warp issuing two rings serializes them; jobs 543628
      (192 thr) and 543717 (224 thr) measured 79.54 -> 78.04 us"
    impact: "per toolchain.tma_delivery(1) the 270 ns issue rate is PER WARP, so
      splitting a ring across more producer warps is one of only three levers
      that divides a copy floor -- this drift was a win and should be a spec
      value; recorded here pending the rev-2 warp_groups rewrite"
  - field: "problem.layouts.x / .hidden (rev 2)"
    spec: "rev 1: both row-major (M, K)"
    actual: "rev 2: both transposed to (K, M_PAD), M-major"
    reason: "toolchain.measured.tma_descriptor_legality -- row-major puts
      M_PAD=64 in the boxDim[1] slot, hard-capping every activation frame at
      8 KB regardless of BK, so BK=128's 16 KB A-frame is UNREACHABLE without
      this change. Weights are already pre-blocked and uncapped."
    impact: "A operand becomes MN-major (math.a_major); `a_phys_elem`'s
      hand-written K-major SW128 mapping must be re-derived for the MN atom --
      see open_questions; GU's hidden store degrades to scalar strided (priced
      in problem.layouts.hidden); DR's epilogue and `out` are untouched"
  - field: non_mma.a_scale
    spec: "cost counted as pure HMUL2 work"
    actual: "realization adds fence.proxy.async + a math-WG named barrier
      (bar.sync 1,128) between the in-place scale and the wgmma batch, per the
      Known risks entry on the same-WG smem rewrite"
    reason: "async-proxy visibility is a correctness requirement the L3 column
      abstracted away"
    impact: "one barrier per stage inside the math column; hidden under the
      copy column like the scale itself"
---

# ffn_taskloop — persistent task-loop prototype, decoder FFN chain

One launch replaces `tl_rms_factor`-fed `tl_ada_scaled_gate` +
`tl_matmul_gated_res`: 160 tasks (128 GU, 32 DR) on 80 persistent CTAs, GU→DR
ordered by 32 gmem counters, weights streamed through rings that never drain at
task boundaries. The prototype exists to measure three mechanisms before the
full decoder inherits them: **ring continuity across tasks** (no per-op
drain/refill or 1.2–1.8 µs ramp), **decoupled prefetch** (DR's W_down ring runs
dep-free while its hidden ring waits on counters), and **the counter protocol's
real cost** (unmeasured; probe pending).

## Loop nest

### L1 — iteration space

```
  ffn_chain(x[M,D] bf16, F[M] bf16, S[D] bf16,
            W1[D,FF] bf16 k-major, W2[D,FF] bf16 k-major, b1[FF], b2[FF],
            Wd[FF,D] bf16 k-major, g[D], x_res[M,D]) -> out[M,D] bf16

  for n in range(0, FF, 32):                 # 128 tiles     parallel   (GU)
    for k in range(0, D, 128):               # 8 steps       SERIAL, contraction D
      H1[0:M, n:n+32] += (x*F*S)[0:M, k:k+128] @ W1[k:k+128, n:n+32]
      H2[0:M, n:n+32] += (x*F*S)[0:M, k:k+128] @ W2[k:k+128, n:n+32]
                          (50,128) @ (128,32) -> (50,32)   twice, one A read
    hidden[0:M, n:n+32] = gelu_tanh(H1 + b1[n:n+32]) f32 * (H2 + b2[n:n+32]) f32   <bf16 store>

  for n in range(0, D, 32):                  # 32 tiles      parallel   (DR)
    for k in range(0, FF, 128):              # 32 steps      SERIAL, contraction FF
      Cd[0:M, n:n+32] += hidden[0:M, k:k+128] @ Wd[k:k+128, n:n+32]
                          (50,128) @ (128,32) -> (50,32)
    out[0:M, n:n+32] = x_res[0:M, n:n+32] + Cd * g[n:n+32] f32   <bf16 store>

  CROSS-LOOP DEPENDENCE: DR's k-step k needs hidden[:, k:k+128] complete,
  i.e. GU tiles n in [k, k+128) -- 4 of them. That edge, not the loops, is
  what the counters realise.
```

### L2 — mapped to hardware

```
  grid: 80 persistent CTAs, static task lists. 160 threads = producer warp + math WG.
  GU CTA c in [0,64): tasks {n=32c'|c'=c, then c+64}; DR CTA d in [0,16): tasks {2d, 2d+1}.

  GU CTA, one task (n fixed), M padded 50->64:
    C1[64,32] = 0; C2[64,32] = 0                       # f32 RF, 16 elems/thread each
    for k in range(0, D, 128):                         # mainloop: 0..1024 step 128, trip 8
      s, phase = giter % 2, (giter // 2) & 1           # giter runs on across tasks
      producer  wait empty[s] @ phase^1
                A_s[s][64,128]  <- x_pad[0:64, k:k+128]        16384 B  TMA-2D
                W1_s[s][128,32] <- W1[k:k+128, n:n+32]          8192 B  TMA-2D
                W2_s[s][128,32] <- W2[k:k+128, n:n+32]          8192 B  TMA-2D
                S_s[s][128]     <- S[k:k+128]                    256 B  TMA-1D
                full[s].arrive_and_expect_tx(33024)
      math      wait full[s] @ phase
                A_s[s][i,j] = A_s[s][i,j] * F_rf[i] * S_s[s][j]     bf16, in place
                C1 += A_s[s][64,128] @ W1_s[s][128,32]   (64,128)@(128,32)->(64,32)
                C2 += A_s[s][64,128] @ W2_s[s][128,32]   same A read, one commit
                wait_group<1>; empty[s_prev].arrive()    release on RETIREMENT
    epilogue  hidden[0:64, n:n+32] = gelu_tanh(C1+b1) f32 * (C2+b2) f32  -> bf16 st.global
              fence.release.gpu; red.add(counter[n//4], 1)          elected lane
    NEXT TASK: producer already filling frames -- giter, not the task, indexes the ring.

  DR CTA, one task (n fixed):
    Cd[64,32] = 0
    for k in range(0, FF, 128):                        # mainloop: 0..4096 step 128, trip 32
      s_w = giter % 4; s_a, phase_a = giter % 2, (giter // 2) & 1
      producer  wait empty_w[s_w]; W_d[s_w][128,32] <- Wd[k:k+128, n:n+32]  8192 B  DEP-FREE, runs ahead
                poll counter[k//128] >= 4 (acquire)                          GATED
                wait empty_a[s_a]; A_h[s_a][64,128] <- hidden[0:64, k:k+128] 16384 B
                full_w / full_a arrive_and_expect_tx
      math      wait full_w[s_w], full_a[s_a]
                Cd += A_h[s_a][64,128] @ W_d[s_w][128,32]
                wait_group<1>; empty_*[prev].arrive()
    epilogue  out[0:64, n:n+32] = x_res[0:64, n:n+32] f32 + Cd * g[n:n+32] f32 -> bf16
```

### L3 — schedule

```
  GU steady state (memory-bound: the COPY column is the critical resource;
  every gap in it is bandwidth lost, which inverts the usual bubble check)

    copy engine (TMA)            CUDA cores (math0)         tensor cores
    ---------------------------- -------------------------- --------------------
 t0 issue A,W1,W2,S -> frame s'  wait full[s]               batch(s-1): 16x m64n32k16
    (s' = frame freed by         a_scale(s): 8 ld.b128,       (C1 8 + C2 8) of the
    empty arrival; at a task       64 HMUL2, 8 st.b128        PREVIOUS stage in flight
    boundary this is the NEXT      per thread
    task's iter-0 -- no drain)
 t1 in flight ~700 ns [I]        wgmma.fence; issue          batch(s) starts as
                                 batch(s); commit            batch(s-1) retires
 t2 full[s'].arrive_and_         wait_group<1> ->
    expect_tx(33024)             empty[s-1].arrive()
                                 (release BEFORE epilogue
                                 work -- the copy engine
                                 waits on this edge)

  ORDERING EDGES, and which are real
    empty[s-1].arrive sits on wgmma RETIREMENT (wait_group<1>), not on logical
      consumption -- wgmma reads smem async (schedule-l3.md, form 3).
    a_scale(s) -> batch(s) is a true edge (in-place rewrite of A_s before the
      descriptor reads it; wgmma.fence orders it). It does NOT gate the copy
      engine: frames s and s' are different buffers.
    the ONE true cross-engine serialisation is full[s] -- bytes before scale.
    at a task switch NOTHING changes in this timeline: giter indexes the ring,
      the task only changes the gmem addresses the producer computes.

  BUBBLE CHECK (memory-bound: judge the COPY column's continuity, ratios [I])
    copy engine   busy issuing 33 KB / ~1.1 us effective per stage at the
                  80-CTA fair share (25.2 MB / 2.77 TB/s / 80 CTAs / 8+32 stages
                  ... the aggregate is the claim: 80 CTAs x 2 frames > 4x BDP
                  (~2 MB at 2.77 TB/s x 700 ns), so HBM stays saturated iff no
                  CTA's ring drains -- which is exactly what task-continuity
                  guarantees and per-op kernels cannot)
    tensor cores  mostly idle BY DESIGN (intensity 47 vs ridge 295); 16 wgmma
                  per stage ~ hundreds of cycles vs ~1.1 us copy -- not a defect
    CUDA cores    a_scale ~80 inst/thread-stage, hidden under the copy column;
                  idle otherwise. Accepted.

  DR differs in one column: the producer's poll on counter[k//128] can stall
  the A_h issue. The schedule bounds it: GU wave 1 (64 tasks, ~16.8/2 = 8.4 MB
  ~ 3 us stream [I]) completes hidden[:, 0:2048) while DR's first 16 k-slices
  cover exactly that range, and W_d's depth-4 ring keeps the copy column busy
  through any residual wait. Refuted by: nsys showing DR TMA gaps > probe RTT.
```

### L4 — instructions and threads

```
      for ki in range(0, 128, 16):           # iter: start 0, stop step=128, step 16, trip 8
        wgmma.m64n32k16(
          A = A_s[s][0:64, ki:ki+16]         smem-desc, k-major, swizzle 128
          B = W1_s[s][ki:ki+16, 0:32]        smem-desc, k-major, swizzle 64
          C = C1[64, 32]                     f32 RF, 64*32/128 = 16 elems/thread
          clear = (task_iter == 0 and ki == 0) )
      # then the same 8 for W2_s -> C2, one commit_batch for both

  PER-THREAD ACCESS -- GENERATED by tv_check.py from accesses-ffn-taskloop.yaml;
  the table below is its --markdown output, verbatim.

    addressing  task descriptor read once per task (const); gmem bases =
                descriptor.n * stride, computed once per task in the producer;
                the ring index giter%depth is the only per-iteration arithmetic
```

| touch | width | count | banks / sectors |
|---|---|---|---|
| A_s / W1_s / W2_s / S_s fill | n/a | n/a | NO per-thread access -- one elected thread issues the whole tile; the copy engine writes smem |
| smem A/W reads by wgmma | n/a | n/a | NO per-thread access -- operands are read through a matrix descriptor, not by ld.shared |
| a_scale A_s read+write (in place) | 128 b/thread | 8 inst x 4 warps | 4-wavefront, ideal 4 -> 1x |
| a_scale S_s read | 128 b/thread | 8 inst x 4 warps | 2-wavefront, ideal 2 -> 1x |
| gu_c_store (hidden, row stride FF=4096) | 32 b/thread | 8 inst x 4 warps | 8 sectors, ideal 4 -> 2x |
| dr_r_load (x_res, row stride D=1024) | 32 b/thread | 8 inst x 4 warps | 8 sectors, ideal 4 -> 2x |
| dr_c_store (out, row stride D=1024) | 32 b/thread | 8 inst x 4 warps | 8 sectors, ideal 4 -> 2x |
| g_load (per-N gate vector) | 16 b/thread | 8 inst x 4 warps | 1 sectors, ideal 1 -> 1x |
| counter release (red.global.add, one elected lane per GU task) | 32 b/thread | 1 inst x 1 warps | 1 sectors, ideal 1 -> 1x |
| counter acquire (ld.global.acquire poll, one elected producer lane) | 32 b/thread | 1 inst x 1 warps | 1 sectors, ideal 1 -> 1x |

The three 2x rows are the named exception from `checks.vectorisation`: a
direct-RF bf16 store at N=32 fills 16 B of each 32 B sector; pinned with
`expect` as deliberate (~1 MB of traffic against 25 MB of weights).

## Warp-group choreography

Deleted with reason: one producer warp and one math group per CTA is plain
producer/consumer, fully shown by L2/L3. The interesting choreography is
**cross-CTA** (GU's counter release → DR's acquire) and lives in the L2 nest,
`grid.persistence.phase_ordering`, and L3's DR column.

## Why these numbers

**BLOCK_N=32, depth 2, 160 threads** — owner-confirmed. N=32 keeps W rows at
64 B (TMA swizzle minimum; N=16 is a known compile failure from the qkv sweep).
Depth 2 suffices because the copy column is the long pole (L3); smem 66.5 KB
leaves occupancy 3/SM for the full-decoder scope where the grid is
machine-sized.

**Two tasks per CTA, single type per CTA** — 160 tasks would give 1 task/CTA on
a machine-filling grid, exercising no task-switch. 80 CTAs × 2 tasks exercises
the ring-continuity claim on every CTA while keeping ~3 MB in flight (>4× BDP).

**Split rings on DR** — W_down has no data dependence; hidden does. Separating
them (depth 4 / depth 2) is the minimal expression of "weights prefetch freely,
activations wait" — the design thesis this prototype exists to validate.

**Floor and target** — bytes: 25.17 MB weights + ~1.0 MB activations = 26.2 MB
→ `1.85 + 26.2/2.77` = **11.3 µs**. Composition replaced: 23.35 µs (PLAN §4.9,
isolated cold; in-graph 12.4 + 7.0 = 19.4 µs). Target **15 µs**: 1.33× floor,
which prices in the DR unlock tail and counter costs; refuted by the acceptance
harness if missed, and by nsys TMA-gap inspection for the *reason*.

## Op dispatch

How one kernel runs two ops, and why this prototype's answer is deliberately
the easy case. Three dispatch classes exist: **data-driven** (one body,
descriptors carry only pointers/extents — CUTLASS grouped GEMM, MoE
megakernels), **opcode switch per task** (`switch(desc.type)` into inlined
bodies — HazyResearch's instruction interpreter, MPK's workers), and
**branch-once-per-CTA** (single-type task lists; one branch at entry, then a
monomorphic loop). v0 is the third, with the two bodies as `__forceinline__`
device functions sharing one calling convention (`taskgraph.dispatch`).

Two facts the choice does NOT change: the **register allocation is the union
of all reachable bodies** (per-kernel, compile-time — typing CTAs does not
reduce it; only `setmaxnreg` per warp *role* or slimming the fat body does),
and a **type switch inside one CTA changes the frame layout**, so mixed-type
lists need either a drain at the boundary (losing exactly the continuity this
kernel exists to prove) or a page-granular smem pool. v0 sidesteps both:
GU and DR are members of the same GEMM family (union ~70 regs, benign), and
single-type lists keep every ring monomorphic. The full-decoder scope inherits
the real problem — fd_split's 128-f32 accumulator sets the union, and paging
becomes the continuity mechanism — recorded below so it arrives as a plan,
not a surprise.

## Known risks

- **Counter RTT is unmeasured** `[I]` — the probe task must land before Phase 2;
  if release→acquire visibility costs ≳2 µs at 80 polling CTAs, the DR unlock
  schedule needs rework (coarser counters, or DR k-order permuted to trail GU).
- **L2-slope exposure**: GU re-reads x 128× (12.8 MB of L2 traffic) and DR
  re-reads hidden 32× (12.8 MB). At [MEAS-A]'s 0.23 µs/MB — measured on one
  other body, may not transfer — that is up to ~6 µs against an 11.3 µs floor.
  BLOCK_N=64 halves both and fits (frame 48.5 KB, depth 2 = 97 KB, still 2/SM);
  it is the first tuning axis if the prototype lands hot.
- **TMA needs x/hidden/out padded to 64 rows, zero-initialised** (PLAN §2.2's
  NaN lesson applies: garbage pad rows stay row-local in GEMM, but only zeros
  are provably finite).
- **Grid 80 < 132 SMs** — accepted for the prototype; co-residency of the
  cross-CTA waits is guaranteed only because of it (see `grid.cooperative`).
- **Counters are host-reset** — replay-safe self-reset (last DR through zeroes)
  is deferred to the full-decoder scope, recorded here so it is not forgotten.
- **wgmma reads a smem frame the same WG just rewrote in place** (a_scale) —
  needs `wgmma.fence` + smem visibility within the WG; a missing fence is
  silent wrong numbers, flagged for Phase 2 review.
- **The register union is attention's, not ours** — when fd_split joins the
  interpreter (full-layer scope), its ~170-reg body caps the whole kernel at
  1 CTA/SM and `setmaxnreg` cannot carve it out (it splits warp roles, not op
  types). Either the attention body is restructured (smaller BLOCK_M or HD
  tiling in the split) or the interpreter accepts 1 CTA/SM permanently —
  decide when it joins, with the ncu register report from this prototype as
  the baseline. Mixed-type task lists additionally need the page-pool smem
  design (`## Op dispatch`).
