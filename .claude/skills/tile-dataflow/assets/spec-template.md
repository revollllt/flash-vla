---
spec_version: 1
kernel: <name>                  # e.g. fp8_gemm_1d1d, mla_decode_splitkv
status: draft                   # new kernel: draft -> review -> approved, and only a human
                                # moves it to approved. Reverse-engineering a kernel that already
                                # exists uses `reference` instead -- nothing to sign off, nothing to unblock.
approved_by:
approved_at:
source: <path or URL>           # only when reverse-engineering an existing kernel

# ------------------------------------------------- 0a. toolchain (Phase 0)
# Where every floor and target below comes from. A denominator with no entry
# here came off a datasheet, which is the failure Phase 0 exists to prevent.
toolchain:
  device: TODO                  # and the specific machine, if the cluster is not uniform
  compiler: TODO                # nvcc / gcc / library versions that the numbers were taken with
  harness: TODO                 # and it must be the one the kernel is ACCEPTED under
  clocks: TODO                  # pinned, or the noise floor if not -- decides what counts as a difference
  measured:                     # Phase 0a; see references/example-phase0.md
    launch_floor: TODO          # empty kernel vs grid size -- the per-launch grid ramp
    bandwidth_fit: TODO         # t_us = a + MB/b, with the range it was fitted over
    bandwidth_vs_ctas: TODO     # smallest grid that reaches bandwidth
    best_known_impl: TODO       # library on the same shape -- calibration and slack, NOT a bound
    barrier_costs: TODO         # cluster sync per cluster size; note it is a zero-skew floor
  measured_0b:                  # re-run once L2 fixes the tile
    occupancy: TODO             # blocks/SM and placeable clusters at the REAL smem+register budget
  job_ids: TODO                 # so every number above can be re-read or discarded

# ---------------------------------------------------------------- 0. problem
arch: sm90a                     # sm80 | sm86 | sm89 | sm90a | sm100a
problem:
  op: <one line of math, e.g. "D = (A@B) * sfa * sfb, A row-major fp8">
  dims: {M: TODO, N: TODO, K: TODO}
  dynamic: [TODO]               # which dims are runtime values; [] if all compile-time
  dtypes: {a: TODO, b: TODO, acc: f32, d: TODO}
  layouts: {a: TODO, b: TODO, d: TODO}    # e.g. "row(M,K)", "col(N,K)", "paged(page=64)"
  regime: TODO                  # what the kernel is tuned for: "latency, M<=64" / "throughput, M>=4096"

# ------------------------------------------------------------- 1. grid / CTA
grid:
  mode: TODO                    # wave | persistent
  ctas: TODO                    # e.g. "ceil(M/128)*ceil(N/128)" or "132 (= num_sms)"
  shape: TODO                   # blockIdx meaning, e.g. "(m_blocks, kv_heads, sm_parts)"
  cta_tile: {M: TODO, N: TODO}  # the output tile one CTA owns
  rasterization: TODO           # linear-id -> (m,n) order, WITH its L2 argument: which operand
                                # consecutive CTAs share, and how much of it is co-resident
  l2_schedule: TODO             # static graph? then the map is an offline optimum, not a heuristic.
                                # "solved, minimising <objective>" | "defaulted to linear"
  persistence:                  # delete with a reason when mode is `wave`
    cta_per_sm: TODO           # intended residency
    grid_realises_it: TODO      # grid >= SM_count * cta_per_sm, or say why not
    scheduler: TODO             # how a CTA picks its next tile, and in what order
    phase_ordering: TODO        # none | grid barrier | per-tile semaphores (name the granularity)
  cooperative: TODO             # false unless CTAs block on each other; see the mutual
                                # exclusion with clusters in the schema
  cluster:                      # delete on sm80 with a reason
    shape: [1, 1, 1]
    multicast: TODO             # which operand is multicast to the cluster, or none
  launch:
    threads: TODO               # must equal sum of warp_groups.threads
    cta_per_sm: TODO
    smem_B: TODO                # derived; see checks.smem
    max_regs_per_thread: TODO

# ------------------------------------------------------------- 2. mainloop
mainloop:
  axis: TODO                    # K | kv_seqlen | ...
  step: TODO                    # extent consumed per iteration (BLOCK_K, PAGE_BLOCK_SIZE, ...)
  trip_count: TODO              # derived
  tail: TODO                    # how a ragged last iteration is handled: predication | pad | none-needed
  operands_per_iter:
    - {name: A, tile: [TODO, TODO], dtype: TODO, bytes: TODO, src: gmem, via: TODO}   # via: TMA-2D | TMA-im2col | cp.async | ld.global
    - {name: B, tile: [TODO, TODO], dtype: TODO, bytes: TODO, src: gmem, via: TODO}
  loop_carried: TODO            # what persists across iterations: "acc in RF" / "acc + running max m + running sum l"
  per_iter_math: TODO           # non-MMA work every iteration: "dequant promote acc into final_acc" / "online softmax rescale" / none

# ------------------------------------------------------------- 3. pipeline
pipeline:
  depth: TODO                   # number of buffered stages
  stage_index: TODO             # e.g. "iter % depth"
  phase: TODO                   # e.g. "(iter / depth) & 1"
  prologue: TODO                # stages prefetched before the steady state
  per_stage_bytes: TODO         # derived, sum over staged buffers
  staged_buffers:
    - {name: TODO, shape: [TODO, TODO], dtype: TODO, bytes: TODO, swizzle: TODO}
  non_staged_buffers:
    - {name: TODO, bytes: TODO, swizzle: TODO, aliases: TODO, alias_safe_because: TODO}
  barriers:
    - {name: full,  kind: TODO, count: TODO, init_arrive_count: TODO, produced_by: TODO, waited_by: TODO}
    - {name: empty, kind: TODO, count: TODO, init_arrive_count: TODO, produced_by: TODO, waited_by: TODO}
  # kind: mbarrier-tx (TMA transaction) | mbarrier | named-barrier | cp.async-group

# ------------------------------------------- 4. warp specialization / roles
warp_groups:                    # one entry per role; delete the section on sm80 with a reason
  - id: TODO                    # producer | math0 | math1 | epilogue
    warps: TODO
    threads: TODO
    regs: TODO                  # after setmaxnreg dealloc/alloc; omit if not reconfigured
    role: TODO                  # one line
    issues: TODO                # which instructions this group actually issues
    elected: TODO               # true when a single elected thread issues (TMA, tcgen05.mma)
inter_group_sync: TODO          # named barriers / mbarriers between groups, and what each orders

# ------------------------------------------------ 5. iters (instruction level)
math:
  - group: TODO                 # which warp_group runs this
    stage_phase: TODO           # where in the stage this fires, when a stage has several math steps
    unit: TODO                  # wgmma.mma_async | mma.sync | tcgen05.mma
    inst_shape: {M: TODO, N: TODO, K: TODO}
    contracts: TODO             # the axis THIS MMA reduces, as a name from problem.dims or
                                # `name=extent`. Usually mainloop.axis, but not always: attention's
                                # QK^T contracts the head dim while the mainloop walks kv_seqlen,
                                # so without this field count_per_stage looks wrong when it is right
    dtype: TODO                 # e.g. "e4m3 x e4m3 -> f32"
    count_per_stage: TODO       # the "iter" count -- derived, = contracts extent / inst_shape.K
    a_source: TODO              # smem-desc | rf | tmem
    b_source: TODO              # smem-desc | tmem
    acc: {name: TODO, location: TODO, elems_per_thread: TODO, dtype: f32, cleared: TODO}
                                # `name` so the L3/L4 nest has a field to trace to --
                                # a stage-local accumulator otherwise appears only in the prose
    accumulate_across_iters: TODO
    after_batch: TODO           # what runs after the wgmma batch commits and waits

# ------------------------------- 5b. non-MMA work (the CUDA-core column of L3)
# One entry per distinct computation that is not an MMA: elementwise scaling,
# activations, the online-softmax rescale, any reduction, any transpose. These
# occupy L3's CUDA-core column, and when one of them sits between a load and the
# MMA that consumes it, it serialises the two engines the kernel exists to
# overlap. `none` is a legal answer for a plain GEMM and an illegal omission.
non_mma:
  - id: TODO                    # online_rescale | gelu_gate | a_tile_scale | row_sumsq
    primitive: TODO             # a name from references/primitives.md, or `none` if bespoke
    params: TODO                # the primitive's parameters; omit when primitive is none
    where: TODO                 # prologue | mainloop.per_iter | mainloop.per_stage | epilogue
    kind: TODO                  # elementwise | reduction | scan | transpose | cast
    over: TODO                  # axis and extent, e.g. "row, BLOCK_N=64" / "column, BLOCK_K"
    span: TODO                  # lane | warp | warpgroup | cta | cluster -- decides the primitive
    mechanism: TODO             # none | shfl.bfly | redux.sync | smem tree | DSMEM | atomic --
                                # HOW it is computed. Distinct from `primitive`, which names the
                                # contract in references/primitives.md; one key cannot carry both
    loop_carried: TODO          # [] | [m, l] for online softmax | [sumsq]
    dtype: TODO                 # compute dtype AND where each rounding lands -- see below
    cost: TODO                  # ops/thread and unit: "32 HMUL2 + 8 ex2.approx, CUDA cores"
    touches: TODO               # buffers read/written and in what layout -- L4 checks these for conflicts
    on_critical_path: TODO      # sits between a load and the MMA consuming it? yes/no + why

# `dtype` is the numerical contract, not a formatting detail. "scale the A tile
# by S" is a different function depending on whether the multiply and its
# rounding happen in bf16 in shared memory or in fp32 on the accumulator, and
# both are defensible -- so the spec has to say which, and the reference the
# parity test compares against has to mirror it.

# ------------------------------------------------------------- 6. epilogue
epilogue:
  position: TODO                # after-mainloop | fused-per-iter | split-k-partial
  math: TODO                    # scaling, bias, activation, residual, cast
  path: TODO                    # rf -> smem -> TMA-store | rf -> gmem st.global | reduce-add
  output: {tile: TODO, dtype: TODO}
  split_reduction: TODO         # none | atomics | separate combine kernel (name it)

# ------------------------------------------------------------- 7. checks
l4_accesses: TODO               # path to the access file whose layouts and thread-value maps
                                # produce the L4 table below; `scripts/tv_check.py` computes every
                                # width, transaction and conflict count in it. See
                                # references/l4-access.md. `none` is legal only when nothing in this
                                # kernel is touched per-thread -- say so rather than deleting it.
checks:                         # fill with the computed value AND pass/fail, not just "ok".
                                # `scripts/budget.py <spec>` computes the arithmetic ones
  smem: TODO                    # depth*per_stage + non_staged <= arch cap
  threads: TODO
  acc_registers: TODO           # per thread
  register_budget: TODO         # threads_per_cta * regs_per_thread * cta_per_sm <= 65536 --
                                # this is what actually forces cta_per_sm
  mma_k: TODO                   # iters*inst.K == mainloop.step
  mma_m: TODO
  mma_n_legal: TODO
  trip_count: TODO
  output_coverage: TODO
  occupancy: TODO
  barrier_arrivals: TODO
  arithmetic_intensity: TODO    # tile FLOP/byte vs arch ridge point

  # --- is the target reachable at all? (Phase 0) ---
  floor: TODO                   # target >= a + MB/b PLUS one launch cost per launch
  reference: TODO               # vs the best existing impl -- calibration and slack, never a bound
  acceptance: TODO              # THE one measurement that decides it, and it is the one it ships under
  falsifiability: TODO          # each perf claim names the measurement that would refute it

  # --- concurrency and lowering (L3, L4) ---
  concurrency: TODO             # L3 bubble check filled; no CUDA-core stretch between copy and tensor
  vectorisation: TODO           # widest legal access, coalesced, bank-conflict-free, or exception named
  addressing: TODO              # per-iteration address arithmetic counted; invariants hoisted
  non_mma_accounting: TODO      # every non_mma entry in L3's CUDA-core column with its cost;
                                # loop_carried names match mainloop.loop_carried
  rounding_contract: TODO       # every non_mma.dtype says where rounding lands; parity ref mirrors it

  # --- grid and locality ---
  residency: TODO               # cta_per_sm with the smem AND register arithmetic behind it;
                                # a value of 1 on a latency-bound kernel is argued, not inherited
  persistence: TODO             # grid >= SM_count*cta_per_sm; cooperative only where CTAs block on
                                # each other, never with a cluster on sm90; semaphores self-resetting
  tile_order: TODO              # rasterization carries an L2 argument; on a static graph, solved or defaulted
  traceability: TODO            # every bound at L4 traces to L2, every name at L2 to L1
  loop_bounds: TODO             # every range states start/stop/step; trip counts match the YAML

# ------------------------------------------------------------- 8. handover
verification:
  reference: TODO               # what the kernel is checked against
  tolerance: TODO
  perf_target: TODO             # metric + number + how it is measured
open_questions: []              # must be empty before status: review
deviations: []                  # filled in Phase 2 when the backend cannot honour the spec
---

# <kernel name>

## Loop nest

Notation: every `range` states start, stop, and step, with the trip count in a
comment. Slices are exact (`T[a : a+t]`), transposes are explicit (`ᵀ`), `=` is
stage-local and `+=` is carried across the mainloop. Annotate every compute line
with `(a,b)@(b,c) -> (a,c)`.

### L1 — iteration space

```
  <kernel>(<In1>[<dim>, <dim>] <dtype> <layout>,
           <In2>[<dim>, <dim>] <dtype> <layout>) -> <Out>[<dim>, <dim>] <dtype> <layout>

  for <p0> in range(0, <DIM>, <tile>):        # <trip> tiles          parallel
    for <p1> in range(0, <DIM>, <tile>):      # <trip> tiles          parallel
      for <r> in range(0, <RDIM>, <step>):    # <trip> steps          SERIAL, contraction
        <Out>[<p0>:<p0>+<t>, <p1>:<p1>+<t>] += <In1>[<p0>:<p0>+<t>, <r>:<r>+<s>]
                                             @ <In2>ᵀ[<r>:<r>+<s>, <p1>:<p1>+<t>]
                                               (a,b) @ (b,c) -> (a,c)
```

### L2 — mapped to hardware

```
  grid <mode>, <ctas> CTAs: <which loops are distributed, which stay serial>
  <threads> threads = <warp group breakdown>; <what each group owns>

  for <tile coord> in <scheduler or grid id>:     # <tiles per CTA>
    <Acc>[<m>, <n>] = 0                           # <dtype> in <RF|TMEM>, <elems>/thread, carried

    for <r> in range(0, <RDIM>, <step>):          # mainloop: start 0, stop <RDIM>, step <step>, trip <count>
      s, phase = (<r>//<step>) % <depth>, (<r>//<step>)//<depth> & 1     # stage

      producer   wait empty[s] @ phase^1
                 <A_s>[s][<m>,<k>] <- <In1>[<slice>, <slice>]     <bytes> B  <TMA|cp.async>
                 <B_s>[s][<n>,<k>] <- <In2>[<slice>, <slice>]     <bytes> B
                 full[s].arrive_and_expect_tx(<per_stage_bytes>)

      math WG w  wait full[s] @ phase
                 <C_s>[<m>,<n>] = <A_s>[s][<slice>] @ <B_s>[s]ᵀ[<slice>]
                                  (a,b) @ (b,c) -> (a,c)   <acc dtype/location>
                 empty[s].arrive()
                 <Acc> += <the loop-carried update, when it differs from C_s>

    epilogue   <Out>[<slice>, <slice>] <- <Acc>
```

Delete the `producer` / `math WG` split with a reason on architectures without
warp specialization, and replace it with the single body.

### L3 — schedule

```
  engine timeline for one steady-state stage

    copy engine (TMA/cp.async)  CUDA cores (LSU/ALU)     tensor cores
    --------------------------- ------------------------ -------------------
 t0 <issued by whom, how many>  <what runs, ops/thread>  <which mma, whose
 t1 <in flight / expect_tx>     <...>                     operands>
 ...

  ORDERING EDGES
    <which issue must be hoisted above which compute, and why that is safe>
    <which pairs genuinely overlap, and on which different units>
    <the ONE true serialisation point, and the barrier that enforces it>

  BUBBLE CHECK
    copy engine idle    <cycles, or 0>
    tensor cores idle   <...>
    CUDA cores idle     <...>
```

An empty column is a bubble, and this is the cheapest place to find one. A long
CUDA-core column sitting *between* the copy column and the tensor column is a
serialisation that L1 and L2 cannot show, and it is the single most common way a
fused kernel loses to the sum of its parts.

### L4 — instructions and threads

```
      for <ki> in range(0, <step>, <inst K>):   # iter: start 0, stop <step>, step <inst K>, trip <count>
        <unit>.m<M>n<N>k<K>(
          A = <tensor>[<exact slice>]    <smem-desc | rf | tmem>, <majorness>
          B = <tensor>[<exact slice>]    <smem-desc | tmem>, <majorness>
          C = <tensor>[<M>, <N>]         <dtype> in <RF|TMEM>, <elems>/thread
          clear = <which iter uses ScaleOut::Zero, or "never — loop-carried"> )

  PER-THREAD ACCESS  (every gmem and smem touch in the stage)
    <buffer>  <bits/thread>  <threads per line, transactions>  <coalesced?>
    <smem>    <swizzle atom>  <bank-conflict ways>
    addressing  <hoisted to prologue: ...>  <per-iteration: ... instructions>
```

Repeat the MMA block once per distinct MMA step (attention has at least two);
the per-thread access table covers the whole stage, not one instruction.

## Warp-group choreography

A different axis from L3, and both are needed when they apply. **L3 is one
stage across the three engines; this is one warp group across many stages.**
Only for kernels whose warp groups **interact** — a seesaw, a ping-pong, any
hand-off through smem under a named barrier. Show the ordering neither the loop
nest nor L3 can: which group waits on whom, in what sequence, through which
barrier, and over how many stages the pattern repeats. Delete this section with
a reason when the split is plain producer/consumer and L3 already shows it.

```
  [0] <step>                          --> <barrier it releases>
  [1] <step>                          <-- <barrier it waits on>
  ...
```

## Why these numbers

<One short paragraph per non-obvious choice: why this tile, why this depth,
why this warp split. This is what the reviewer disagrees with, so make the
reasoning attackable rather than asserting the conclusion.>

## Known risks

<Register pressure, smem headroom, tail behaviour, anything measured as tight
in `checks`. If nothing is tight, say so.>
