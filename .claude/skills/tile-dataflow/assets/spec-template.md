---
spec_version: 1
kernel: <name>                  # e.g. fp8_gemm_1d1d, mla_decode_splitkv
status: draft                   # draft -> review -> approved.  Only a human moves it to approved.
approved_by:
approved_at:
source: <path or URL>           # only when reverse-engineering an existing kernel

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
  rasterization: TODO           # linear-id -> (m,n) order; "row-major" | "N-major group of 8" | "scheduler struct"
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
    - {name: TODO, bytes: TODO, aliases: TODO, alias_safe_because: TODO}
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
    dtype: TODO                 # e.g. "e4m3 x e4m3 -> f32"
    count_per_stage: TODO       # the "iter" count -- derived, = mainloop.step / inst_shape.K
    a_source: TODO              # smem-desc | rf | tmem
    b_source: TODO              # smem-desc | tmem
    acc: {location: TODO, elems_per_thread: TODO, dtype: f32, cleared: TODO}
    accumulate_across_iters: TODO
    after_batch: TODO           # what runs after the wgmma batch commits and waits

# ------------------------------------------------------------- 6. epilogue
epilogue:
  position: TODO                # after-mainloop | fused-per-iter | split-k-partial
  math: TODO                    # scaling, bias, activation, residual, cast
  path: TODO                    # rf -> smem -> TMA-store | rf -> gmem st.global | reduce-add
  output: {tile: TODO, dtype: TODO}
  split_reduction: TODO         # none | atomics | separate combine kernel (name it)

# ------------------------------------------------------------- 7. checks
checks:                         # fill with the computed value AND pass/fail, not just "ok"
  smem: TODO                    # depth*per_stage + non_staged <= arch cap
  threads: TODO
  acc_registers: TODO
  mma_k: TODO                   # iters*inst.K == mainloop.step
  mma_m: TODO
  mma_n_legal: TODO
  trip_count: TODO
  output_coverage: TODO
  occupancy: TODO
  barrier_arrivals: TODO
  arithmetic_intensity: TODO    # tile FLOP/byte vs arch ridge point

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

### L3 — innermost body, expanded

```
      for <ki> in range(0, <step>, <inst K>):   # iter: start 0, stop <step>, step <inst K>, trip <count>
        <unit>.m<M>n<N>k<K>(
          A = <tensor>[<exact slice>]    <smem-desc | rf | tmem>, <majorness>
          B = <tensor>[<exact slice>]    <smem-desc | tmem>, <majorness>
          C = <tensor>[<M>, <N>]         <dtype> in <RF|TMEM>, <elems>/thread
          clear = <which iter uses ScaleOut::Zero, or "never — loop-carried"> )
```

Repeat the L3 block once per distinct MMA step (attention has at least two).

## Schedule

Only for kernels whose warp groups **interact** — a seesaw, a ping-pong, any
hand-off through smem under a named barrier. Show the ordering the loop nest
cannot: which group waits on whom, in what sequence, through which barrier.
Delete this section with a reason when the split is plain producer/consumer and
the L2 nest already shows it.

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
