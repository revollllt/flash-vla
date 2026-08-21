---
name: tile-dataflow
description: Design a GPU kernel in three phases — first measure the machine's real cost model, then write a reviewable dataflow spec (CTA tiling, residency, mainloop, pipeline stages, the per-stage schedule across copy engine / CUDA cores / tensor cores, and the lowered per-thread accesses), get human sign-off, then generate the kernel in TileLang, CuTeDSL, CUTLASS/CuTe, or raw CUDA/PTX. Use this whenever the user wants to write, design, port, or plan a GPU kernel — GEMM, attention, MoE, or a fused op — or mentions tile/block sizes, pipeline stages, warp specialization, producer/consumer warp groups, wgmma/TMA/mbarrier scheduling, or wants the dataflow described before any code is written. Also use when asked to reverse-engineer an existing kernel (DeepGEMM, FlashMLA, FlashAttention, CUTLASS collectives) into a tile-level description, or to port a kernel between TileLang / CUTLASS / CuTeDSL / CUDA.
---

# Tile-Level Dataflow → Kernel

A kernel that gets written before its dataflow is pinned down gets rewritten.
The expensive decisions — how the output is tiled across CTAs, how deep the
pipeline is, which warps load and which warps compute, how many MMA
instructions fire per stage — are all made in the first ten minutes and are
all painful to change once there is code. So this skill splits the work:

**Phase 0** measures what the machine actually costs, so that the spec's floors
and targets are denominators someone measured rather than numbers off a
datasheet. **Phase 1** produces a spec that a human kernel engineer can review
and correct in minutes. **Phase 2** turns the approved spec into code. A hard
gate sits between 1 and 2: *no kernel code is written until a human signs off
on the spec.*

Phase 0 exists because of one repeated failure: floors computed as
`bytes / peak_bandwidth` omit the per-launch grid ramp and assume a bandwidth the
machine never reaches, so targets end up *below their own floor* — which makes
every later "we missed by 2x" unfalsifiable.

The gate is the entire point. Do not soften it, do not "start a draft while we
discuss", do not write "illustrative" code in Phase 1.

## The mental model

A GPU kernel is one nested structure. Each level answers *what is the unit of
work here, and what data moves*:

```
grid        which output tile does this CTA own?          -> cta_tile, rasterization, cluster
 mainloop   which slice of the reduction axis is this?    -> axis, step, trip_count
  stage     which smem buffer, and who filled it?         -> depth, barriers, producer/consumer
   iter     which MMA instruction, acc living where?      -> inst_shape, count, acc location
```

`stage` and `iter` are the two levels people leave vague, and they are the two
that decide performance. One mainloop iteration *usually* occupies one pipeline
stage — one tile-granular load, one tile-granular compute. When it does not, say
the ratio: an iteration consuming two stages makes them a **compute pair**, not
a fill/drain pair, which changes what `depth: 2` means. Inside that one compute
there are typically several MMA instructions, because the hardware MMA shape is
smaller than the tile — that count is the `iter` granularity. Say both numbers
explicitly, every time.

Warp specialization cuts across all four levels: on Hopper and Blackwell the
CTA's warps are split by *role* (a producer warp group issuing TMA, one or more
math warp groups issuing MMA, sometimes a separate epilogue group), and the
stage barriers are what couple them. A spec that names stages but not the warp
groups that arrive on them is incomplete.

## The four levels

The loop nest above says *when* things happen. It does not say what **shape**
anything is, nor where a loop starts, stops, and how far it steps — and that is
what a reviewer actually checks. So every spec carries a loop-structured
pseudocode block at four levels, each a refinement of the one above it:

| Level | Is | Answers |
|---|---|---|
| **L1 — kernel** | the iteration space, no hardware in it | What crosses the kernel boundary, and what loops cover the whole problem. Which loops are parallel and which are the contraction. |
| **L2 — CTA and stage** | the same nest, mapped to hardware | Which loops become the grid, which stays serial inside a CTA, what one stage loads and computes, with tile shapes and byte counts. |
| **L3 — schedule** | one stage, seen as three engines running concurrently | What the **async copy engine (TMA/cp.async)**, the **CUDA cores** and the **tensor cores** each issue during this stage, which edges order them, and where the bubbles are. |
| **L4 — instructions and threads** | the innermost body, lowered | The MMA instruction and its `m x n x k`, the iter loop's bounds, where A, B, C live — and per thread, the access width, the coalescing, the bank conflicts, and where the address arithmetic lives. |

**L3 decides *when*; L4 decides *what and how wide*.** A bubble is
an L3 defect; a bank conflict is an L4 defect. Keeping them in one level is what
lets a spec assert an overlap that nobody can review.

### Which level owns which resource

Shared resources are **budgeted high and realised low**.

| Resource | Budget decided at | Realisation checked at | Failure mode when only the low level names it |
|---|---|---|---|
| smem capacity | **L2** — `staged_buffers` × `depth` + non-staged | checks.smem | depth chosen, then found not to fit |
| smem **layout**: swizzle **and row stride** | **L2** — the swizzle atom, and the tile extent that sets the stride | **L4** — bank-conflict ways | the atom that suits the MMA is wrong for the epilogue or reduction reading the same buffer. And an unswizzled buffer's conflicts are set by its **row stride**, so the only fix is a different tile extent — an L4 symptom whose sole cure is an L2 number |
| **registers** | **L2** — acc elems/thread, `setmaxnreg` split, `max_regs_per_thread` → `Block Limit Registers` | **L4** — spills, whether the accumulator really stays in RF | occupancy is capped by registers co-equally with smem; treat it as codegen and the grid is already wrong before a line is written |
| engine concurrency | **L3** — the timeline | L3's own bubble check | asserted, not specified |
| grid / CTA count | **L1/L2** | Phase 0 measurement 3 | a grid too small to reach bandwidth, discovered at the end |

**A resource appearing only at L4 was treated as a code-generation problem when
it was a tiling decision.**

### Why L3 is not optional in a fused kernel

L1 and L2 describe *dependency*. A fused kernel's whole thesis is *concurrency* —
three engines busy at once — and a spec naming only which warp group holds which
*role* has **asserted** the overlap, not specified it.

Observed, not imagined. A spec said "the producer warp group owns the A-tile
transform, which overlaps it with the next stage's TMA". What got built was:

```
    wait TMA(s) -> transform s on CUDA cores -> release -> MMA(s)
                                                ^ TMA(s+1) issued HERE
```

The copy engine idles for the whole transform. Hoisting `TMA(s+1)` above it — one
line, invisible at L1/L2 and at the instruction level — was worth 0.8 us `[MEAS-A]` on a
14 us kernel, and was found by accident in Phase 2. In the same kernel the
transform was 24-27% of cycles, top stall `short scoreboard` 42.7%, 0.30 eligible
warps per scheduler. All L3 information; none of it survives into an instruction
list.

One rule covers most of these: **find what actually gates the next copy, because
it is later than it looks.** Three forms, all found by writing the timeline down:

- the issue sits below CUDA-core work that does not gate it — hoist it;
- the *release* sits below that work — `empty[s].arrive()` before the promote,
  not after, since the release is what the copy engine waits on;
- the buffer is not dead yet. wgmma reads smem **asynchronously**, so a stage is
  reusable only once `wgmma.wait_group` has *retired* the last instruction
  reading it — not when its barrier fired. In a seesaw that can be several
  steps after the data was consumed logically.

So every stage gets a timeline plus its ordering edges — three columns, or more
when warp groups must be kept out of phase. **An empty column is a bubble the
spec shows you before the kernel exists.**

**Where the cycle counts come from, since Phase 0 does not measure them.** None
of its five measurements yields wgmma cycles per instruction, CUDA-core issue
rate, or TMA issue latency. So:

- **the ordering edges and which column is empty are structural** — they follow
  from the dependencies and need no cycles at all. That is the part that catches
  bubbles, and it is the part worth arguing over;
- **cycle counts are `[I]` and must be marked so.** Published peaks are an
  acceptable source *for these* precisely because the conclusion does not rest on
  their absolute values;
- **the criterion is the ratio between columns, never the absolutes.** "The copy
  column is 527 against the tensor column's 512, so they are balanced within 3%"
  survives both numbers being 20% wrong. "The copy column is 527 cycles" does not.
  **But state the rate once, with its derivation, and check which columns it
  actually scales** — a column counted in *instructions* does not move with an
  FLOP/cycle rate, so halving that rate doubles its share while leaving the
  tensor column's ratio intact. A ratio is only robust between columns derived
  the same way.

On sm80 there is no separate copy engine: `cp.async` consumes LSU issue slots in
the same warps that compute, so the three-column model's premise — different
units, no contention — does not hold. Draw two columns and say so.

### Notation

```
for i in range(start, stop, step):   ALWAYS all three, plus the trip count in a comment.
                                     `for k in range(N)` hides the two numbers that matter.
T[a : a+t, c : c+u]                  the exact slice this iteration touches — not "a tile of T"
Name[dim, dim]                       named dims from problem.dims, not bare numbers where a name exists
Aᵀ                                   explicit transpose; on an MMA operand this is what majorness means
@                                    contraction over the shared inner dim
=  vs  +=                            recomputed each stage  vs  accumulated across the mainloop
(a,b)@(b,c) -> (a,c)                 concrete extents, on every compute line
reduce_<op>(T[i, lo:hi], axis)       a reduction with its axis and extent explicit --
                                     never a bare "reduce"; the extent is its cost
m, l = carry(...)                    a LOOP-CARRIED reduction (online softmax's running
                                     max and sum). The rule is broader than this form:
                                     **any name the nest carries across the mainloop --
                                     including a plain `+=` accumulator -- must appear in
                                     `mainloop.loop_carried`**, or the spec disagrees with
                                     itself. Both worked examples broke this, once each,
                                     and only the accumulator form escaped the narrow rule
f(x) <dtype>                         an elementwise op annotated with the dtype it
                                     computes in -- `gelu(c) f32` and `gelu(c) bf16` are
                                     different functions, not formatting
```

At L1 the contraction axis is visible because the names are symbolic. **At L2 the
slices are numeric, so `[0:128] @ [0:128]` says nothing when `BLOCK_K == BLOCK_N`
— annotate the axis name there whenever the extents collide.** SKILL.md's own
Shape example needed this.

Three things fall out for free. The **shared inner name is the contraction
axis**, so a reader finds the reduction by eye — and if it does not match
`mainloop.axis` the spec is wrong. The **`=` / `+=` split makes
`accumulate_across_iters` visible** without prose. And **explicit `range` bounds
make the tail policy checkable**: `range(0, K, 128)` with `K % 128 != 0` and no
predication noted is a bug you can see.

### Write the form the hardware computes

```
Oᵀ[dv, B_H] = Vᵀ[dv, kv] @ Pᵀ[kv, B_H]      and      O[B_H, dv] = P[B_H, kv] @ V[kv, dv]
```

are the same math and two different kernels. They assign different tensors to
the MMA's A and B operands, imply different smem layouts, and need different
`Major::` flags. Which one you write *is* the decision — leaving it implicit is
the bug. So transposes are never elided for tidiness: `ᵀ` on an operand is a
claim about how that tensor sits in shared memory.

### Shape

A scaled FP8 GEMM, complete at all four levels:

```
L1 ------------------------------------------------------------ iteration space
  gemm(A[M,K] e4m3 k-major, B[N,K] e4m3 k-major,
       SFA[M, K/128] f32, SFB[N, K/128] f32) -> D[M,N] f32

  for m0 in range(0, M, 128):                 # ceil(M/128) tiles      parallel
    for n0 in range(0, N, 128):               # ceil(N/128) tiles      parallel
      for k0 in range(0, K, 128):             # K/128 steps            SERIAL, contraction
        D[m0:m0+128, n0:n0+128] += SFA[m0:m0+128, k0//128] * SFB[n0:n0+128, k0//128]
                                   * ( A[m0:m0+128, k0:k0+128] @ Bᵀ[k0:k0+128, n0:n0+128] )
                                       (128,128) @ (128,128) -> (128,128)

L2 ------------------------------------------------------- mapped to hardware
  grid 132 persistent CTAs: (m0,n0) distributed by the scheduler, k0 stays serial per CTA
  384 threads = 1 producer WG + 2 math WGs; math WG w owns rows [64w, 64w+64)

  for (m0, n0) in scheduler(cta_id):          # ~ceil(M/128)*ceil(N/128)/132 tiles per CTA
    D_acc[64, 128] = 0                        # f32 RF, 64 elems/thread, carried across k0

    for k0 in range(0, K, 128):               # mainloop: start 0, stop K, step 128, trip K/128
      s, phase = (k0//128) % 4, (k0//128)//4 & 1        # stage, depth 4

      producer  wait empty[s] @ phase^1
                A_s[s][128,128] <- A[m0:m0+128, k0:k0+128]    16384 B  TMA
                B_s[s][128,128] <- B[n0:n0+128, k0:k0+128]    16384 B  TMA
                sfa_s[s][128]   <- SFA[m0:m0+128, k0//128]      512 B
                sfb_s[s][128]   <- SFB[n0:n0+128, k0//128]      512 B
                full[s].arrive_and_expect_tx(33792)

      math WG w wait full[s] @ phase
                C_s[64,128] = A_s[s][64w:64w+64, 0:128] @ B_s[s]ᵀ[0:128, 0:128]
                                (64,128) @ (128,128) -> (64,128)   f32 RF, stage-local
                empty[s].arrive()                                   release before promoting
                D_acc += sfa_s[s][64w:64w+64] * sfb_s[s][0:128] * C_s     64 CUDA-core FMAs

    epilogue  D[m0+64w : m0+64w+64, n0:n0+128] <- D_acc

L3 ------------------------------------------------ schedule, one stage
  engine timeline for stage s (three columns = three engines, top to bottom = time)

    copy engine (TMA)         CUDA cores (LSU/ALU)      tensor cores (WGMMA)
    ------------------------- ------------------------- ----------------------
 t0 issue A_s[s'],B_s[s']     WG0 promote(s-1)          WG1 mma k-blk 0..3 of s-1
    elected thread, 1 inst    64 f32 FMA/thread
    s' = most recently released buffer, NOT s+1: in a depth-d pipeline the
    producer floats 1..d-1 stages ahead and the distance drifts with the balance
 t1 in flight ~700 ns [I]     WG0 wait full[s] @ phase  WG1  "
 t2 full[s].arrive_and_       WG1 wait<0> returns;      WG0 mma k-blk 0..3 of s
    expect_tx(33792)          empty[s-1].arrive();
                              promote(s-1)
 t3 wait empty[s'+1] @ phase  WG0 wait<0> returns;      WG0  "
                              empty[s].arrive();
                              promote(s)

  Columns are per WARP GROUP on the math side, and that is load-bearing: within
  ONE warp group `wgmma.wait_group<0>` is a full barrier on the batch, so its
  promote sits strictly BETWEEN two batches and the tensor cores idle for the
  promote's whole duration. The overlap above is real only because the promote
  belongs to the *other* group. Nothing orders WG0 against WG1; the tensor cores
  serialising their two batches is the only thing that staggers them.

  ORDERING EDGES, and which are real
    the TMA issue sits ABOVE the promote, not below: it depends on empty[s']
      only, never on stage s's compute. Put it after any CUDA-core work and the
      copy engine idles for exactly that work's duration.
    empty[s].arrive() likewise sits ABOVE the promote -- the release is what the
      copy engine waits on. Safe only because the scales were pulled into
      registers before warpgroup_arrive.
    the ONE true serialisation is full[s] -- bytes must land before wgmma reads.

  BUBBLE CHECK   (cycles are [I]; the criterion is the RATIO, not the absolutes)
    copy engine idle    the t3 wait on empty[s'+1] only, and only when the copy
                        column runs UNDER the math column. Balanced here, so the
                        residual wait lands on the math side as full[s] instead.
    tensor cores idle   prologue (stages 0..depth-1). In steady state the two
                        groups' batches tile the stage back to back.
    CUDA cores idle     ~50%: one promote per group inside a two-batch stage.
                        Fine -- each group's promote is covered by the other's
                        batch, which is exactly what the column labels show.

L4 ------------------------------------- instructions and threads, one stage
      for ki in range(0, 128, 32):            # iter: start 0, stop BLOCK_K=128, step 32, trip 4
        wgmma.m64n128k32(
          A = A_s[s][64w:64w+64, ki:ki+32]    smem-desc, k-major
          B = B_s[s][ki:ki+32, 0:128]ᵀ        smem-desc, k-major
          C = C_s[64, 128]                    f32 RF, 64*128/128 = 64 elems/thread
          clear = (ki == 0) )                 # ScaleOut::Zero on iter 0, accumulate on 1..3

  PER-THREAD ACCESS, every gmem/smem touch in the stage
    A_s, B_s    NO per-thread access -- the copy engine writes smem. One elected
                thread issues one cp.async.bulk.tensor per operand, which is why
                the producer needs so few registers. Per-thread rows are for
                what THREADS touch: the wgmma smem reads, and the epilogue.
    sfa load     32 b/thread, broadcast within a quad -> 1 transaction
    smem A read 128 B swizzle atom, aligned -> 0-way bank conflict
    addressing  tile base computed once in the prologue, carried in a register;
                the stage index is the only per-iteration arithmetic (1 IADD)
```

The `=` on `C_s` beside the `+=` on `D_acc` is the whole design in two lines: the
tensor-core accumulator is stage-local because the scales change every K block,
so a second fp32 accumulator carries the mainloop. Every bound at L4 traces to
L2 and every name at L2 traces to L1 — a number that cannot be traced upward
means a field is missing.

---

# Phase 0 — Calibrate the machine

**Before any tile is chosen, measure what the machine costs.** Not the
datasheet — the machine, at this shape, through the harness that will later
decide whether the kernel is accepted. Phase 0 is roughly an hour of jobs and it
is the cheapest hour in the project: every number it produces becomes a
denominator in Phase 1, and a denominator that came from a spec sheet is how a
spec ends up targeting a number no implementation can reach.

Five measurements. **1-4 and the barrier half of 5 are shape-independent (0a):
run them once per machine, before any tile is chosen. The occupancy half of 5
(0b) needs the real smem and register budget, so it re-runs after L2 fixes the
tile and before you commit to it.**
None needs the kernel to exist. Run them through **the harness the kernel will be
accepted under** — see the `benchmark-kernel` skill, and
`references/example-phase0.md` for a full worked run with real numbers.

| # | Measure | Why it decides the design |
|---|---|---|
| 1 | **Empty-kernel cost vs grid size.** A kernel with an empty body at the real CTA count and smem. | This is the per-launch floor. It is *not* removed by a CUDA graph — it is grid ramp. If it is 1.3 µs and your kernel is 5 µs, a quarter of your budget is gone before the first byte moves, and **launch count becomes a first-class term in every fusion decision.** |
| 2 | **Cold streaming bandwidth vs transfer size.** A pure read of 1×, 2×, 4×, 8× your weight footprint. | Fit `t = a + MB/b`. `b` is the *marginal* bandwidth you can actually reach (often 60–85 % of peak) and `a` folds in ramp. `bytes / peak_BW` is not a floor; it is a fantasy. |
| 3 | **Bandwidth vs CTA count** at the real footprint. | Tells you the smallest grid that reaches bandwidth. This is what justifies (or kills) split-K, finer tiles, and persistent grids — and the curve usually flattens well before "one CTA per SM". |
| 4 | **The best existing implementation on the exact shape.** cuBLAS / cuDNN / the library kernel you are replacing. | Two uses, neither of them a bound — see the note below. It calibrates the floor model, and it measures your implementation slack. |
| 5 | **Sync and placement primitives**: cluster barrier cost at each cluster size, and `cudaOccupancyMaxActiveClusters` / `...MaxActiveBlocksPerMultiprocessor` at the real smem and register budget. | Cluster placement is capped by the *occupancy query*, not the SM count, so a grid that looks like one wave can silently be two. Barriers cost differently at the start and end of a kernel. |

**Measurement 4 is not a bound, and a target below it is not a bug.** A library
kernel solves a strictly harder problem — any shape, any layout, a fixed I/O
contract — while a specialised kernel solves an easier one and fuses across
boundaries the library cannot cross. Going below it is the point. (Done here: a
fused FFN down-projection at 7.94 µs against cuBLAS's 8.65 µs, same shape `[MEAS-C]`.)
Its two real uses:

- **calibrating the floor model** — a well-tuned library kernel should land near
  `a + MB/b`. Far *below* it means the model is too pessimistic and every target
  derived from it is wrong in the safe-looking direction;
- **measuring implementation slack** — when the library beats you on *identical
  bytes*, a harder problem is being solved better than your easier one.

For a **fused** kernel the honest reference is not one library call but the
**composition it replaces**: every call, plus the intermediates that round-trip
through HBM, plus one launch each. Against a single call the fusion looks worse
than it is.

A fitted `a + MB/b` says *what* the machine costs, never *why*. An architectural
model with per-level cache latency and per-instruction throughput would attribute
a miss instead — `example-deepgemm.md` carries one, DeepGEMM's own. Calibrate any
such model against measurements 1-3: an unchecked model is this phase's failure
one level up.

Write the results into the spec's `toolchain` block with the job ids. Phase 1
then derives every floor and every target from **these** numbers, and each is
traceable to a measurement rather than to a peak.

**Two traps Phase 0 exists to prevent, both observed:**

- **Utilisation counters do not prove headroom is free.** A kernel can sit at
  26 % DRAM and 32 % L1 and still pay a *linear* 0.23 µs per MB of L2 traffic `[MEAS-A]`.
  "Nothing is saturated, so this extra traffic is absorbed" is not an argument;
  a slope measurement is.
- **A knob's slope measured on one kernel body does not transfer to another.**
  Pipeline depth was worth 1.6× on one body and *negative* on another, because
  one was below the knee and one above. Re-measure per body.

---

# Phase 1 — Write the spec

## Where it goes

Default `specs/tile/<kernel-name>.md`, created if absent. If the repo already
has a convention for design docs, follow that instead. One spec per kernel;
name it after the kernel, not after the task.

Start from `assets/spec-template.md`. Every field is either filled, marked
`TODO`, or explicitly deleted with a one-line reason (`# no cluster: sm80`).
A silently missing field is the bug this format is designed to make impossible.

Read `references/spec-schema.md` for what each field means and which are
required per architecture. It also carries the arch capability table — read it
before the first question, because it decides *which questions even apply*
(there is no point asking about warp specialization on sm80, or about TMEM
below sm100).

## How to interview

The human on the other side knows the answer to most of these questions and
does not want to be quizzed. Interviewing well means minimizing the number of
answers they have to invent.

**Batch by section, not by field.** Ask about the grid, get an answer, then ask
about the mainloop. Five questions in one message about one layer is fine; one
question at a time across twenty messages is not.

**Propose, do not quiz.** Almost every field has a defensible default derivable
from the arch, the shapes, and the fields already settled. Lead with it and
show the arithmetic, so the reply is "yes" or a correction rather than an
essay:

> Bad: "How many pipeline stages do you want?"
>
> Good: "I'd take `depth: 5`. Per stage is A 128×128 fp8 (16 KB) + B 128×128
> fp8 (16 KB) + scales (0.6 KB) = 32.6 KB, so 5 stages is 163 KB, leaving
> 64.4 KB of the 227 KB cap for the 128×128 fp32 epilogue buffer (64 KB) —
> **0.4 KB spare, which is knife-edge**. depth 6 needs 196 KB and does not fit.
> Confirm 5, or drop BLOCK_N to 64 and go deeper?"

**Derive rather than ask, and show the derivation.** If a value follows from
what is already settled, compute it and label it derived. Numbers to derive,
never ask: per-stage smem bytes, total smem, accumulator registers per thread,
MMA instruction count per stage, mainloop trip count, threads per CTA, waves
per grid.

**Ask when the answer encodes intent you cannot see.** The reduction axis when
there is more than one candidate, the fusion boundary, what is dynamic at
runtime versus baked in as a constant, the numerical contract (accumulate in
fp32? where does dequant scaling apply — every K-block or once at the end?),
whether the target regime is latency at tiny M or throughput at large M. These
change the kernel and cannot be guessed from shapes.

**Ask again when an answer contradicts another.** Contradictions are the most
valuable thing you will find. Report them as arithmetic, not as doubt:
"BLOCK_M=128 with a single math warp group puts 128 accumulator registers per
thread plus operands; that overruns the 255-register budget. Either two math
warp groups (64 rows each, matching wgmma M=64), or BLOCK_M=64. Which?"

**Never invent a number to look complete.** `TODO` plus a note on what would
settle it is strictly better than a plausible fabrication, because the reviewer
will trust the fabrication.

`references/interview.md` holds the question bank organized by section and by
kernel archetype (GEMM, attention/flash, grouped/MoE, reduction/normalization),
including the questions people forget. Consult it when the kernel is not a
plain GEMM, or when a section feels thin.

## Persistent, cooperative, clusters — three decisions, not one

| | Is | Needs |
|---|---|---|
| **persistent** | grid sized by the machine, each CTA loops over tiles | a tile scheduler, nothing else |
| **residency** | every CTA simultaneously live — what licenses a CTA to *wait on* another | cooperative launch |
| **clusters** | a small group co-scheduled in a GPC, sharing smem and multicast | neither of the above |

**Grid = `SM_count × cta_per_sm`, not `SM_count`.** `cta_per_sm` comes from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` and is capped by smem **and**
registers — read `Block Limit Registers`, not only `Block Limit Shared Mem`.
Conversely, capacity is not residency: 128 CTAs on 132 SMs land one per SM
whatever the query says, so freeing smem to reach 3 CTAs/SM buys nothing until
the grid reaches `3 × SM_count`. Only the grid converts capacity into warps.

### Cooperative: one guarantee, three use cases

The guarantee is residency, and nothing else. It matters because the GPU does
not preempt within a kernel: if CTA 500 is unscheduled and CTA 0 spins on it,
the failure is a **hang**, not an error.

At `SM_count × cta_per_sm` residency already holds in practice; cooperative only
makes it *checked* — the launch fails instead of hanging after someone's edit
pushes registers over a threshold. A safety property, not a reason on its own.

| Worth it for | Because two launches cannot |
|---|---|
| **cross-phase state in registers/smem** | a launch boundary destroys both. If an accumulator or resident tile would otherwise round-trip through HBM, a grid barrier keeps it. **Usually the only argument that pays** — cost it in bytes against ~one launch |
| device-decided phase counts | a data-dependent trip count cannot be unrolled into launches without a host round-trip |
| global work stealing | a CTA taking from a queue needs every producer live |

Not worth it for *"fewer launches"* — a grid barrier is *believed* to cost about
what a launch costs `[I, UNMEASURED]`, so only retained state wins. **Check this
before relying on it**: Phase 0's measurement 1 gives the launch cost and 5 gives
the *cluster* barrier, but nothing here has costed a grid barrier. Also not worth
it for *"more persistent"*, a grid-sizing decision and orthogonal.

**Cooperative and clusters are not mutually exclusive — but together they need an
explicit placement check.** A cooperative launch is rejected with
`cudaErrorCooperativeLaunchTooLarge` when its grid exceeds what can be
co-resident, and with a cluster that ceiling is
`cudaOccupancyMaxActiveClusters × cluster_size`, not `SM_count × cta_per_sm`.
Measured here `[MEAS-B]`: cluster 8 at 207360 B gives 15 placeable clusters
against the 16 the grid needed, so the pair was rejected — a failure of *that
footprint*, not a prohibition. Deep pipelines make it the common case on sm90,
so budget for it; at cluster 2 with modest smem the pair launches.
Both are individually capturable into a CUDA graph. Since multicast usually beats retained state, the
default for a static fused chain is still **persistent yes, cooperative no** —
but as a default, not because the hardware forbids the pair.

### CTA count hides barriers; warp count does not

A warp stalled on DRAM is hidden by any eligible warp, including one in the same
CTA. **A warp stalled on a block-level barrier is not** — `__syncthreads`, an
mbarrier wait, a producer/consumer handoff stop *every warp in the CTA at once*.
Only **another CTA**, whose barriers do not align with yours, fills that gap.

So at equal warps per SM, **more CTAs of fewer warps beats fewer CTAs of more
warps** whenever barriers sit on the critical path — which every software-pipelined
fused kernel has, once per stage. Aim for `cta_per_sm` 2 or 4 when latency-bound.

Measured: 1 CTA/SM at 8 warps gave **0.30 eligible warps per scheduler**, top
stall `short scoreboard` at 42.7%. Widening the CTA would not have helped.

**This opposes the warp-specialisation idiom**, and the tension should be
resolved explicitly. Producer/consumer groups want large CTAs and deep pipelines
want large per-CTA smem; both drive `cta_per_sm` to 1. Right at large M, usually
wrong at small M — four 128-thread CTAs at 56 KB can beat one 256-thread CTA at
204 KB.

**Two orthogonal knobs, and they buy different things.** `cta_per_sm` hides
**barriers**, as above. `warps_per_cta` hides **memory latency** — more eligible
warps *between* barriers, which is what covers a DRAM miss. A latency-bound
kernel wants both, and neither is capped by the other: raise total warps toward
the SM's 64 and let the register and smem budgets say where you stop.

H100 per SM: 233472 B smem, 65536 registers, **64 warps / 2048 threads**, 32 CTAs.
The budget is joint, so read the row you want and check all three columns:

| `cta_per_sm` × threads/CTA | warps/SM | regs/thread | smem/CTA |
|---|---|---|---|
| 1 × 256 | 8 | 255 | 227 KB |
| 2 × 256 | 16 | 128 | 114 KB |
| 4 × 256 | 32 | 64 | 57 KB |
| 2 × 512 | 32 | 64 | 114 KB |
| 4 × 512 | 64 | 32 | 57 KB |
| 8 × 256 | 64 | 32 | 28 KB |

`regs/thread = 65536 / (cta_per_sm × threads_per_cta)`, so **the real trade is
warps against accumulator size**: 64 warps/SM leaves 32 registers per thread,
which a 64-f32-per-thread accumulator alone already exceeds. Small-M kernels are
the ones that can afford it — an accumulator of `BLOCK_M×BLOCK_N/threads` is
16 f32/thread at 64×32 over 128 threads, so ~64 regs/thread is comfortable and
32 warps/SM is reachable. Large-tile throughput kernels cannot go there and
should not try.

### Size the producer; do not inherit 128

**wgmma** needs whole 128-thread warp groups — that binds the *math* side.
**TMA does not**: `cp.async.bulk.tensor` is issued by a single elected thread
(`cute::elect_one_sync()`, `sm90_mma_tma_gmma_ss_warpspecialized.hpp:320`), so a
producer can be **one warp**. CUTLASS producers are 128 threads because
`setmaxnreg.{inc,dec}.sync.**aligned**.u32` is warpgroup-granular
(`arch/reg_reconfig.h:76,84`) — a register-redistribution constraint, not a TMA
one. So `threads_per_cta` moves in steps of **32**, with the math side a
multiple of 128.

Two reasons to spend more than one warp on the producer, and you should name
which applies:

- **you want `setmaxnreg`** — to run the producer at 24-40 registers so the math
  groups can hold 232-240. Needs the producer to be a full warp group;
- **the producer does real work** — an in-mainloop transform, a squares
  reduction, smem staging. That is CUDA-core work and wants warps — and it is
  what puts the producer on L3's critical path.

If neither applies, one warp is enough, and the saving is not just 96 threads:
**it converts register budget into CTA count, which is what hides barriers.** At
128 registers per thread, a 256-thread CTA fits 2 per SM (16 warps) while a
160-thread CTA fits 3 (15 warps) — nearly the same warps, but three independent
barrier schedules instead of two. (Registers allocate in units of 8 per thread,
so round down: 65536/(4×160) = 102.4 becomes **96**.)

One cost of a lone producer warp: `setmaxnreg` becomes unusable **for the whole
CTA**, not just the producer — PTX requires every warp of a warpgroup to execute
it, and the register pool is per-CTA.

| variant | threads | warps/SM @4 CTAs | regs/thread @4 CTAs |
|---|---|---|---|
| producer WG + math WG | 256 | 32 | 64 |
| producer warp + math WG | 160 | 20 | 96 |
| producer warp + 2 math WGs | 288 | 36 | 56 |

The costs are real: smaller tiles re-read the shared operand more (price it at
the Phase 0 L2 slope), and less smem caps depth — which only pays to the knee
anyway. A spec that raises either knob without naming what it gave up has made
half a trade.

### Ordering between phases

**The mechanism is a consequence of the tile shape, not an independent choice.**
An unsplit reduction over the full K makes every consumer depend on every
producer — only a grid barrier serves that, and it costs about a launch, so just
launch twice. Split-K makes each consumer depend on `1/splits` of the producers,
and then per-tile semaphores are far cheaper.

Inside a captured graph: a semaphore must be **self-resetting** (replay reruns
identical nodes with identical arguments and there is no host reset in the replay
path, so the last CTA through clears the counters), and anything derived from a
launch-time counter must be a kernel **parameter**, not captured state.

### A static offline schedule

- **No work queue, no atomic.** Claiming a tile is index arithmetic or a
  constant-memory lookup. Say so; a reviewer should not have to guess.
- **The schedule must respect the cluster shape.** Co-clustered CTAs must be
  assigned tiles that actually share the multicast operand. Cluster boundaries
  constrain the offline optimiser; getting it wrong turns multicast into a
  broadcast nobody needs.

## Tile order is an L2 decision

`grid.rasterization` decides the order the machine walks the weight slab and how
much of the shared operand is co-resident in L2. `row-major` is a default, not a
decision; label it as one. Three questions, in increasing order of how often they
are skipped:

1. **Which operand do consecutive CTAs share, and how much fits?** A panel
   sweeping one A tile against many B tiles turns N re-reads into one — while the
   panel's working set stays in L2. Panel width is `L2_bytes /
   shared_operand_bytes`: a number, not a taste.
2. **What DRAM read order does the map produce?** Tidy in tile space can be
   scattered in address space. Sequential sweeps get row-buffer locality.
3. **What happens at the seam between kernels?** In a fused chain the tail of *i*
   overlaps the head of *i+1*; whether *i+1*'s first tiles are warm is a property
   of the pair, and it is what dependent launch exists to exploit.

**On a static graph, solve it instead of guessing.** Fixed shapes and a fixed
layer order mean the CTA→tile map is an offline optimisation with an optimum, not
a runtime heuristic. Bake it in as index arithmetic or a constant-memory table,
and record whether it was **solved or defaulted** — "defaulted to linear" is an
acceptable answer and an unacceptable omission. Validate with
`lts__t_sector_hit_rate` and `dram__bytes_read`: measured DRAM bytes above the
unique-bytes count means the map is re-fetching.

**The lever's size depends on there being reuse.** At small M, where each weight
element is read once and weights dominate, intra-kernel L2 reuse is ~nil and the
lever is the activation re-read plus the seam. Measure the slope first.

## Consistency arithmetic — run this before every hand-back

Most under-specification shows up as an equation that does not balance, and
finding it yourself is worth more than another round of questions. Compute all
of these, put the results in the spec's `checks` block, and lead the hand-back
with any that fail:

| Check | Must hold |
|---|---|
| smem | `depth × per_stage_bytes + non_staged_bytes ≤ arch smem/CTA` (232448 B = 227 KB per CTA on sm90/sm100 -- the per-SM figure is 233472 B and is a different number, 164 KB sm80) — and if aliasing is used, say which buffers alias and why that is safe |
| threads | `sum(warp_groups.threads) == launch threads`, each group a multiple of 128 when it issues wgmma |
| acc_registers | fp32 accumulator elems/thread `= cta_tile.M × cta_tile.N / math_threads`; plus operands and addressing must fit 255 (sm90 RF) — or live in TMEM (sm100) |
| mma_k | `iters_per_stage × inst_shape.K == mainloop.step` |
| mma_m | `inst_shape.M × num_math_groups == cta_tile.M` (or state the split rule that replaces it) |
| mma_n_legal | `inst_shape.N` is a legal atom for the arch/dtype (wgmma N ∈ {8,16,…,256} step 8) |
| trip_count | `mainloop.trip_count × mainloop.step ≥ problem reduction extent`, and the tail policy is named |
| output_coverage | grid tiles × cta_tile covers the output exactly; predication/masking named for ragged edges |
| occupancy | `cta_per_sm` implied by smem and registers matches what the spec claims; for persistent grids `ctas == num_sms × cta_per_sm` |
| barrier_arrivals | every stage buffer has a full and an empty barrier (or the stated equivalent); arrival counts match the number of arriving warps/CTAs; the phase-flip rule is written down |
| traceability | every bound and extent at L4 traces to an L2 loop, every name at L2 traces to an L1 dim, and the shared inner name of each `@` is `mainloop.axis` — an untraceable number means a missing field |
| loop_bounds | every `range` in the nest states start, stop, and step, and its trip count matches the corresponding YAML field (`mainloop.trip_count`, `math.count_per_stage`) |
| arithmetic_intensity | tile arithmetic intensity `≈ 2·M·N·K / (bytes moved)` versus the arch ridge point — flags a tile that cannot reach peak no matter how good the code is |
| **floor** | every `perf_target` ≥ the Phase 0 floor for that kernel, computed as `a + MB/b` from measurement 2 **plus one launch cost per kernel launch** — a target below its own floor makes the whole spec unfalsifiable |
| **reference** | measurement 4 is used to *calibrate the floor model*, never as a bound. See Phase 0 measurement 4 |
| **acceptance** | the spec names the *one* measurement that decides acceptance, and it is the one the kernel will ship under. Designing against an isolated cold benchmark and shipping against an in-graph profile is a 2× discrepancy waiting to happen |
| **falsifiability** | every performance claim in `Why these numbers` names the measurement that would refute it. A claim with no refuting measurement is an assumption wearing a number |
| **concurrency** | L3's bubble check is filled in: for a steady-state stage, each of the three engines is either busy or its idle time is named and accepted. |
| **vectorisation** | every gmem and smem touch at L4 states bits/thread and transactions; each is the widest legal access (128 b where alignment allows), coalesced, and bank-conflict-free — or the exception is named with its reason |
| **addressing** | per-iteration address arithmetic is counted. Anything loop-invariant is hoisted to the prologue and carried in a register; if the mainloop recomputes a base each stage, say why |
| **register_budget** | `threads_per_cta × regs_per_thread × cta_per_sm ≤ 65536`, computed. This is what forces `cta_per_sm` in practice, and `acc_registers` (per thread) does not cover it |
| **residency** | `cta_per_sm` is stated with the smem *and* register arithmetic that produces it, and for a latency-bound kernel a value of 1 is justified against the 2-or-4 alternative rather than inherited from the warp-specialisation idiom |
| **persistence** | if `mode: persistent`, the grid is at least `SM_count x cta_per_sm` or the shortfall is named; `cooperative` is `true` only where CTAs block on each other, and when combined with a cluster the grid is checked against `cudaOccupancyMaxActiveClusters x cluster_size`; any semaphore is self-resetting under graph replay |
| **tile_order** | `grid.rasterization` carries an L2 argument, not just a name, and on a static graph `grid.l2_schedule` says whether the map was solved offline or defaulted |
| **non_mma_accounting** | every `non_mma` entry appears in L3's CUDA-core column with its `cost`, and every name in its `loop_carried` also appears in `mainloop.loop_carried`. Any entry with `on_critical_path: yes` shows what the copy engine is doing during it |
| **rounding_contract** | every `non_mma.dtype` says where each rounding lands, and the parity reference in `verification` mirrors it. An algebraically identical op at a different precision is a different function and will not compare |

When a check fails, the spec is wrong, not the check. Fix the spec or ask.

## Ending Phase 1

The spec is ready to hand over when every field is filled or explicitly
deleted, the L1-L4 nest is complete with explicit bounds and every
number in it traces upward, `open_questions` is empty, and every consistency check passes. Set
`status: review`, then **stop and hand back**. In the hand-back message give:

1. The path to the spec.
2. The shape in one line: `grid × cta_tile @ cta_per_sm / mainloop / stages /
   iters`, and the L2 compute line — what an expert can sanity-check without
   opening the file.
3. **The floor and the target, with the Phase 0 numbers they came from**, so the
   reviewer can see the target is reachable before reading anything else.
4. The consistency check results — especially anything tight (smem within a few
   KB of the cap, registers near 255, `cta_per_sm` at 1 on a latency-bound
   kernel) and **L3's bubble check**, which is the one a reviewer of a fused
   kernel should read first.
5. The decisions you made on the human's behalf and the reasoning, so they know
   where to look first.
6. Anything you could not settle, phrased as a decision rather than a question.

Then stop. Do not begin Phase 2 in the same turn, and do not ask "shall I
implement it now?" — the reviewer needs to read the spec before that question
means anything.

Phase 2 starts when the human approves: they set `status: approved` themselves,
or they say so in the conversation, in which case set `status: approved` with
their name in `approved_by` before writing any code. Corrections during review
go into the spec first, then into the code — the spec is the source of truth
from that point on, not the transcript.

---

# Phase 2 — Generate the kernel

## Pick the backend first

Ask which backend, and generate exactly one. Generating three implementations
of an unproven spec triples the review burden and the debugging surface for no
information gain — the second implementation is cheap only *after* the first
one is correct.

| Backend | Fits when |
|---|---|
| **TileLang** | Fastest path to a working kernel; the compiler owns pipelining and the producer/consumer split. Best when the spec's stage structure is regular. |
| **CUTLASS / CuTe (C++)** | Production Hopper/Blackwell GEMM shapes, epilogue fusion, when the collective builders already express the schedule. |
| **CuTeDSL (Python)** | CuTe layout algebra and explicit wgmma/tcgen05 control without the C++ template build cycle. |
| **Raw CUDA + inline PTX** | The schedule is irregular enough that no library expresses it — asymmetric warp roles, hand-placed barriers, smem aliasing. Most control, most work. |

If the user has no preference, recommend one from the spec: an irregular warp
schedule (like FlashMLA's seesaw) argues against TileLang; a textbook staged
GEMM argues for it; an existing CUTLASS-based codebase argues for CuTe.

Read the matching section of `references/backends.md` — it maps every spec
field onto that backend's construct, and lists what that backend *cannot*
express, which is where a spec quietly stops being implementable.

Defer to the deeper skills for API detail once the mapping is clear:
`cutlass_skill` for CUTLASS/CuTe/CuTeDSL, `cuda_skill` for PTX and profiling,
`triton_skill` for Triton/Gluon.

## Rules while generating

**The spec is the contract.** Every tile size, stage count, and instruction
count in the code comes from the spec. When the code needs a number the spec
does not have, that is a Phase 1 defect: add the field to the spec, tell the
user what you added and why, and continue.

**Diverge loudly.** If the backend cannot express something the way the spec
says — TileLang choosing its own stage placement, a CUTLASS builder overriding
the cluster shape — say so, record the deviation in the spec's `deviations`
block, and keep the spec's intent as the comment next to the code. A silent
divergence turns the spec into a lie for whoever reads it next.

**Comment against spec sections.** `# mainloop, stage k%5` beats a restatement
of what the line does. It lets the reviewer diff code against spec by eye.

**State how it was verified.** Say plainly what you ran: compiled only, ran
against a reference, benchmarked. Numbers without a named measurement method
are noise — this repo's `benchmark-kernel` skill covers per-kernel timing.

---

# Reverse-engineering an existing kernel

Same format, opposite direction: read the source and fill the spec from it.
The output is a spec whose numbers are all cited to source lines, so a reviewer
can check them. `open_questions` here means "the source does not make this
obvious", not "the human has not decided". Both worked examples
(`references/example-deepgemm.md`, `references/example-flashmla.md`) were
produced this way and show the citation style.

Mark every claim's provenance, because a reverse-engineered spec that reads as
uniformly authoritative is wrong in places nobody can check: **`[D]`** derived
from cited source, **`[I]`** inferred from how the hardware must work, and
`TODO — needs source` when neither. Do not fabricate line numbers. A line
citation *is* `[D]`, so `[I]` is only needed on the unciteable.

**Use `status: reference`, not `approved`.** There is no human sign-off to record
and no Phase 2 to unblock — the spec documents a kernel that already exists.
`draft → review → approved` is the new-kernel flow; a reverse-engineered spec
sits outside it, and `open_questions` may stay non-empty because here the term
means "the source does not settle this".

**Name buffers in the nest with the source's own identifiers**, not the generic
`A_s` / `C_s` of the Shape example. That is what makes `traceability` mechanically
checkable against code instead of a reading exercise, and it closes the naming
gap without an alias field.

This is also the best way to start a *new* kernel that resembles a known one:
extract the known kernel's spec, then edit it with the human.

---

# Where this skill's own numbers come from

This skill holds specs to `[D]` / `[I]` / `TODO — needs source`; it owes the same.
Every figure above is from one project — an H100 SXM5 Pi0.5 decoder on this
cluster, torch 2.11.0+cu130, clocks **unpinnable** so ~6% is the noise floor.
One machine's numbers, not constants: **re-measure before porting them.**

| tag | what | source |
|---|---|---|
| `[MEAS-A]` | 0.30 eligible warps/sched, `short scoreboard` 42.7%, transform 24-27% of cycles, the 0.8 us TMA hoist, 0.23 µs/MB of L2 activation re-read | `ncu --set full` plus A/B rebuilds of **one** fused FFN kernel. Where it appears twice above, that is one finding cited twice, not two corroborating ones |
| `[MEAS-B]` | cooperative + cluster rejected on sm90 | probe kernel, cluster 8 at 207360 B: `cudaOccupancyMaxActiveClusters` returns 15 against the 16 the grid needs, and cooperative+cluster returns `cudaErrorCooperativeLaunchTooLarge`. Reproduced behaviourally — 120 blocks reach a spin barrier, 128 time out |
| `[MEAS-C]` | 7.94 µs against cuBLAS 8.65 µs | split-K gated residual, M=50 K=4096 N=1024 bf16, both timed in one process, cold |
| `[I, UNMEASURED]` | grid barrier ≈ launch cost | inference. Load-bearing for rejecting cooperative and **not measured** — the claim here a reader should check first |
| `[I]` | `~700 ns` HBM latency in the L3 example | a round figure for illustration |

The depth knee (`4→8 = 10.38→6.67 us` on one body, `8→11 = 14.45→14.99` on
another) is two kernels from that project — which is the rule's own point: a
slope does not transfer between bodies.

# Reference files

| File | Read when |
|---|---|
| `assets/spec-template.md` | Always — Phase 1 starts by copying it |
| `references/spec-schema.md` | Always — field meanings, required-per-arch, arch capability table |
| `references/example-phase0.md` | Running Phase 0 — a full worked calibration with real numbers, the 0a/0b split, and what it does *not* give you |
| `references/primitives.md` | The kernel contains a softmax, an online/streaming reduction, a fused norm, or a cross-CTA split reduction — named contracts with their state and hazards |
| `references/interview.md` | The kernel is not a plain GEMM, or a section feels thin |
| `references/backends.md` | Phase 2, after the backend is chosen |
| `references/example-deepgemm.md` | A GEMM, or a producer/consumer warp split — SM90 FP8, persistent, TMA multicast, per-K-block dequant |
| `references/example-flashmla.md` | Attention/decode, or a non-producer/consumer warp split — SM90 MLA, two math warp groups in a seesaw, split-KV |
