---
name: tile-dataflow
description: Design a GPU kernel in two phases — first write a reviewable tile-level dataflow spec (CTA tiling, mainloop, pipeline stages, warp specialization, per-instruction mma/wgmma/tcgen05 iters), get human sign-off, then generate the kernel in TileLang, CuTeDSL, CUTLASS/CuTe, or raw CUDA/PTX. Use this whenever the user wants to write, design, port, or plan a GPU kernel — GEMM, attention, MoE, or a fused op — or mentions tile/block sizes, pipeline stages, warp specialization, producer/consumer warp groups, wgmma/TMA/mbarrier scheduling, or wants the dataflow described before any code is written. Also use when asked to reverse-engineer an existing kernel (DeepGEMM, FlashMLA, FlashAttention, CUTLASS collectives) into a tile-level description, or to port a kernel between TileLang / CUTLASS / CuTeDSL / CUDA.
---

# Tile-Level Dataflow → Kernel

A kernel that gets written before its dataflow is pinned down gets rewritten.
The expensive decisions — how the output is tiled across CTAs, how deep the
pipeline is, which warps load and which warps compute, how many MMA
instructions fire per stage — are all made in the first ten minutes and are
all painful to change once there is code. So this skill splits the work:

**Phase 1** produces a spec that a human kernel engineer can review and correct
in minutes. **Phase 2** turns the approved spec into code. A hard gate sits
between them: *no kernel code is written until a human signs off on the spec.*

The gate is the entire point. Do not soften it, do not "start a draft while we
discuss", do not write "illustrative" code in Phase 1. A plausible-looking
kernel invites the reviewer to review the code instead of the dataflow, which
is the failure mode this skill exists to prevent.

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
that decide performance. One mainloop iteration occupies one pipeline stage:
one tile-granular load and one tile-granular compute. Inside that one compute
there are typically several MMA instructions, because the hardware MMA shape is
smaller than the tile — that count is the `iter` granularity. Say both numbers
explicitly, every time.

Warp specialization cuts across all four levels: on Hopper and Blackwell the
CTA's warps are split by *role* (a producer warp group issuing TMA, one or more
math warp groups issuing MMA, sometimes a separate epilogue group), and the
stage barriers are what couple them. A spec that names stages but not the warp
groups that arrive on them is incomplete.

## The three levels

The loop nest above says *when* things happen. It does not say what **shape**
anything is, nor where a loop starts, stops, and how far it steps — and that is
what a reviewer actually checks. So every spec carries a loop-structured
pseudocode block at three levels, each a refinement of the one above it:

| Level | Is | Answers |
|---|---|---|
| **L1 — kernel** | the iteration space, no hardware in it | What crosses the kernel boundary, and what loops cover the whole problem. Which loops are parallel and which are the contraction. |
| **L2 — CTA and stage** | the same nest, mapped to hardware | Which loops become the grid, which stays serial inside a CTA, what one stage loads and computes, with tile shapes and byte counts. |
| **L3 — iters** | the innermost body, expanded | The MMA instruction, its `m x n x k`, the iter loop's bounds, and where A, B, C live. |

Reading L1 → L2 → L3 is reading the same computation at three zoom levels, so a
reviewer can check the interface, then the tiling, then the instruction
sequence, and see that each follows from the last.

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
```

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

A scaled FP8 GEMM, complete at all three levels:

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

L3 ------------------------------------------------- innermost body, expanded
      for ki in range(0, 128, 32):            # iter: start 0, stop BLOCK_K=128, step 32, trip 4
        wgmma.m64n128k32(
          A = A_s[s][64w:64w+64, ki:ki+32]    smem-desc, k-major
          B = B_s[s][ki:ki+32, 0:128]ᵀ        smem-desc, k-major
          C = C_s[64, 128]                    f32 RF, 64*128/128 = 64 elems/thread
          clear = (ki == 0) )                 # ScaleOut::Zero on iter 0, accumulate on 1..3
```

The `=` on `C_s` beside the `+=` on `D_acc` is the whole design in two lines: the
tensor-core accumulator is stage-local because the scales change every K block,
so a second fp32 accumulator carries the mainloop. Every bound at L3 traces to
L2 and every name at L2 traces to L1 — a number that cannot be traced upward
means a field is missing.

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
> 64 KB of the 228 KB for the 128×128 fp32 epilogue buffer (64 KB) with 1 KB
> spare. depth 6 would need 196 KB and does not fit. Confirm 5, or drop
> BLOCK_N to 64 and go deeper?"

**Derive rather than ask, and show the derivation.** If a value follows from
what is already settled, compute it and label it derived. Numbers to derive,
never ask: per-stage smem bytes, total smem, accumulator registers per thread,
MMA instruction count per stage, mainloop trip count, threads per CTA, waves
per grid. Asking for these wastes the reviewer's attention on arithmetic and
hides the one number they actually needed to think about.

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

## Consistency arithmetic — run this before every hand-back

Most under-specification shows up as an equation that does not balance, and
finding it yourself is worth more than another round of questions. Compute all
of these, put the results in the spec's `checks` block, and lead the hand-back
with any that fail:

| Check | Must hold |
|---|---|
| smem | `depth × per_stage_bytes + non_staged_bytes ≤ arch smem/CTA` (228 KB on sm90/sm100, 164 KB sm80) — and if aliasing is used, say which buffers alias and why that is safe |
| threads | `sum(warp_groups.threads) == launch threads`, each group a multiple of 128 when it issues wgmma |
| acc registers | fp32 accumulator elems/thread `= cta_tile.M × cta_tile.N / math_threads`; plus operands and addressing must fit 255 (sm90 RF) — or live in TMEM (sm100) |
| MMA K | `iters_per_stage × inst_shape.K == mainloop.step` |
| MMA M | `inst_shape.M × num_math_groups == cta_tile.M` (or state the split rule that replaces it) |
| MMA N | `inst_shape.N` is a legal atom for the arch/dtype (wgmma N ∈ {8,16,…,256} step 8) |
| trip count | `mainloop.trip_count × mainloop.step ≥ problem reduction extent`, and the tail policy is named |
| coverage | grid tiles × cta_tile covers the output exactly; predication/masking named for ragged edges |
| occupancy | `cta_per_sm` implied by smem and registers matches what the spec claims; for persistent grids `ctas == num_sms × cta_per_sm` |
| barriers | every stage buffer has a full and an empty barrier (or the stated equivalent); arrival counts match the number of arriving warps/CTAs; the phase-flip rule is written down |
| traceability | every bound and extent at L3 traces to an L2 loop, every name at L2 traces to an L1 dim, and the shared inner name of each `@` is `mainloop.axis` — an untraceable number means a missing field |
| loop bounds | every `range` in the nest states start, stop, and step, and its trip count matches the corresponding YAML field (`mainloop.trip_count`, `math.count_per_stage`) |
| intensity | tile arithmetic intensity `≈ 2·M·N·K / (bytes moved)` versus the arch ridge point — flags a tile that cannot reach peak no matter how good the code is |

When a check fails, the spec is wrong, not the check. Fix the spec or ask.

## Ending Phase 1

The spec is ready to hand over when every field is filled or explicitly
deleted, the L1/L2/L3 loop nest is complete with explicit bounds and every
number in it traces upward, `open_questions` is empty, and every consistency check passes. Set
`status: review`, then **stop and hand back**. In the hand-back message give:

1. The path to the spec.
2. The four numbers in one line: `grid × cta_tile / mainloop / stages / iters`,
   and the L2 compute line — the one line an expert can sanity-check without
   opening the file.
3. The consistency check results — especially anything tight (smem within a few
   KB of the cap, registers near 255).
4. The decisions you made on the human's behalf and the reasoning, so they know
   where to look first.
5. Anything you could not settle, phrased as a decision rather than a question.

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

This is also the best way to start a *new* kernel that resembles a known one:
extract the known kernel's spec, then edit it with the human.

---

# Reference files

| File | Read when |
|---|---|
| `assets/spec-template.md` | Always — Phase 1 starts by copying it |
| `references/spec-schema.md` | Always — field meanings, required-per-arch, arch capability table |
| `references/interview.md` | The kernel is not a plain GEMM, or a section feels thin |
| `references/backends.md` | Phase 2, after the backend is chosen |
| `references/example-deepgemm.md` | A GEMM, or a producer/consumer warp split — SM90 FP8, persistent, TMA multicast, per-K-block dequant |
| `references/example-flashmla.md` | Attention/decode, or a non-producer/consumer warp split — SM90 MLA, two math warp groups in a seesaw, split-KV |
