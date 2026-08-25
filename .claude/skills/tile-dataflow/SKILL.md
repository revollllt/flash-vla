---
name: tile-dataflow
description: Design a GPU kernel in three phases — first measure the machine's real cost model, then write a reviewable dataflow spec (CTA tiling, residency, mainloop, pipeline stages, the per-stage schedule across copy engine / CUDA cores / tensor cores, and the lowered per-thread accesses), get human sign-off, then generate the kernel in TileLang, CuTeDSL, CUTLASS/CuTe, or raw CUDA/PTX. Use this whenever the user wants to write, design, port, or plan a GPU kernel — GEMM, attention, MoE, or a fused op — or mentions tile/block sizes, pipeline stages, warp specialization, producer/consumer warp groups, wgmma/TMA/mbarrier scheduling, or wants the dataflow described before any code is written. Also use when asked to reverse-engineer an existing kernel (DeepGEMM, FlashMLA, FlashAttention, CUTLASS collectives) into a tile-level description, or to port a kernel between TileLang / CUTLASS / CuTeDSL / CUDA.
---

# Tile-Level Dataflow → Kernel

A kernel written before its dataflow is pinned down gets rewritten. The expensive
decisions — how the output is tiled across CTAs, how deep the pipeline is, which
warps load and which compute, how many MMA instructions fire per stage — are all
made in the first ten minutes and all painful to change once there is code. So
the work splits into three phases with a hard gate:

- **Phase 0 — calibrate.** Measure what the machine actually costs, so every floor
  and target in the spec is a denominator someone measured, not a datasheet
  number. (The failure it prevents: floors computed as `bytes / peak_bandwidth`
  sit *below* what any kernel can reach, making every later "we missed by 2×"
  unfalsifiable.)
- **Phase 1 — spec.** Produce a dataflow spec a kernel engineer can review and
  correct in minutes.
- **Phase 2 — code.** Turn the *approved* spec into one kernel.

**The gate between 1 and 2 is the whole point: no kernel code is written until a
human signs off on the spec.** Do not soften it, do not "start a draft while we
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

`stage` and `iter` are the two levels people leave vague, and the two that decide
performance. One mainloop iteration *usually* occupies one pipeline stage; when
it does not, say the ratio (an iteration consuming two stages is a **compute
pair**, not a fill/drain pair — it changes what `depth: 2` means). Inside one
compute there are typically several MMA instructions, because the hardware MMA
shape is smaller than the tile — that count is the `iter` granularity. Say both
numbers explicitly, every time.

Warp specialization cuts across all four levels: on Hopper/Blackwell the CTA's
warps split by *role* (a producer issuing TMA, one or more math groups issuing
MMA, sometimes a separate epilogue group), and the stage barriers couple them. A
spec that names stages but not the warp groups arriving on them is incomplete.

## The four levels

Every spec carries a loop-structured pseudocode block at four levels, each a
refinement of the one above it — the loop nest says *when* things happen; these
say what **shape** anything is, and where a loop starts, stops, and steps.

| Level | Is | Answers |
|---|---|---|
| **L1 — kernel** | the iteration space, no hardware in it | What crosses the kernel boundary, what loops cover the whole problem, which are parallel and which is the contraction. |
| **L2 — CTA and stage** | the same nest, mapped to hardware | Which loops become the grid, which stays serial in a CTA, what one stage loads and computes, with tile shapes and byte counts. |
| **L3 — schedule** | one stage, three engines running concurrently | What the **async copy engine (TMA/cp.async)**, the **CUDA cores** and the **tensor cores** each issue during this stage, which edges order them, where the bubbles are. |
| **L4 — instructions and threads** | the innermost body, lowered | The MMA instruction and its `m×n×k`, the iter loop's bounds, where A/B/C live — and per thread, access width, coalescing, bank conflicts, and where the address arithmetic lives. |

**L3 decides *when*; L4 decides *what and how wide*.** A bubble is an L3 defect; a
bank conflict is an L4 defect. Keeping them in one level is what lets a spec
assert an overlap nobody can review. `references/example-shape.md` shows all four
levels filled in on one FP8 GEMM stage.

### Which level owns which resource

Shared resources are **budgeted high and realised low**.

| Resource | Budget decided at | Realisation checked at | Failure mode when only the low level names it |
|---|---|---|---|
| smem capacity | **L2** — `staged_buffers` × `depth` + non-staged | checks.smem | depth chosen, then found not to fit |
| smem **layout**: swizzle **and row stride** | **L2** — the swizzle atom, and the tile extent that sets the stride | **L4** — bank-conflict ways | the atom that suits the MMA is wrong for the epilogue reading the same buffer; an unswizzled buffer's conflicts are set by its **row stride**, so the only fix is a different tile extent — an L4 symptom whose sole cure is an L2 number |
| **registers** | **L2** — acc elems/thread, `setmaxnreg` split, `max_regs_per_thread` → `Block Limit Registers` | **L4** — spills, whether the accumulator really stays in RF | occupancy is capped by registers co-equally with smem; treat it as codegen and the grid is already wrong before a line is written |
| engine concurrency | **L3** — the timeline | L3's own bubble check | asserted, not specified |
| grid / CTA count | **L1/L2** | Phase 0 measurement 3 | a grid too small to reach bandwidth, discovered at the end |

**A resource appearing only at L4 was treated as a code-generation problem when
it was a tiling decision.**

### L3 is not optional in a fused kernel

L1 and L2 describe *dependency*; a fused kernel's thesis is *concurrency* — three
engines busy at once. A spec naming only which warp group holds which *role* has
**asserted** the overlap, not specified it. So every stage gets a three-column
timeline (copy / CUDA / tensor) with its ordering edges, and one rule catches
most defects: **find what actually gates the next copy, because it is later than
it looks** — the issue, the *release*, or a buffer wgmma has not finished reading
asynchronously. **An empty column is a bubble the spec shows you before the kernel
exists.**

The war story, the three gate forms, and how to reason about cycle counts Phase 0
did not measure (ratios between columns, never absolutes) are in
`references/schedule-l3.md`. On sm80 there is no separate copy engine — draw two
columns and say so.

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
reduce_<op>(T[i, lo:hi], axis)       a reduction with its axis and extent explicit -- never a bare
                                     "reduce"; the extent is its cost
m, l = carry(...)                    a LOOP-CARRIED reduction. The rule is broader: **any name the
                                     nest carries across the mainloop -- including a plain `+=`
                                     accumulator -- must appear in `mainloop.loop_carried`**
f(x) <dtype>                         an elementwise op annotated with the dtype it computes in --
                                     `gelu(c) f32` and `gelu(c) bf16` are different functions
```

Three things fall out for free: the **shared inner name is the contraction
axis**, so a reader finds the reduction by eye (and if it does not match
`mainloop.axis` the spec is wrong); the **`=` / `+=` split makes
`accumulate_across_iters` visible**; and **explicit `range` bounds make the tail
policy checkable** (`range(0, K, 128)` with `K % 128 != 0` and no predication is a
visible bug). At L2 the slices are numeric, so annotate the axis name whenever
extents collide — `[0:128] @ [0:128]` says nothing when `BLOCK_K == BLOCK_N`.

**Write the form the hardware computes.** `Oᵀ = Vᵀ @ Pᵀ` and `O = P @ V` are the
same math and two different kernels — different A/B operands, smem layouts, and
`Major::` flags. Transposes are never elided: `ᵀ` on an operand is a claim about
how it sits in shared memory.

**L4 is computed, not asserted.** Every width and conflict count is a function of
exactly two maps — the buffer layout (with swizzle) and the thread-value map — so
it is generated by `tv_check.py`, not written in prose. `128 B swizzle,
aligned -> 0-way` is a conclusion that cannot be rechecked or regenerated when L2
changes a tile extent. Format and the bank model are in `references/l4-access.md`;
a worked table and the reasoning are in `references/example-shape.md`.

---

# Phase 0 — Calibrate the machine

**Before any tile is chosen, measure what the machine costs** — not the
datasheet, the machine at this shape through the harness the kernel will be
accepted under. Roughly an hour of jobs, and the cheapest hour in the project:
every number becomes a denominator in Phase 1. See the `benchmark-kernel` skill
for the harness and `references/example-phase0.md` for a full worked run.

Five measurements. **1–4 and the barrier half of 5 are shape-independent (0a):
run once per machine, before any tile is chosen. The occupancy half of 5 (0b)
needs the real smem/register budget, so it re-runs after L2 fixes the tile.**
None needs the kernel to exist.

| # | Measure | Why it decides the design |
|---|---|---|
| 1 | **Empty-kernel cost vs grid size** (empty body at the real CTA count and smem). | The per-launch floor. *Not* removed by a CUDA graph — it is grid ramp. If it is 1.3 µs and your kernel is 5 µs, a quarter of the budget is gone before the first byte moves, and **launch count becomes a first-class term in every fusion decision.** |
| 2 | **Cold streaming bandwidth vs transfer size** (1×, 2×, 4×, 8× your weight footprint). | Fit `t = a + MB/b`. `b` is the *marginal* bandwidth you can reach (often 60–85% of peak) and `a` folds in ramp. `bytes / peak_BW` is not a floor; it is a fantasy. |
| 3 | **Bandwidth vs CTA count** at the real footprint. | The smallest grid that reaches bandwidth. Justifies or kills split-K, finer tiles, persistent grids — the curve usually flattens well before one CTA per SM. |
| 4 | **The best existing implementation on the exact shape** (cuBLAS / cuDNN / the kernel you replace). | Two uses, neither a bound (below). Calibrates the floor model, and measures your implementation slack. |
| 5 | **Sync and placement primitives**: cluster barrier cost per cluster size, and `cudaOccupancyMaxActiveClusters` / `...MaxActiveBlocksPerMultiprocessor` at the real budget. | Cluster placement is capped by the *occupancy query*, not the SM count, so a grid that looks like one wave can be two. Barriers cost differently at the start and end of a kernel. |

**Measurement 4 is not a bound, and a target below it is not a bug.** A library
kernel solves a strictly harder problem (any shape, a fixed I/O contract); a
specialised kernel fuses across boundaries it cannot cross, and going below it is
the point (done here: a fused FFN down-projection at 7.94 µs against cuBLAS's
8.65 µs, same shape `[MEAS-C]`). Its two real uses: **calibrating the floor
model** — a well-tuned library kernel should land near `a + MB/b`; far *below*
means the model is too pessimistic and every derived target is wrong in the
safe-looking direction — and **measuring implementation slack** when the library
beats you on *identical bytes*. For a **fused** kernel the honest reference is the
**composition it replaces**: every call, plus the intermediates that round-trip
through HBM, plus one launch each.

Write the results into the spec's `toolchain` block with job ids; every floor and
target then traces to a measurement, not a peak. **Two traps this prevents, both
observed:** utilisation counters do not prove headroom is free (a kernel at 26%
DRAM and 32% L1 still paid a linear 0.23 µs/MB of L2 traffic `[MEAS-A]` — measure
the slope, do not assume it is absorbed), and a knob's slope on one kernel body
does not transfer to another (pipeline depth was 1.6× on one body, negative on
another; re-measure per body). The provenance of every number in this skill is in
`references/example-phase0.md`.

---

# Phase 1 — Write the spec

## Where it goes

Default `specs/tile/<kernel-name>.md`, created if absent; follow the repo's
design-doc convention if it has one. One spec per kernel, named after the kernel.
Start from `assets/spec-template.md`: every field is filled, marked `TODO`, or
explicitly deleted with a one-line reason (`# no cluster: sm80`). A silently
missing field is the bug this format exists to prevent.

Read `references/spec-schema.md` first — field meanings, required-per-arch, and
the **arch capability table** that decides *which questions even apply* (no warp
specialization on sm80, no TMEM below sm100).

## How to interview

The human knows the answer to most questions and does not want to be quizzed.
Interview well by minimizing the answers they have to invent:

- **Batch by section, not by field.** Five questions about the grid in one message
  is fine; one question at a time across twenty messages is not.
- **Propose, do not quiz.** Almost every field has a defensible default from the
  arch, the shapes, and settled fields. Lead with it and show the arithmetic, so
  the reply is "yes" or a correction: *"I'd take `depth: 5`. Per stage is 32.6 KB,
  so 5 stages is 163 KB, leaving 64.4 KB of the 227 KB cap for the fp32 epilogue
  (64 KB) — 0.4 KB spare, knife-edge. depth 6 needs 196 KB and does not fit.
  Confirm 5, or drop BLOCK_N to 64 and go deeper?"*
- **Derive rather than ask, and show it.** Per-stage smem, total smem, acc
  registers/thread, MMA count/stage, trip count, threads/CTA, waves/grid — all
  derived, never asked.
- **Ask when the answer encodes intent you cannot see:** the reduction axis when
  there is more than one candidate, the fusion boundary, what is dynamic vs baked
  in, the numerical contract (fp32 accumulate? dequant every K-block or once?),
  latency-at-tiny-M vs throughput-at-large-M.
- **Ask again when an answer contradicts another** — report it as arithmetic:
  *"BLOCK_M=128 with one math warp group is 128 acc registers/thread plus
  operands; that overruns 255. Two math groups (64 rows each) or BLOCK_M=64?"*
- **Never invent a number to look complete.** `TODO` plus what would settle it
  beats a plausible fabrication the reviewer will trust.

`references/interview.md` has the question bank by section and archetype (GEMM,
attention, MoE, reduction), including the questions people forget.

## Residency — persistent, cooperative, clusters

Three separate decisions, not one:

| | Is | Needs |
|---|---|---|
| **persistent** | grid sized by the machine, each CTA loops over tiles | a tile scheduler |
| **residency** | every CTA simultaneously live — what licenses a CTA to *wait on* another | cooperative launch |
| **clusters** | a small group co-scheduled in a GPC, sharing smem and multicast | neither |

**Grid = `SM_count × cta_per_sm`, not `SM_count`.** `cta_per_sm` comes from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, capped by smem **and** registers
(read `Block Limit Registers`, not only shared mem). Capacity is not residency:
128 CTAs on 132 SMs land one per SM whatever the query says, so freeing smem to
reach 3 CTAs/SM buys nothing until the grid reaches `3 × SM_count`.

**The trade that decides latency-bound kernels: `cta_per_sm` hides barriers,
`warps_per_cta` hides memory latency — orthogonal, and they buy different
things.** A block-level barrier stops every warp in the CTA at once, so only
*another CTA* fills that gap; at equal warps/SM, more CTAs of fewer warps beats
fewer of more whenever barriers sit on the critical path. This opposes the
warp-specialisation idiom (which drives `cta_per_sm` to 1) — resolve the tension
explicitly, not by inheriting 1.

`references/residency.md` is the detail: when cooperative actually pays, the
cooperative+cluster placement check, the H100 register/occupancy table, how to
size the producer (TMA needs one warp, not 128), and how phase ordering (grid
barrier vs semaphores) follows from the tile shape. Read it whenever the kernel
is persistent, warp-specialized, latency-bound, or clustered.

## Tile order is an L2 decision

`grid.rasterization` decides the order the machine walks the weight slab and how
much of the shared operand stays in L2 — `row-major` is a default to *label*, not
a decision to skip. On a static graph the CTA→tile map is an offline optimisation
with an optimum; solve it and record **solved or defaulted**. The three questions
(which operand consecutive CTAs share and how much fits; what DRAM read order the
map produces; what happens at the seam between fused kernels) and the validation
counters (`lts__t_sector_hit_rate`, `dram__bytes_read`) live in
`references/spec-schema.md` under `grid.rasterization` / `grid.l2_schedule`. The
lever's size depends on there being reuse — at small M it is ~nil; measure the
slope first.

## Consistency arithmetic — run before every hand-back

Most under-specification shows up as an equation that does not balance, and
finding it yourself beats another round of questions. **Run the script, do not do
this in your head:**

```
python3 scripts/budget.py <spec.md> --sms <SM count>
python3 scripts/budget.py <spec.md> --sms <SM count> --gate   # what `status: review` must clear
```

It reports `PASS` / `FAIL` / `TIGHT` / `SKIP` / `MANUAL` per row with the computed
number, follows `l4_accesses` into `tv_check.py`, and rejects duplicate YAML keys.
`SKIP` is not a pass (a field it needs is still `TODO`); `MANUAL` marks the rows
that are not arithmetic and still need a human (L3's bubble check, traceability,
the rasterization argument). The **canonical check list and its formulas are in
`references/spec-schema.md` §7** — that is what `budget.py` implements. Put the
results in the spec's `checks` block and lead the hand-back with anything that
fails.

## Ending Phase 1

The spec is ready when every field is filled or explicitly deleted, the L1–L4
nest is complete with explicit bounds that all trace upward, `open_questions` is
empty, and every check passes. Set `status: review`, run the gate, then **stop and
hand back:**

```
python3 scripts/budget.py <spec.md> --sms <SM count> --gate
```

`--gate` fails on `SKIP` as well as `FAIL` — exactly the difference between a
draft and a spec ready for review. Every remaining row must be `PASS`, `TIGHT` or
`MANUAL`, and each `MANUAL` is a claim the reviewer is asked to read. A spec moved
to `review` with the checker unrun is not ready; paste the output.

The hand-back message gives:

1. The path to the spec.
2. The shape in one line: `grid × cta_tile @ cta_per_sm / mainloop / stages /
   iters`, plus the L2 compute line — what an expert can sanity-check without
   opening the file.
3. **The floor and the target, with the Phase 0 numbers they came from**, so the
   reviewer sees the target is reachable before reading anything else.
4. The `budget.py --gate` output, especially anything tight (smem within a few KB
   of the cap, registers near 255, `cta_per_sm` at 1 on a latency-bound kernel),
   and **L3's bubble check** — the level the tools do *not* check, so where a
   reviewer's attention is worth most.
5. The decisions you made on the human's behalf, and the reasoning.
6. Anything you could not settle, phrased as a decision rather than a question.

Then stop. Do not begin Phase 2 in the same turn, and do not ask "shall I
implement it now?" — the reviewer needs to read the spec first. Phase 2 starts
when the human sets `status: approved` (or says so, in which case set it with
their name in `approved_by`). Corrections during review go into the spec first,
then the code — the spec is the source of truth from that point on, not the
transcript.

---

# Phase 2 — Generate the kernel

## Pick the backend first

Ask which backend and generate exactly one. Three implementations of an unproven
spec triple the review burden and debugging surface for no information gain; the
second is cheap only *after* the first is correct.

| Backend | Fits when |
|---|---|
| **TileLang** | Fastest path to a working kernel; the compiler owns pipelining and the producer/consumer split. Best when the stage structure is regular. |
| **CUTLASS / CuTe (C++)** | Production Hopper/Blackwell GEMM shapes, epilogue fusion, when the collective builders already express the schedule. |
| **CuTeDSL (Python)** | CuTe layout algebra and explicit wgmma/tcgen05 control without the C++ template build cycle. |
| **Raw CUDA + inline PTX** | The schedule is irregular enough that no library expresses it — asymmetric warp roles, hand-placed barriers, smem aliasing. Most control, most work. |

If the user has no preference, recommend from the spec: an irregular warp schedule
(FlashMLA's seesaw) argues against TileLang; a textbook staged GEMM argues for it;
an existing CUTLASS codebase argues for CuTe. Read the matching section of
`references/backends.md` — it maps every spec field onto that backend's construct
and lists what that backend *cannot* express, which is where a spec quietly stops
being implementable. Defer to `cutlass_skill`, `cuda_skill`, `triton_skill` for
API detail once the mapping is clear.

## Rules while generating

- **The spec is the contract.** Every tile size, stage count, instruction count
  comes from the spec. When the code needs a number the spec lacks, that is a
  Phase 1 defect: add the field, tell the user what you added and why, continue.
- **Diverge loudly.** If the backend cannot express something the spec's way
  (TileLang choosing its own stage placement, a builder overriding the cluster),
  record it in the spec's `deviations` block and keep the spec's intent as the
  comment next to the code. A silent divergence turns the spec into a lie for the
  next reader.
- **Comment against spec sections.** `# mainloop, stage k%5` beats restating what
  the line does — it lets the reviewer diff code against spec by eye.
- **State how it was verified.** Say plainly what you ran: compiled only, ran
  against a reference, benchmarked. This repo's `benchmark-kernel` skill covers
  per-kernel timing; numbers without a named method are noise.

---

# Reverse-engineering an existing kernel

Same format, opposite direction: read the source and fill the spec from it, every
number cited to source lines so a reviewer can check them. `open_questions` here
means "the source does not make this obvious", not "the human has not decided".
Both worked examples (`references/example-deepgemm.md`,
`references/example-flashmla.md`) were produced this way and show the citation
style.

- **Mark provenance:** `[D]` derived from cited source, `[I]` inferred from how
  the hardware must work, `TODO — needs source` when neither. A line citation *is*
  `[D]`, so `[I]` is only for the unciteable. Do not fabricate line numbers.
- **Use `status: reference`, not `approved`** — no sign-off to record, no Phase 2
  to unblock; `open_questions` may stay non-empty.
- **Name buffers in the nest with the source's own identifiers**, so
  `traceability` is mechanically checkable against the code.

This is also the best way to start a *new* kernel resembling a known one: extract
the known kernel's spec, then edit it with the human.

---

# Reference files

| File | Read when |
|---|---|
| `assets/spec-template.md` | Always — Phase 1 starts by copying it |
| `references/spec-schema.md` | Always — field meanings, required-per-arch, arch capability table, the canonical check list (§7) |
| `references/example-shape.md` | Writing a nest — one FP8 GEMM stage filled in at all four levels, with the "L4 is computed, not asserted" argument |
| `references/example-phase0.md` | Running Phase 0 — a full worked calibration with real numbers, the 0a/0b split, what it does *not* give you, and the provenance of this skill's numbers |
| `references/schedule-l3.md` | Writing or reviewing L3 — why concurrency must be specified not asserted, the gate-the-next-copy rule, cycle-count reasoning |
| `references/residency.md` | The kernel is persistent, warp-specialized, latency-bound, or clustered — cooperative launch, the CTA/warp trade, producer sizing, phase ordering |
| `references/primitives.md` | The kernel contains a softmax, online/streaming reduction, fused norm, or cross-CTA split reduction — named contracts with their state and hazards |
| `references/interview.md` | The kernel is not a plain GEMM, or a section feels thin |
| `references/backends.md` | Phase 2, after the backend is chosen |
| `references/example-deepgemm.md` | A GEMM, or a producer/consumer warp split — SM90 FP8, persistent, TMA multicast, per-K-block dequant |
| `references/example-flashmla.md` | Attention/decode, or a non-producer/consumer warp split — SM90 MLA, two math warp groups in a seesaw, split-KV |
| `references/l4-access.md` | Writing L4 — the access-file format, the layout and thread-value maps, the bank model |

# Scripts

Design-time only: pure python, no CUDA and no torch, so they run on a login node.

| Command | Does |
|---|---|
| `python3 scripts/budget.py <spec.md> [--sms N] [--gate]` | The consistency arithmetic, one line per check with the computed number. `--gate` is what `status: review` must clear. Follows `l4_accesses` into `tv_check.py` |
| `python3 scripts/tv_check.py <access-file> [--markdown]` | L4's per-thread table: widths, vector legality, transactions, bank conflicts against their ideal. `--markdown` emits the table to paste into the spec |
| `python3 scripts/tests/test_tvlayout.py` | Cross-validates the layout/swizzle algebra against CUTLASS's `pycute` and runs every shipped known answer. Run after touching `scripts/` |

The layout algebra in `scripts/tvlayout.py` reimplements the part of CuTe these
checks need, so a spec can be checked without a CUDA toolchain; it is compared
against `pycute` in the test suite rather than trusted — a confidently wrong bank
count is the exact failure the checker exists to remove.
