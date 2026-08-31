# The four levels, resource ownership, and notation

Moved out of `SKILL.md` 2026-08-25: this is the *rationale* behind the level
split. SKILL.md carries the table you need to act; this file carries the
argument you need once, when you are learning the format or defending a
choice in review.

## The mental model

A GPU kernel is one nested structure. Each level answers *what is the unit of
work here, and what data moves*:

```
grid        which output tile does this CTA own?          -> cta_tile, rasterization, cluster
 mainloop   which slice of the reduction axis is this?    -> axis, step, trip_count
  stage     which smem buffer, and who filled it?         -> depth, barriers, producer/consumer
   iter     which MMA instruction, acc living where?      -> inst_shape, count, acc location
```

`stage` and `iter` are the two levels people leave vague, and the two that
decide performance. One mainloop iteration *usually* occupies one pipeline
stage; when it does not, say the ratio (an iteration consuming two stages is a
**compute pair**, not a fill/drain pair — it changes what `depth: 2` means).
Inside one compute there are typically several MMA instructions, because the
hardware MMA shape is smaller than the tile — that count is the `iter`
granularity. Say both numbers explicitly, every time.

Warp specialization cuts across all four levels: on Hopper/Blackwell the CTA's
warps split by *role* (a producer issuing TMA, one or more math groups issuing
MMA, sometimes a separate epilogue group), and the stage barriers couple them.
A spec that names stages but not the warp groups arriving on them is
incomplete.

## What each level is

| Level | Is | Answers |
|---|---|---|
| **L1 — kernel** | the iteration space, no hardware in it | What crosses the kernel boundary, what loops cover the whole problem, which are parallel and which is the contraction. |
| **L2 — CTA and stage** | the same nest, mapped to hardware | Which loops become the grid, which stays serial in a CTA, what one stage loads and computes, with tile shapes and byte counts. |
| **L3 — schedule** | one stage, three engines running concurrently | What the **async copy engine (TMA/cp.async)**, the **CUDA cores** and the **tensor cores** each issue during this stage, which edges order them, where the bubbles are. |
| **L4 — instructions and threads** | the innermost body, lowered | The MMA instruction and its `m×n×k`, the iter loop's bounds, where A/B/C live — and per thread, access width, coalescing, bank conflicts, and where the address arithmetic lives. |

**L3 decides *when*; L4 decides *what and how wide*.** A bubble is an L3 defect;
a bank conflict is an L4 defect. Keeping them in one level is what lets a spec
assert an overlap nobody can review.

## Which level owns which resource

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

## L3 is not optional in a fused kernel

SKILL.md carries this rule in full. The war story behind it, the three forms a
gate can take, and how to reason about cycle counts Phase 0 does not measure
(ratios between columns, never absolutes) are in `schedule-l3.md`. On sm80 there
is no separate copy engine — draw two columns and say so.

## Notation

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
policy checkable** (`range(0, K, 128)` with `K % 128 != 0` and no predication is
a visible bug). At L2 the slices are numeric, so annotate the axis name whenever
extents collide — `[0:128] @ [0:128]` says nothing when `BLOCK_K == BLOCK_N`.

**Write the form the hardware computes.** `Oᵀ = Vᵀ @ Pᵀ` and `O = P @ V` are the
same math and two different kernels — different A/B operands, smem layouts, and
`Major::` flags. Transposes are never elided: `ᵀ` on an operand is a claim about
how it sits in shared memory.

**L4 is computed, not asserted.** Every width and conflict count is a function
of exactly two maps — the buffer layout (with swizzle) and the thread-value map
— so it is generated by `tv_check.py`, not written in prose. `128 B swizzle,
aligned -> 0-way` is a conclusion that cannot be rechecked or regenerated when
L2 changes a tile extent. Format and the bank model are in `l4-access.md`; a
worked table and the reasoning are in `example-shape.md`.

## What is deliberately NOT here

Each of these has its own home; duplicating them here is how a reference file
becomes another SKILL.md:

| Topic | Lives in |
|---|---|
| residency, cooperative launch, producer sizing, the CTA-vs-warp trade | `residency.md` |
| L3's war story, the three gate forms, cycle-count reasoning | `schedule-l3.md` |
| the question bank per section and archetype | `interview.md` |
| tile order / L2 schedule fields and their validation counters | `spec-schema.md` |
| every machine number and its provenance | the `hardware-unit-test` skill |
