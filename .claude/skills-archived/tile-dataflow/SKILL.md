# Tile-Level Dataflow → Kernel

The expensive kernel decisions — output tiling across CTAs, pipeline depth,
which warps load and which compute, how many MMA instructions per stage — are
made in the first ten minutes and painful to change once there is code. This
skill is the spec format that pins them down, plus the checker that keeps the
spec self-consistent.

## When this applies

**Match the process to the change. Most work is row 3.**

| You are | Do |
|---|---|
| designing a **new kernel**, or changing a spec'd kernel's **dataflow thesis** — the fusion boundary, warp roles, what overlaps what, the reduction axis | the full gate: spec first, human sign-off, then code |
| **retuning** a spec'd kernel — tile sizes, ring depth, BK, a buffer layout, a descriptor, a constant | edit the **changed fields only**, re-run `budget.py --gate`, then **go straight to code**. No spec rewrite, no formal hand-back. Escalate to row 1 only if a floor moves or a check newly fails |
| a constant, a comment, a harness, a probe, a benchmark, a profiler run | **not this skill** |

Row 1 is a hard gate: **no kernel code until a human signs off.** Do not soften
it, do not "start a draft while we discuss". Row 2 is not a gate — the dataflow
is already pinned, only numbers moved, and `budget.py` catches the arithmetic.

## The four levels

Every spec carries a loop-structured pseudocode block at four levels, each
refining the one above.

| Level | Is | Answers |
|---|---|---|
| **L1** | the iteration space, no hardware | what crosses the kernel boundary; which loops are parallel, which is the contraction |
| **L2** | the same nest, mapped to hardware | grid vs serial-in-CTA; what one stage loads and computes, with shapes and byte counts |
| **L3** | one stage, three engines concurrent | what the copy engine / CUDA cores / tensor cores each issue, the ordering edges, the bubbles |
| **L4** | the innermost body, lowered | the MMA `m×n×k`, iter bounds, where A/B/C live; per-thread width, coalescing, bank conflicts |

**L3 decides *when*; L4 decides *what and how wide*.** A bubble is an L3 defect;
a bank conflict is an L4 defect. **L4 is computed by `tv_check.py`, never
asserted in prose.** Resources are budgeted high and realised low — smem,
registers and grid size are **L2** decisions that merely *show up* at L4.

**In a fused kernel L3 is mandatory**, because the thesis is concurrency, not
dependency. Give every stage a three-column timeline and apply one rule: *find
what actually gates the next copy, because it is later than it looks* — the
issue, the release, or a buffer wgmma has not finished reading. An empty column
is a bubble the spec shows you before the kernel exists.

## Phase 0 — calibrate

**The `hardware-unit-test` skill owns this.** Read its table before running
anything — `python3 .claude/skills/hardware-unit-test/scripts/constants.py` — and
measure only what it does not cover.

Cite its tags (`[tma.issue.warp]`, `[ld.bw.dev.dram]`, `[launch.lat.dev.ramp]`) in the spec's
`toolchain` block, with job ids, so every floor traces to a measurement rather
than a datasheet peak. Never restate its numbers — a number in two places
drifts. If a floor needs a constant that skill marks a **GAP**, the floor is
**blocked**: say so instead of substituting a peak.

## Phase 1 — write the spec

Default `specs/tile/<kernel-name>.md`. Start from `assets/spec-template.md`:
**every field is filled, marked `TODO`, or explicitly deleted with a one-line
reason** (`# no cluster: sm80`). A silently missing field is the bug this
format exists to prevent.

`references/spec-schema.md` has field meanings, required-per-arch, and the arch
capability table that decides which questions even apply.

Interview by **proposing with arithmetic, not quizzing** — derive everything
derivable (per-stage smem, acc regs/thread, MMA count, trip count, waves) and
ask only where the answer encodes intent you cannot see. Never invent a number
to look complete.

Run the consistency arithmetic — **the script, not your head**:

```
python3 scripts/budget.py <spec.md> --sms <N> [--gate]
```

`--gate` is what `status: review` must clear. `SKIP` is not a pass (a field it
needs is `TODO`); `MANUAL` marks rows that need a human. Put results in `checks`.

### Ending Phase 1 (row 1 only)

Set `status: review`, run `--gate`, then **stop and hand back**: the spec path;
the shape in one line (`grid × cta_tile @ cta_per_sm / mainloop / stages /
iters`) plus the L2 compute line; **the floor and target with the measurements
they came from**; the `--gate` output, especially anything tight and **L3's
bubble check** (the level the tools do not check); the decisions you made on the
human's behalf; and what you could not settle, phrased as a decision.

Then stop — do not begin Phase 2 in the same turn, and do not ask "shall I
implement it now?". Phase 2 starts when the human sets `status: approved`.
**Corrections during review go into the spec first, then the code**: the spec is
the source of truth from that point on, not the transcript.

## Phase 2 — generate the kernel

Ask which backend and generate **exactly one**.

| Backend | Fits when |
|---|---|
| **TileLang** | fastest path to working; compiler owns pipelining and the producer/consumer split. Best when stage structure is regular |
| **CUTLASS / CuTe** | production Hopper/Blackwell GEMM shapes, epilogue fusion, when the collective builders already express the schedule |
| **CuTeDSL** | CuTe layout algebra and explicit wgmma/tcgen05 control without the C++ build cycle |
| **Raw CUDA + PTX** | asymmetric warp roles, hand-placed barriers, smem aliasing — most control, most work |

`references/backends.md` maps every spec field onto that backend's construct and
lists what it *cannot* express.

- **The spec is the contract.** A number the code needs and the spec lacks is a
  spec defect: add the field, say what you added, continue.
- **Diverge loudly.** Record it in `deviations`, keep the spec's intent as the
  comment next to the code. A silent divergence makes the spec a lie.
- **Comment against spec sections** (`# mainloop, stage k%5`).
- **State how it was verified** — compiled, run against a reference, or
  benchmarked. Numbers without a named method are noise.

## Reverse-engineering an existing kernel

Same format, opposite direction: fill the spec from the source, every number
cited to a line. Mark provenance `[D]` derived from cited source / `[I]`
inferred from how the hardware must work / `TODO — needs source`; a line
citation *is* `[D]`. Do not fabricate line numbers. Use `status: reference`, and
name buffers with the source's own identifiers so `traceability` is checkable.

## Reference files

| File | Read when |
|---|---|
| `assets/spec-template.md` | Phase 1 starts by copying it |
| `references/spec-schema.md` | field meanings, arch capability table, the check list (§7) |
| `references/example-shape.md` | one GEMM stage filled in at all four levels — the format demo |
| `references/levels-and-notation.md` | the notation contract, resource ownership, residency, tile order |
| `references/schedule-l3.md` | writing or reviewing L3 |
| `references/residency.md` | persistent, warp-specialized, latency-bound, or clustered |
| `references/l4-access.md` | writing L4 — access-file format, layout and TV maps, bank model |
| `references/primitives.md` | softmax, online reduction, fused norm, cross-CTA split reduction |
| `references/interview.md` | not a plain GEMM, or a section feels thin |
| `references/backends.md` | Phase 2, after the backend is chosen |

The live worked example is `specs/tile/ffn_taskloop.md` — a real gated spec for
a kernel this repo owns. Two frozen reference specs (DeepGEMM, FlashMLA) sit in
`specs/reference/`; they are reading material for an unfamiliar archetype, not
part of this routing table.

## Scripts

Design-time only: pure python, no CUDA and no torch, so they run on a login node.

| Command | Does |
|---|---|
| `python3 scripts/budget.py <spec.md> [--sms N] [--gate]` | the consistency arithmetic, one line per check. Follows `l4_accesses` into `tv_check.py` |
| `python3 scripts/tv_check.py <access-file> [--markdown]` | L4's per-thread table: widths, vector legality, transactions, bank conflicts |
| `python3 scripts/tests/test_tvlayout.py` | cross-validates the layout algebra against CUTLASS `pycute`. Run after touching `scripts/` |
