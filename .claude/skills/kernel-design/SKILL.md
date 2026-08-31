---
name: kernel-design
description: The end-to-end workflow for writing or optimizing a GPU kernel in this repo — a one-screen task contract, a dimension-annotated torch reference, a parity harness, then a candidate loop (implement -> validate -> benchmark -> profile) with recorded evidence and an explicit promotion rule. Two modes, auto (agent-driven, one contract receipt, then autonomous to a stop condition) and human (plan signed off before code, human steers between candidates). Use when asked to write a new kernel, port an op to CUDA or TileLang, fuse ops across a boundary, tune a fixed-shape kernel, or run an optimization pass on an existing kernel.
---

# Kernel Design — a contract, a reference, then an evidence loop

This repo ships fixed-workload targets: one device, one model revision, one
shape profile (`ARCHITECTURE.md`). A kernel tuned to those exact shapes
routinely beats a general-purpose library kernel on them, and the fusions this
pipeline wants have no library form at all — so kernels are written here, in a
loop built for iteration speed with evidence discipline. Evidence lives
project-side — in each task's workspace and the Agent Notes; skills, this one
included, carry only the distilled experience.

## Two modes, one backbone

The user picks the mode per task; the contract records it.

| | auto | human |
|---|---|---|
| contract | filled by the agent, **handed back once** for an ack before any GPU time is spent | same |
| plan | agent writes `docs/draft.md` -> `docs/plan.md`, then proceeds | `plan.md` needs human sign-off before candidate 1 |
| steering | none until a stop condition | human may redirect between candidates; directives are folded into `plan.md` |
| returns | at a stop condition, with the evidence pack | each round, or as agreed in the contract |

## The backbone

1. **Contract** — copy `assets/contract-template.md`, fill every field: tensor
   table (named dims -> fixed numbers, dtype, mutation), fusion region,
   validation command, baselines, floor from measured tags, promotion
   criteria, budget, mode. Hand it back — one screen — and wait for the ack.
   A wrong op boundary is more expensive than any lost optimization.
2. **Baselines first** — measure the best existing implementations of this op
   at the exact production shape BEFORE any candidate exists: the current
   production route, torch/SDPA, a library kernel where one applies. Targets
   set without this step have ended up below their own floors before.
3. **Reference** — the torch oracle, per `references/reference-tiers.md`. The
   ABI-mirror tier is mandatory; the per-stage decomposition tier is required
   when the fusion region spans two or more pipeline stages.
4. **Parity** — per `references/parity.md`: one shared five-metric module,
   tolerances split into gates vs reports.
5. **The loop** — per `references/loop.md`: workspace under
   `artifacts/ktasks/<task>/` (never committed), one candidate at a time,
   every candidate validated and logged in `candidates.jsonl` with parent
   links and reject reasons.
6. **Promotion** — the winner ships in one PR: kernel into the target's
   `backends/`, parity script into `eval/correctness/`, a built-in benchmark
   case, the Agent Note, and the evidence summary copied out of the workspace.

## Handoffs — this skill sequences, others own

| Need | Go to |
|---|---|
| a machine number, a floor, "is this target reachable" | `hardware-unit-test` — cite tags, never datasheet peaks |
| per-kernel timing, comparing two implementations | `benchmark-kernel` |
| capturing a trace or an NCU report | `gpu-profiler-analysis` |
| reading an NCU report -> naming the bottleneck | `ncu-report` |
| choosing the next optimization move | `references/wiki/README.md` — symptom-indexed |

## Files

| File | Read when |
|---|---|
| `assets/contract-template.md` | starting any kernel task |
| `references/reference-tiers.md` | writing the torch reference |
| `references/parity.md` | writing or judging a parity harness |
| `references/loop.md` | running the candidate loop; the promotion checklist |
| `references/wiki/README.md` | picking the next optimization move |
