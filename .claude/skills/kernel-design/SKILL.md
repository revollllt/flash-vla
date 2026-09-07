---
name: kernel-design
description: The end-to-end workflow for writing or optimizing a GPU kernel in this repo — a one-screen task contract, a dimension-annotated torch reference, a parity harness, then a candidate loop (implement -> validate -> benchmark -> profile) with recorded evidence and an explicit promotion rule. Two modes, auto (agent-driven, one contract receipt, then autonomous to a stop condition) and human (plan signed off before code, human steers between candidates). Use when asked to write a new kernel, port an op to CUDA or TileLang, fuse ops across a boundary, tune a fixed-shape kernel, or run an optimization pass on an existing kernel. Do NOT use for a pipeline- or stream-level profile (torch profiler and nsys through gpu-profiler-analysis), a machine constant (hardware-unit-test), a timing claim on its own (benchmark-kernel), or a knowledge question about an sm90 mechanism or symptom (kernel-wiki).
argument-hint: "[auto|human] <op or fusion region> | --candidate <plan> | --profile <run>"
---

# Kernel Design — a contract, a reference, then an evidence loop

This repo ships fixed-workload targets: one device, one model revision, one
shape profile (`ARCHITECTURE.md`). A kernel tuned to those exact shapes
routinely beats a general-purpose library kernel on them, and the fusions this
pipeline wants have no library form at all — so kernels are written here, in a
loop built for iteration speed with evidence discipline. This skill is the
flow only, after kernel-design-agents: the mechanisms, the diagnoses and the
compilable sm90 templates are the `kernel-wiki` skill; evidence lives
project-side, in each task's workspace and the Agent Notes.

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
   validation command, baselines, ceiling from measured tags and the upper
   bound of gain, promotion criteria (the registry's `promotion_bar_ms` and
   tail bound, cited, not a number of its own), budget, mode. Hand it back — one screen — and wait
   for the ack.
   A wrong op boundary is more expensive than any lost optimization.
2. **Baselines first** — measure the best existing implementations of this op
   at the exact production shape BEFORE any candidate exists: the current
   production route, torch/SDPA, a library kernel where one applies. Targets
   set without this step have ended up below their own floors before.
3. **Reference** — the torch oracle, per `references/reference-tiers.md`. The
   ABI-mirror tier is mandatory; the per-stage decomposition tier is required
   when the fusion region spans two or more pipeline stages.
4. **Parity** — per `references/parity.md`: one shared six-metric module,
   tolerances split into gates vs reports.
5. **The loop** — per `references/loop.md`: workspace under
   `artifacts/ktasks/<task>/` (never committed), one candidate at a time,
   every candidate validated and logged in `candidates.jsonl` with parent
   links and reject reasons.
6. **Promotion** — the winner ships in one PR: kernel into the target's
   `backends/`, its check into `eval/` (a shipped kernel) or `lab/` (a candidate), a built-in benchmark
   case, the Agent Note, and the evidence summary copied out of the workspace.

## Handoffs — this skill sequences, others own

Profiling is two-level, and the rule is stated once in
`references/loop.md`: pipeline and stream questions go to the torch
profiler and `nsys`, kernel questions to `ncu`; the loop's profile is an ncu
pass whose symptom is the wiki's index.

| Need | Go to |
|---|---|
| a machine number, a floor, "is this target reachable" | `hardware-unit-test` — the measured table under the hardware axis; the ceiling cites tags, the roofline cites `spec.py`, neither is a target |
| per-kernel timing, comparing two implementations | `benchmark-kernel` |
| why a candidate is slow — the kernel-level profile: an ncu pass, capture plan through `REPORT.md`, read into a named bottleneck | `ncu-report` |
| where the pipeline's time goes — the stream-level profile: stage / launch / graph timeline with the torch profiler or `nsys`; also the capture runner behind an ncu pass | `gpu-profiler-analysis` |
| choosing the next optimization move, or what a mechanism costs and how it fails | `kernel-wiki` — `scripts/query.py --symptom <s>` from the report's stall reason, then the pattern page's candidate techniques |
| how a sm90 mechanism is actually spelled | `kernel-wiki` — the sm90 templates bundle (`artifacts/kernels/sm90-templates/variants`, `queries/by-template.md`); compilable, graded skeletons |

## Files

| File | Read when |
|---|---|
| `assets/contract-template.md` | starting any kernel task |
| `references/reference-tiers.md` | writing the torch reference |
| `references/parity.md` | writing or judging a parity harness |
| `references/loop.md` | running the candidate loop; the promotion checklist |
