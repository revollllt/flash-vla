# Agent Note: kernel-design workflow replaces the tile-dataflow spec gate

Status: implemented

## Problem

The spec-first gate (tile-dataflow: an L1-L4 spec, budget checks, and human
sign-off before any kernel code) was too heavy for the loop it guarded:
complex constraints, a large per-task token cost, no directly usable
examples, and it slowed optimization more than it protected it. There was
also no single entry point for "write a kernel" — the flow was spread over
several skills and rules.

## Decision

- Retire `tile-dataflow` to `.claude/skills-archived/` (out of skill
  discovery; its checker scripts remain greppable for salvage).
- Adopt `kernel-design` as the one entry point: contract -> torch reference
  -> parity -> candidate loop -> promotion, with two modes. Auto is
  KDA-style: one contract ack from the human, then autonomous to a stop
  condition. Human mode signs the plan before code and steers between
  candidates. Method adapted from mit-han-lab/kernel-design-agents; no
  content copied (that repository carries no license).
- Companion knowledge: a symptom-indexed sm90 wiki under
  `kernel-design/references/wiki/`, and an `ncu-report` skill for report
  interpretation (capture stays with `gpu-profiler-analysis`).
- Skills carry distilled, portable experience only. Evidence — job ids,
  measurements, experiment history — lives project-side: in Agent Notes and
  in per-task workspaces under `artifacts/ktasks/` (gitignored).

## Alternatives considered

- Keep the spec gate and bolt a reference stage onto it: rejected — the
  gate's cost was the problem, not its coverage.
- One monolithic skill inlining benchmark/profiling guidance: rejected —
  it would duplicate contracts that `benchmark-kernel`,
  `hardware-unit-test`, and `gpu-profiler-analysis` already own.
- Citing experiment records (jobs, spec rows) inside wiki entries:
  rejected — couples skills to project history; refactors would cascade.

## Consequences

- New kernel work enters through the kernel-design contract; floors still
  divide by measured hardware-unit-test tags, baselines are measured before
  any candidate, and `min` is read under unpinned clocks.
- Workspaces are disposable and never committed; promotion ships the
  kernel, its parity script, a benchmark case, and this note's update in
  one PR.

## Verification

The `ncu-report` capture+interpret walkthrough ran end-to-end on this
cluster (sbatch on an ncu-capable node; per-line hotspots resolved), and
its query tool parses existing reports on the login node. Skill and wiki
texts grep clean of experiment-record residue.
