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
  candidates. Method adapted from mit-han-lab/kernel-design-agents and its
  MIT-licensed KernelWiki; no content is copied, because that base is
  Blackwell-first and its sm100 instruction vocabulary has no sm90
  counterpart.
- Companion knowledge in two layers: a symptom-indexed sm90 wiki under
  `kernel-design/references/wiki/` says which move to make, and
  `kernel-design/references/templates/` says how it is spelled. The templates
  are two tiers: a mechanism ladder (TMA/mbarrier ring, warp-role split, wgmma
  batch, bulk-store epilogue) and kernel archetypes distilled from the pinned
  `third_party/` sources -- persistent warp-specialized GEMM with cluster
  multicast, sm90 fp8 two-level accumulation, FlashAttention-3 online softmax,
  split-KV decode with an LSE combine, and grouped/masked MoE GEMM -- and a
  mixed-precision family (INT4A16, INT4A8, NVFP4A16, MXFP4A16, MXFP4A8) whose
  organizing fact is that sm90 has no sub-8-bit tensor core, so every low-bit
  kernel unpacks in registers before the MMA and the port to sm120 deletes that
  unpack rather than translating it. That family ships its offline repack
  alongside the kernels, because a quantized kernel and its packer are one
  artifact: the permutation exists to make the dequant emit MMA-ordered pairs,
  so a kernel without its matching packer is wrong rather than slow. A fourth
  tier covers the memory-bound glue between GEMMs -- norm, swiglu with fused
  per-token quantization, rope and softmax -- whose shared constraints are
  traversals of the row and the launch ramp, so they fuse and they all carry
  PDL. A fifth tier covers the fusion endgame -- the four published megakernel forms
  (HazyResearch's planned interpreter, Mirage MPK's task-graph runtime,
  DeepGEMM's Mega MoE ported to sm90, learn-cuda's flag-barrier kernel), one
  runnable and self-timing template each, plus the MoE align/finalize pass
  around the grouped GEMM -- where launch cost stops being tunable and becomes
  structural. The megakernel templates are deliberately the largest files in
  the set: the idiom's difficulty is entirely in machinery a sketch omits, so a
  short version of it has no reference value; see the
  [megakernel reference templates note](2026-09-05-megakernel-reference-templates.md). An `ncu-report` skill owns report
  interpretation (capture stays with `gpu-profiler-analysis`); its current
  shape is recorded in the
  [ncu-report skill template note](2026-09-06-ncu-report-skill-template.md).
  Profiling is two-level by default: pipeline / stream questions (which
  stage, launch, gap or overlap) go to the torch profiler and nsys through
  `gpu-profiler-analysis`; kernel questions (why this kernel is slow) go to
  Nsight Compute read through `ncu-report`, and that is the profile the
  candidate loop runs.
- Templates are toolkit-only and de-projectized, so they stay portable
  experience rather than a second copy of the kernels; each declares the PTX
  instructions it exists to demonstrate and `scripts/check_templates.py`
  compiles it and asserts them. This is a deliberate departure from the
  upstream wiki, whose snippets are verbatim upstream excerpts checked for
  provenance rather than for compilability. Each template declares a grade the
  checker enforces — `structural` for a skeleton that may claim no measurement,
  `reference` for a machine that runs and reports its own — per the
  [template grades and wiki linter note](2026-09-06-template-grades-and-wiki-linter.md);
  for a structural template the guarantee catches a missing or eliminated
  instruction, never a wrong value, and parity harnesses remain the numerical
  authority. Archetypes fix a shape and omit tail handling, predication and
  autotuning; each names the upstream file that carries the production version.
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

`python3 .claude/skills/kernel-design/scripts/check_templates.py` passes 23/23
on the login node with `cuda/13.1` and `gcc/13.3` (`-arch=sm_90a -ptx`, no GPU).
The checker was negative-tested both ways: an unsatisfiable assertion and a
deliberate compile error each fail it. `check_wiki.py` passes 25/25 entries with
no corpus-wide failure. Two
claims are carried by compile-time assertions inside the templates rather than
by prose: that GEMM 1's accumulator and GEMM 2's A operand share a thread
mapping (template 12), and that the archetype shared-memory pools fit the
per-CTA limit.
