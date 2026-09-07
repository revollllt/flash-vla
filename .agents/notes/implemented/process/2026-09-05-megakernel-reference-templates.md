# Agent Note: one runnable template per published megakernel reference

Status: implemented

## Problem

The kernel-design template library carried one megakernel template (40), a
reduced HazyResearch-style interpreter that lost to three plain launches at
every depth tried and whose header ended with a hypothesis: that the loss was
the planner's, not the interpreter's. Four upstream references were being
cited for the idiom -- HazyResearch/Megakernels, Mirage MPK, DeepGEMM Mega
MoE and learn-cuda's 12_megakernel -- with none of their mechanisms in a form
that compiles here, and no measurement that could be set beside their
published numbers. The load-policy vocabulary from learn-cuda (`L1::no_allocate`,
`L2::evict_first`) had never been tried on sm90.

## Decision

- Four templates, one per reference, under
  `.claude/skills/kernel-wiki/artifacts/kernels/sm90-templates/variants/`
  (moved there from `kernel-design/references/templates/` per the
  [references evidence-discipline note](2026-09-07-kernel-design-references-evidence-discipline.md)): `42_hazy_llama_megakernel.cu`,
  `43_mpk_task_graph_runtime.cu`, `44_megamoe_sm90.cu`,
  `45_flag_barrier_megakernel.cu`. Each is the whole machine of its reference
  written against `sm90_common.cuh` (no ThunderKittens, no Mirage runtime, no
  DeepGEMM headers), builds standalone, checks itself against a
  double-precision reference with the device's bf16 rounding points, and times
  itself graph-captured. Its header carries the upstream's published numbers
  and its own STATUS block; the wiki entry `ext-mpk-megakernel` carries the
  rules common to all four.
- Templates 42 and 43 run one workload (a Llama-3.2-1B decode step: 16 layers
  and lm_head, KV position 1023) so the static per-SM planner and the dynamic
  worker/scheduler runtime are compared on equal terms. Template 44 is an sm90
  port of the sm100 design (DeepGEMM has no sm90 Mega MoE source, nor does the
  sgl-project fork); template 45 keeps learn-cuda's Qwen3-0.6B shapes so its
  bandwidth accounting matches the upstream table.
- Deviations from each upstream are listed in each header under "what was
  changed"; none change the design. The material ones: 42 reads activations
  and norm weights straight into registers instead of through a page, uses all
  16 consumer warps for attention, and publishes an instruction's counters once
  after one write-completion wait rather than per block; 43 is single-process
  and single-kernel, with warp-per-row GEMV tasks in place of the Hopper
  TMA+wgmma tasks (batch 1 has no use for the tensor core); 44 is single rank,
  bf16, resets counters by a memset per launch, and reads the combine slots
  with plain loads.
- Template 40 stays as the ledger of the losing design, with a header pointer
  to 42-45 naming the one structural difference: it waits on the dependency
  before issuing any operand copy.
- The templates are run on the cluster through `sbatch/kernel_template.sh`
  (builds on the compute node, runs the harness under a timeout so a deadlocked
  persistent kernel ends a run rather than a job).

## Alternatives considered

- Extending template 40 into the HazyResearch machine in place: rejected --
  the losing version has reference value only if it stays legible as what it
  is, and the four references differ in the layer that matters (planner,
  scheduler, ring counters, flag barrier).
- Building the upstream code (ThunderKittens, the Mirage runtime) to measure
  it directly: rejected -- the templates must be toolkit-only, and a number
  from a foreign build proves nothing about a toolkit port. The upstream
  numbers are cited as published and compared through the fraction of the
  measured bandwidth floor.
- Porting DeepGEMM's fp8 Mega MoE variant: deferred -- the scheduler and
  counter mechanisms are identical in the bf16 variant and the fp8 path adds
  the scale-factor pool without adding a mechanism.
- Attention on one warp with mma as upstream (42): rejected -- 16 warps with
  an LSE merge through page 0 keeps the instruction semantics and pages and
  removes the single-warp serialization.

## Consequences

- The Mega MoE stream rate is set by BLOCK_M: an expert is streamed once per
  pool block, so the largest block that keeps each expert in one block is the
  lever, which is what upstream's per-call JIT choice of BLOCK_M is for.
- The wiki's first megakernel rule is now measured rather than inferred: the
  dependency wait must sit under the weight stream. Template 40 waits first
  and loses at every depth; template 42 streams weights while waiting and
  reaches 84% of the bandwidth floor on the same op mix.
- The scheduler comparison is on record: pre-launched static or round-robin
  schedules (42 rr/dag, 43 aot) land within ~10% of each other; launch-on-fire
  (43 jit) costs a scheduler hop per task and loses 1.5x.
- Attention parallelism is a first-order term for a GQA megakernel: 8
  instructions on 132 SMs idle the weight stream once the prefetch fills;
  the split-KV partials knob is worth ~15% of the step.
- On sm90, `ld.global.L2::evict_first` is not accepted by ptxas (256-bit loads
  only); the legal spelling is `L2::cache_hint` with a `createpolicy`
  descriptor, and neither it nor `L1::no_allocate` moves a decode GEMV. The
  upstream's `relaxed.cta` loads are strong scoped loads (`LDG.E.NA.STRONG.SM`),
  not a cache policy.
- The checker covers 23 templates; the new four assert the instructions that
  are their point (setmaxnreg, mbarrier inval, bulk reduce-add, release/acquire
  atomics, the queue CAS, bulk L2 prefetch, wgmma m64n64k16, 3-D TMA).

## Verification

- `python3 .claude/skills/kernel-wiki/scripts/check_templates.py` passes 23/23
  on the login node (`cuda/13.1`, `gcc/13.3`), the four new templates among
  them.
- GPU runs via `sbatch/kernel_template.sh` on `acd_u` H100 SXM5 nodes (clocks
  not lockable; `min` reported); logs under `sbatch/logs/mkref_*.out`. Each
  template's STATUS block is the current record:
  - 45: MLP 11.55 us / 1634 GB/s (upstream 11.65 us / 1622 GB/s on H200),
    attention 23.7 us at kv 4096 (upstream Triton v2 29.8 us), exact match to
    the double reference.
  - 42: 1.081 ms per decode step at partials=8, 84% of the 0.906 ms floor,
    2.32 TB/s (upstream: under 1 ms, 78% of the datasheet peak); argmax
    matches the reference at every setting.
  - 43: 1.208 ms per decode step (aot), 74% of floor; jit 1.883 ms; argmax
    matches over 8 chained decode steps.
  - 44: 16 experts x (7168, 2048) bf16, top-8: 563 us at 128 tokens with
    BLOCK_M 128 (93% of the load floor, 2.56 TB/s) and 2.9-3.5 TB/s on the
    streamed bytes where experts span several pool blocks; passes the double
    reference at 8, 128 and 512 tokens with no element beyond a quarter of
    the rms.
- Deadlock coverage: 42 was run with truncated tables per op kind
  (`stop=qkv|attn|upgate`), 1, 2 and 16 layers, partials 1/4/8/16, both
  schedulers; 43 with 1/2/16 layers, 1/2/3/4/8 chained steps, aot/jit,
  prefetch on/off; 44 with the ring larger than, equal to and smaller than the
  pool (8, 128, 512 tokens).
