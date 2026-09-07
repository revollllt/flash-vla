---
id: kernel-megakernel-forms
title: "Megakernel forms: planner interpreter, task-graph runtime, Mega MoE, flag barrier"
type: kernel
architectures: [sm90]
tags: [megakernel, persistent-kernel, static-scheduling, flag-barrier, task-loop, moe, decode, fused-kernel, tma, mbarrier, cache-hint]
confidence: measured
reproducibility: benchmarked
kernel_types: [fused-kernel, decode, moe]
languages: [cuda-cpp]
related: [pattern-fusion-latency-chain, technique-prefetch-across-dependency, technique-producer-fusion-pdl, pattern-cold-burst-ceiling, kernel-deepgemm, kernel-fused-moe, kernel-grouped-gemm]
sources: [blog-hazyresearch-megakernels, doc-mirage-mpk, blog-deepgemm, blog-learn-cuda-megakernel, doc-kernel-design-templates, doc-hardware-unit-test, note-2026-09-05-megakernel-reference-templates]
evidence_basis:
  - evidence_type: benchmark
    source_id: doc-kernel-design-templates
  - evidence_type: upstream-code
    source_id: blog-hazyresearch-megakernels
performance_claims:
  - gpu: H100 SXM5
    dtype: bf16
    shape: "Llama-1B-shaped decode step, 16 layers + lm_head, pos 1023, batch 1, partials=8; template 42, static planner"
    metric: ms
    value: 1.081
    utilization: "84% of the [ld.bw.dev.dram] floor"
    source_id: doc-kernel-design-templates
    source_locator: "42_hazy_llama_megakernel.cu STATUS block (CUDA 13.1, clocks not pinned, min of 3 x 20)"
  - gpu: H100 SXM5
    dtype: bf16
    shape: "same decode step; template 43, task-graph runtime, mode=aot, 8 steps per launch"
    metric: ms
    value: 1.227
    utilization: "74% of the [ld.bw.dev.dram] floor"
    source_id: doc-kernel-design-templates
    source_locator: "43_mpk_task_graph_runtime.cu STATUS block"
  - gpu: H100 SXM5
    dtype: bf16
    shape: "d=2048 ffn=8192, batch 1, one layer; template 40 reduced interpreter, 132 CTAs, against 90.75 us for three launches"
    metric: us
    value: 115.25
    source_id: doc-kernel-design-templates
    source_locator: "40_megakernel_interpreter.cu STATUS block"
  - gpu: H100 SXM5
    dtype: bf16
    shape: "MLP M=1 N=3072 K=1024, L2-cold weights; template 45 flag-barrier megakernel, against 12.22 us for three launches"
    metric: us
    value: 11.57
    source_id: doc-kernel-design-templates
    source_locator: "45_flag_barrier_megakernel.cu STATUS block"
  - gpu: H100 SXM5
    dtype: bf16
    shape: "16 experts x (7168, 2048), top-8, 128 tokens, BLOCK_M=64; template 44 Mega MoE sm90 port"
    metric: us
    value: 673
    utilization: "2.93 TB/s streamed, 106% of the [ld.bw.dev.dram] floor (L2 hits on repeated experts)"
    source_id: doc-kernel-design-templates
    source_locator: "44_megamoe_sm90.cu STATUS block"
---

# Megakernel forms

The megakernel idiom in its four published forms, each carried by one
compilable, self-checking, self-timing sm90 template in
`doc-kernel-design-templates`:

| Reference | Form | Template |
|---|---|---|
| HazyResearch/Megakernels (MIT), `blog-hazyresearch-megakernels` | offline planner emits per-SM instruction streams; a warp-specialized interpreter runs them | `42_hazy_llama_megakernel.cu` |
| Mirage Persistent Kernel (Apache-2.0), `doc-mirage-mpk` | compiler emits tasks and events; worker CTAs and scheduler warps run them, across decode steps | `43_mpk_task_graph_runtime.cu` |
| DeepGEMM Mega MoE (MIT), `blog-deepgemm` | two dependent GEMM stages over one worker pool, with a scheduler that stays ahead of the dependency | `44_megamoe_sm90.cu` (sm90 port; upstream ships sm100 only) |
| learn-cuda 12_megakernel, `blog-learn-cuda-megakernel` | one launch, phases split by a global-memory flag | `45_flag_barrier_megakernel.cu` |

Templates 42 and 43 run the same Llama-1B decode step, so the static planner
and the dynamic scheduler are compared on one workload. Template 40 is the
reduced interpreter that loses to plain launches, kept as the ledger of why.

## Portable design rules

- **The dependency wait must sit under the weight stream, never in front of
  it.** Weights have no dependency; only the activation does. A loader that
  streams the next instruction's weights while the consumer spins on the
  counter hides the hop [atom.lat.dev.hop] and the TMA ramp; an interpreter
  that checks the dependency before issuing any copy pays fetch, wait, arm,
  issue in series per instruction, and loses to launches at any depth. This
  is the difference between template 40 and template 42.
- **Schedule as data, dispatch as code.** Both winning forms put the schedule
  in memory (per-SM instruction table; task and event tables) and keep the
  kernel a fixed interpreter. Topological order per worker plus counters that
  only count up is the deadlock-freedom argument; every bounded resource
  added on top (pages, rings, accumulator slots) needs its own sizing
  argument.
- **Pre-launch beats launch-on-fire.** MPK's current runtime demotes every
  launch event to a counter and pre-enqueues the whole graph round robin;
  letting events launch their dependents costs a scheduler hop per task and
  ran about 1.5x slower on the same graph. Keep the JIT path for graphs whose
  shape is data-dependent.
- **Give every phase at least an SM's worth of parallelism.** GQA attention
  written as one instruction per KV head is 8 instructions on 132 SMs; the
  weight prefetch fills about 3 stages and then the machine idles. Splitting
  keys (a partial plus LSE-reduction pair) recovered about 15% of the whole
  step.
- **Publish once per instruction.** Wait for the read of shared memory per
  block, and for the writes once at the end, then bump every counter the
  instruction touched; waiting for each block's global write to land before
  bumping serializes the output pipeline at microseconds per block.
- **Pages cross instruction boundaries in an op-declared order**, and every
  instruction releases every page exactly once, unused ones on entry.
- **Two dependent stages over one pool need a warmup.** Mega MoE issues
  L1-only waves sized from the per-block L1:L2 task ratio before
  interleaving; ring-buffer counters are cumulative and nothing is reset
  mid-kernel.
- **Make the epilogue's data layout with the weights.** Interleaving gate and
  up rows at granularity 8 puts (gate, up) of one feature in the same
  thread's accumulator registers, so SwiGLU needs no shuffle; apply the top-k
  weight there, so the combine is a plain sum.
- **A flag barrier costs what a launch boundary costs.** At 10-30 us kernels,
  replacing four launches with four release/acquire flags moved nothing
  measurable. The flag form is worth having for the fusions it enables
  (redundant norms, no scratch round trips), not for the ramp it removes.
- **Cache-policy hints on a decode weight stream are free and null on sm90.**
  `L1::no_allocate` and `L2::cache_hint(evict_first)` measured within noise
  of plain loads; `ld.global.L2::evict_first` is not even legal on sm_90a.
- **An expert is streamed once per pool block.** Weight traffic is blocks x
  expert bytes, not experts x expert bytes; choose the largest BLOCK_M that
  keeps each expert in one block before tuning anything in the kernel.
- **A megakernel removes launches, not traffic.** Data still travels between
  instructions through global memory exactly as it would between kernels;
  if a chain is bandwidth-bound rather than launch-bound the machine buys
  nothing. Price it first (`pattern-fusion-latency-chain`).
- **The register tax is the usual killer.** Consumer warps are sized once,
  for the maximum over every op the machine can run; one op that wants a
  big accumulator makes every instruction expensive. Splitting that op back
  out into its own kernel is a legitimate answer and often the right one.
- **Deadlock freedom is two properties, and it breaks the moment a bounded
  resource is grabbed on demand rather than owned by a ring slot**: CTAs
  claim instructions monotonically from a topologically ordered program, so
  the lowest-indexed instruction in flight always has its predecessors
  complete and some CTA can advance. Pages tied to the ring stage are the
  version that is obviously correct; op-declared release order is faster and
  needs the every-page-released-once invariant.
- **Event granularity is a compiler decision.** An edge between two ops is
  cut into as many events as the consumer can start on; cutting qkv to
  attention per KV head needs the QKV weight rows laid out so a head's q, k
  and v rows are contiguous, the kind of layout choice made so partitions
  line up.
- **Cross-task weight pre-loading measured null here.** Bulk L2 prefetch of
  the next task's weights before spinning on the current task's event
  reaches only one task ahead inside a fetch batch, and the tasks that would
  profit are the short latency-bound ones; upstream reports 1.2-1.3x on its
  linear tasks.
- **A one-warp scheduler feeds several consumers through a shared-memory
  pipeline**: a two-slot ring with a full barrier of one arrival and an
  empty barrier of one elected arrival per consumer is what decouples them.
- Budget the ledger before building: launches removed x
  [launch.lat.dev.ramp] against hops added x [atom.lat.dev.hop], per
  `pattern-fusion-latency-chain`; the decode-step templates reach 70-85% of
  the measured bandwidth floor [ld.bw.dev.dram], the range their upstream
  projects report, and the Mega MoE stream runs at the TMA rate
  [tma.bw.dev.dram].

## Source-backed fragment

The loader principle that makes a single loader warp viable, from
`sm90_common.cuh` in `doc-kernel-design-templates`: one thread moves the
whole span and the copy engine does the work.

```cpp
__device__ __forceinline__ void bulk_load_1d(void* smem_dst, const void* gmem_src,
                                             uint32_t bytes, uint64_t* full) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      ::"r"(smem_u32(smem_dst)), "l"(gmem_src), "r"(bytes), "r"(smem_u32(full))
      : "memory");
}
```

## Performance boundary

Every number above is a template's own STATUS measurement on this
repository's H100 with unpinned clocks, reported as a fraction of the
measured bandwidth floor. Upstream speedups are serving workloads on their
own machines and are kept beside, not beneath, those measurements.
