---
id: ext-mpk-megakernel
type: external
arch: sm90
tags: [megakernel, task-graph, persistent-kernel, planner, scheduler, moe]
confidence: measured
---

# Megakernel idioms: four references, four templates

## What to read it for

The megakernel idiom in its four published forms, each carried by one
compilable, self-checking, self-timing template in
[`../templates/`](../templates/README.md):

| Reference | Form | Template |
|---|---|---|
| HazyResearch/Megakernels (MIT) | offline planner emits per-SM instruction streams; a warp-specialized interpreter runs them | `42_hazy_llama_megakernel.cu` |
| Mirage Persistent Kernel, MPK (Apache-2.0) | compiler emits tasks and events; worker CTAs and scheduler warps run them, across decode steps | `43_mpk_task_graph_runtime.cu` |
| DeepGEMM Mega MoE (MIT) | two dependent GEMM stages over one worker pool, with a scheduler that stays ahead of the dependency | `44_megamoe_sm90.cu` |
| gau-nernst/learn-cuda 12_megakernel | one launch, phases split by a global-memory flag | `45_flag_barrier_megakernel.cu` |

Templates 42 and 43 run the same Llama-1B decode step, so the static
planner and the dynamic scheduler are compared on one workload. Template
40 is the reduced interpreter that loses to plain launches, kept as the
ledger of why.

## Portable design rules for the idiom

- **The dependency wait must sit UNDER the weight stream, never in front of
  it.** Weights have no dependency; only the activation does. A loader that
  streams the next instruction's weights while the consumer spins on the
  counter hides the hop [atom.lat.dev.hop] and the TMA ramp; an interpreter
  that checks the dependency before issuing any copy pays fetch -> wait ->
  arm -> issue in series per instruction, and loses to launches at any
  depth. This is the difference between template 40 and template 42, and
  it is where a megakernel is won.
- **Schedule as data, dispatch as code.** Both winning forms put the
  schedule in memory (per-SM instruction table; task/event tables) and keep
  the kernel a fixed interpreter. Topological order per worker plus
  counters that only count up is the deadlock-freedom argument; every
  bounded resource added on top (pages, rings, accumulator slots) needs its
  own sizing argument.
- **Pre-launch beats launch-on-fire.** MPK's current runtime demotes every
  launch event to a counter and pre-enqueues the whole graph round robin;
  letting events launch their dependents instead costs a scheduler hop per
  task and ran ~1.5x slower on the same graph. Keep the JIT path for graphs
  whose shape is data-dependent, not for a fixed decode step.
- **Give every phase at least an SM's worth of parallelism.** GQA attention
  written as one instruction per KV head is 8 instructions on 132 SMs; the
  weight prefetch fills ~3 stages and then the machine idles for the rest
  of the phase. Splitting keys (a partial + LSE-reduction pair) recovered
  ~15% of the whole step. Size splits so attention has >= SM-count
  instructions per layer.
- **Publish once per instruction.** A storer that waits for each block's
  global write to land (`cp.async.bulk.wait_group 0`) before bumping a
  counter serialises the output pipeline at microseconds per block; wait
  for the READ of shared memory per block, and for the writes once at the
  end, then bump every counter the instruction touched.
- **Pages cross instruction boundaries in an op-declared order.** Each op
  names the order it frees its logical pages; the controller maps the next
  instruction's pages onto those. The invariant that makes it sound is that
  every instruction releases every page exactly once, unused ones on
  entry.
- **Two dependent stages over one pool need a warmup.** Mega MoE issues
  L1-only waves sized from the per-block L1:L2 task ratio before
  interleaving, and an L2 task additionally waits until the L1 task
  COUNTER has passed its block. Ring-buffer counters are cumulative and
  their targets scale with the slot's use count; nothing is reset
  mid-kernel.
- **Make the epilogue's data layout with the weights.** Interleaving gate
  and up rows at granularity 8 puts (gate, up) of one feature in the same
  thread's accumulator registers; SwiGLU then needs no shuffle. Apply the
  top-k weight there, so the combine is a plain sum.
- **A flag barrier costs what a launch boundary costs.** At 10-30 us
  kernels, replacing four launches with four release/acquire flags moved
  nothing measurable; replacing two launches on a bandwidth-bound MLP saved
  under a microsecond. The flag form is worth having for the fusions it
  enables (redundant norms, no scratch round trips), not for the ramp it
  removes.
- **Cache-policy hints on a decode weight stream are free and null on
  sm90.** `L1::no_allocate` and `L2::cache_hint(evict_first)` measured
  within noise of plain loads; `ld.global.L2::evict_first` is not even
  legal on sm_90a (256-bit loads only). Use them for hygiene, do not expect
  a number.
- Budget the ledger before building: launches removed x
  [launch.lat.dev.ramp] against hops added x [atom.lat.dev.hop], per
  [fusion-economics](fusion-economics.md); the decode-step templates reach
  70-85% of the measured bandwidth floor [ld.bw.dev.dram], which is the
  range their upstream projects report, and the Mega MoE stream runs at the
  TMA rate [tma.bw.dev.dram].
- **An expert is streamed once per pool block.** In Mega MoE the weight
  traffic is blocks x expert bytes, not experts x expert bytes; choose the
  largest BLOCK_M that keeps each expert in one block (upstream picks it per
  call, from candidates up to 192) before tuning anything in the kernel.

## Source

- HazyResearch/Megakernels: https://github.com/HazyResearch/Megakernels
  (blog: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Mirage Persistent Kernel: https://github.com/mirage-project/mirage
  (paper: https://arxiv.org/abs/2512.22219)
- DeepGEMM Mega MoE: `third_party/deepgemm/deep_gemm/include/deep_gemm/{scheduler,layout}/mega_moe.cuh`,
  `impls/sm100_bf16_mega_moe.cuh`; benchmarks in
  https://github.com/deepseek-ai/DeepGEMM/pull/316. No sm90 source exists
  upstream; template 44 is the sm90 port.
- gau-nernst/learn-cuda `12_megakernel`: https://github.com/gau-nernst/learn-cuda
  (no license file at the revision read; distilled, not excerpted)

Their speedups are serving workloads on their machines; each template's
header carries the upstream numbers next to its own STATUS block.
