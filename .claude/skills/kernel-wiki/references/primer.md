# Topic Map / Primer

A compact map of the knowledge base. Use it when the question is broad; each
row says which page to open. Page ids resolve via `get_page.py <id>`; paths
are relative to the wiki root. Hopper (sm90) rows come first; the Blackwell
appendix is at the end.

---

## Hopper Hardware Features (SM90)

| Feature | Page ID | Path | Notes |
|---|---|---|---|
| wgmma | `hw-wgmma` | `wiki/hardware/wgmma.md` | Warpgroup-collective async MMA; SS/RS operands; fence/arrive/commit/wait batches; N floor and stage knee by tag. |
| TMA | `hw-tma` | `wiki/hardware/tma.md` | Descriptor-driven bulk tensor copies; multicast; swizzle must agree with the smem layout. |
| mbarrier | `hw-mbarrier` | `wiki/hardware/mbarrier.md` | Arrival count and transaction count; phase parity derived from the loop index. |
| Clusters / DSMEM | `hw-cluster-dsmem` | `wiki/hardware/cluster-dsmem.md` | `mapa`, peer arrives, cluster barriers; co-residency limit by tag. |
| PDL / GDC | `hw-pdl-gdc` | `wiki/hardware/pdl-gdc.md` | `griddepcontrol.wait` orders memory; `launch_dependents` only schedules. |

---

## Hopper Techniques

| Technique | Page ID | When to use |
|---|---|---|
| Release on retirement | `technique-release-on-retirement` | A TMA ring stalls its producers at the measured knee: free the frame on wgmma group retirement, before any epilogue work. |
| Fragment parity + unroll | `technique-wgmma-rs-fragment-parity` | ptxas prints C7518 on an RS wgmma loop; ship both halves of the fix. |
| Scale on the register fragment | `technique-scale-on-register-fragment` | A per-K factor on the A operand; never an in-smem read-modify-write. |
| 3-D TMA box | `technique-tma-3d-box-row-major` | A row-major deep-K tile (BK 128/256) that seems to need several boxes or a transpose. |
| Bulk-store publish | `technique-bulk-store-publish` | A partial published through thousands of small stores behind a release fence. |
| Cluster barrier placement | `technique-cluster-barrier-placement` | Any cluster/DSMEM design: trailing barriers are near free, hoisted ones add skew; push, do not pull; scope the arrive. |
| PDL placement | `technique-pdl-placement` | A PDL chain that did not get faster: derive the wait, sweep the trigger, record both. |
| Producer fusion + PDL | `technique-producer-fusion-pdl` | Small kernels between two heavy stages: one cooperative producer, early trigger, consumer waits at entry. |
| Prefetch across the dependency | `technique-prefetch-across-dependency` | Inputs only partly dependent: issue the producer-free frames at task start, wait only where the dependency lands. |
| Reduction as its own task kind | `technique-reduction-own-task-kind` | A split-K join or attention combine folded by split 0 or the last arriver. |
| Same-process A/B/A | `technique-same-process-aba` | Any comparison under unpinned clocks; any "fuse or not" decision (in-graph regime). |
| Warp specialization | `technique-warp-specialization` | Producer/consumer roles when profiling justifies the synchronization cost (KernelWiki, sm90+sm100). |
| Pipeline stages, double buffering, swizzling, tile scheduling, epilogue fusion, cache policy, vectorized loads, register budgeting, fine-grained quantization, kernel fusion, chunk parallelism | `technique-*` | KernelWiki's pages; `architectures:` lists sm90 where the page applies to Hopper. |

---

## Problem → Pattern (Diagnosis), Hopper

| Symptom (as `ncu-report` names it) | Pattern page | Candidate techniques |
|---|---|---|
| `gmma` / `warpgroup_arrive` stalls at ~3x the issue floor, C7518 in the log | `pattern-serialized-wgmma` | fragment parity, register-fragment scaling |
| `mio_throttle`, smem-bandwidth-bound wgmma at N=32 | `pattern-wgmma-tile-n-floor` | keep N >= 64, or `mma.sync` below the crossover |
| A cold weight stream at 40-60% of steady-state DRAM; depth, box, CTA count all null | `pattern-cold-burst-ceiling` | warmth, continuity, prefetch across the dependency |
| Short-K GEMM at ~30% of the ceiling, tile sweep flat, cuBLAS faster | `pattern-epilogue-bound-short-k` | staged store, residual through smem, library epilogue |
| Widening a split-K tile made the phase slower | `pattern-serial-epilogue-owner` | reduction as its own task kind, distributed epilogue |
| No eligible warp, single-digit DRAM and SM; fused kernel loses to launches | `pattern-fusion-latency-chain` | price boundaries vs hops; producer fusion; prefetch across the dependency |
| Folding a tiny PDL primary away regressed the chain | `pattern-pdl-primary-is-a-resource` | PDL placement, producer fusion |
| A pre-arranged layout would help but must be produced | `pattern-layout-production-cost` | producer fusion; a 3-D box that removes the need |
| A fusion wins on a cold isolated timer, loses in the graph | `pattern-isolated-timer-overstates-fusion` | same-process A/B/A in the graph regime |
| Persistent-kernel timing has a heavy tail; cross-job deltas do not reproduce | `pattern-persistent-kernel-timing-artifacts` | same-process A/B/A |
| An ablation says removing costs a lot; you plan to add | `pattern-one-sided-gradient` | measure the direction you ship |
| Every lever measures null; the last gradient looks like a pole | `pattern-stacked-floors` | phase vs machine rate, cold and warm, in one job |
| Low SM utilization, tail effect, memory-bound, compute-bound, pipeline stalls, register pressure, MoE imbalance | `pattern-*` (KernelWiki) | see `queries/by-problem.md` |

---

## Kernel Case Studies

| Kernel | Page ID | Headline | Key techniques |
|---|---|---|---|
| FlashAttention-3 | `kernel-flash-attention-3` | 740 TFLOPS FP16 H100 (paper sweep, 75%) | warp specialization, pingpong; when not to reach for it |
| Megakernel forms | `kernel-megakernel-forms` | 1.081 ms Llama-1B decode step on this H100 (template 42 STATUS) | planner interpreter, task-graph runtime, Mega MoE, flag barrier; the dependency wait under the weight stream |
| DeepGEMM | `kernel-deepgemm` | 1550 TFLOPS H800 FP8 (README, shape unstated) | persistent scheduling, fine-grained scaling; sm90 choreography section |
| FlashMLA | `kernel-flashmla` | DeepSeek V3 decode (README modes) | split-KV with a separate combine; sm90 decode structure section |
| Grouped GEMM, fused MoE, FP8 block-scale GEMM, gated dual GEMM, GatedDeltaNet, sparse MLA | `kernel-*` | KernelWiki pages | see `queries/by-kernel-type.md` |

---

## Machine constants and evidence

- A bracketed `[engine.quantity.scope]` is a `hardware-unit-test` tag; the
  number lives there with its validity range
  (`python3 .claude/skills/hardware-unit-test/scripts/constants.py --tag <t>`).
- `confidence: measured` names its evidence in `evidence_basis`; a
  `note-<stem>` source is the Agent Note holding the jobs and revisions.
- Numbers in a template's `STATUS` block are the only kernel numbers a page
  quotes as this repository's own; everything else is upstream's and says so.

---

## Canonical Aliases

| User-typed | Canonical |
|---|---|
| Hopper, H100, H200, H800, SM90 | `sm90` |
| WGMMA, wgmma.mma_async | `wgmma` |
| TMA, tensor memory accelerator, cp.async.bulk | `tma` |
| PDL, programmatic dependent launch | `pdl` |
| UMMA, tensor_core_gen05, tcgen05.mma | `tcgen05` |
| TMEM, tensor memory | `tmem` |
| CLC, Cluster Launch Control | `clc` |
| B200, GB200, SM100 | `sm100` |
| MoE, mixture of experts | `moe` |
| MLA, multi-head latent attention | `mla` |

`query.py` applies these automatically when scoring and when using `--tag`
or `--architecture`.

---

## Confidence & Evidence

See [`schema.md`](schema.md); the one-line rule for an answer: a `measured` page names its evidence, a `source-reported` number is re-measured before it prices anything, `inferred` and `experimental` are flagged when quoted.

---

## Blackwell Appendix (SM100)

| Feature / page | Page ID |
|---|---|
| tcgen05 MMA, TMEM, CLC, TMA, 2-SM cooperative, NVFP4 | `hw-tcgen05-mma`, `hw-tmem`, `hw-clc`, `hw-tma`, `hw-2sm-cooperative`, `hw-nvfp4` |
| wgmma → tcgen05, register accumulators → TMEM | `migration-wgmma-to-tcgen05`, `migration-register-to-tmem` |
| FlashAttention-4, NVFP4 GEMM / GEMV, MLA top-k, TensorRT-LLM indexer | `kernel-flash-attention-4`, `kernel-nvfp4-gemm`, `kernel-nvfp4-gemv`, `kernel-flash-attention-sm100-mla-topk`, `kernel-tensorrt-llm-blackwell-indexer` |
| CuTe DSL, CUDA C++, PTX (SM100), Triton on Blackwell | `lang-cute-dsl`, `lang-cuda-cpp`, `lang-ptx`, `lang-triton` |
| Contests | `sources/contests/` (GPU Mode NVFP4, FlashInfer MLSys 2026) |
