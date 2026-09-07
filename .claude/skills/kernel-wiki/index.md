# Hopper Kernel Optimization Knowledge Base

> Knowledge base for GPU kernel optimization on NVIDIA Hopper (SM90), with
> Blackwell (SM100) as the port-forward appendix. Built on KernelWiki.
> **For Claude Code agents**: this directory is a skill — see [SKILL.md](SKILL.md);
> schema and conventions in [CLAUDE.md](CLAUDE.md).

## Recommended Query Tools (for LLM agents)

```bash
.venv/bin/python scripts/query.py --symptom <stall-reason> --compact
.venv/bin/python scripts/query.py "<natural language>" [--architecture sm90] [--type <kernel|technique|pattern|hardware>]
.venv/bin/python scripts/get_page.py <page-id-or-path> [--follow-sources]
.venv/bin/python scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

See [references/examples.md](references/examples.md) for worked query patterns.

## Quick Navigation

| I want to... | Go to |
|---|---|
| Turn a Nsight Compute finding into a move | [queries/by-problem.md](queries/by-problem.md) |
| Learn a specific technique | [queries/by-technique.md](queries/by-technique.md) |
| Use a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| Write a specific kernel type | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Stay on sm90, or read the Blackwell appendix | [queries/by-architecture.md](queries/by-architecture.md) |
| Use a specific language/DSL | [queries/by-language.md](queries/by-language.md) |
| See what a repo contributed | [queries/by-repo.md](queries/by-repo.md) |

## Hopper Hardware

- [hw-wgmma](wiki/hardware/wgmma.md) — asynchronous warpgroup MMA: batches, RS/SS operands, the N floor, the stage knee
- [hw-tma](wiki/hardware/tma.md) — Tensor Memory Accelerator (async bulk loads, multicast)
- [hw-mbarrier](wiki/hardware/mbarrier.md) — arrival and transaction counts, phase parity
- [hw-cluster-dsmem](wiki/hardware/cluster-dsmem.md) — clusters, DSMEM, cluster barriers and their placement limit
- [hw-pdl-gdc](wiki/hardware/pdl-gdc.md) — programmatic dependent launch

## Hopper Techniques (this repository's measured pages)

- [technique-release-on-retirement](wiki/techniques/release-on-retirement.md) — free ring frames on wgmma retirement, before the epilogue
- [technique-wgmma-rs-fragment-parity](wiki/techniques/wgmma-rs-fragment-parity.md) — the fix for ptxas C7518
- [technique-scale-on-register-fragment](wiki/techniques/scale-on-register-fragment.md) — per-K factors on the register fragment, not in smem
- [technique-tma-3d-box-row-major](wiki/techniques/tma-3d-box-row-major.md) — one 3-D box loads a row-major deep-K tile
- [technique-bulk-store-publish](wiki/techniques/bulk-store-publish.md) — publish a partial with one bulk store
- [technique-cluster-barrier-placement](wiki/techniques/cluster-barrier-placement.md) — place, count and scope cluster barriers
- [technique-pdl-placement](wiki/techniques/pdl-placement.md) — the wait is derived, the trigger is swept
- [technique-producer-fusion-pdl](wiki/techniques/producer-fusion-pdl.md) — fuse the producer chain, overlap the consumer
- [technique-prefetch-across-dependency](wiki/techniques/prefetch-across-dependency.md) — frame-level, not task-level, dependencies
- [technique-reduction-own-task-kind](wiki/techniques/reduction-own-task-kind.md) — a reduction is its own parallel stage
- [technique-same-process-aba](wiki/techniques/same-process-aba.md) — A/B/A, `min`, the in-graph regime

## Hopper Diagnosis Patterns

- [pattern-serialized-wgmma](wiki/patterns/serialized-wgmma.md) — `gmma` stalls at ~3x the issue floor
- [pattern-wgmma-tile-n-floor](wiki/patterns/wgmma-tile-n-floor.md) — a wgmma below N=64 is shared-memory-bound
- [pattern-cold-burst-ceiling](wiki/patterns/cold-burst-ceiling.md) — a cold weight stream is bounded by the burst curve
- [pattern-epilogue-bound-short-k](wiki/patterns/epilogue-bound-short-k.md) — a flat tile sweep at short K
- [pattern-serial-epilogue-owner](wiki/patterns/serial-epilogue-owner.md) — widening a tile lengthens one owner's chain
- [pattern-fusion-latency-chain](wiki/patterns/fusion-latency-chain.md) — no eligible warp, single-digit DRAM: a latency chain
- [pattern-pdl-primary-is-a-resource](wiki/patterns/pdl-primary-is-a-resource.md) — folding the small primary away regresses the chain
- [pattern-layout-production-cost](wiki/patterns/layout-production-cost.md) — a layout's producer must fit under the gain
- [pattern-isolated-timer-overstates-fusion](wiki/patterns/isolated-timer-overstates-fusion.md) — cold isolated timing vs in-graph
- [pattern-persistent-kernel-timing-artifacts](wiki/patterns/persistent-kernel-timing-artifacts.md) — replay tails and cross-job noise
- [pattern-one-sided-gradient](wiki/patterns/one-sided-gradient.md) — price the direction you intend to move
- [pattern-stacked-floors](wiki/patterns/stacked-floors.md) — two floors make every single lever read as null

KernelWiki's generic patterns (compute-bound, memory-bound, pipeline stalls,
register pressure, tail effect, low SM utilization, MoE imbalance) are in the
same directory and the same index.

## Kernel Case Studies

- [kernel-flash-attention-3](wiki/kernels/flash-attention-3.md) — Hopper warp specialization and pingpong, and when to reach for it
- [kernel-megakernel-forms](wiki/kernels/megakernel-forms.md) — planner interpreter, task-graph runtime, Mega MoE, flag barrier: four templates, measured
- [kernel-deepgemm](wiki/kernels/deepgemm.md) — DeepGEMM, with the sm90 choreography section
- [kernel-flashmla](wiki/kernels/flashmla.md) — FlashMLA, with the sm90 decode structure section
- [kernel-grouped-gemm](wiki/kernels/grouped-gemm.md), [kernel-fused-moe](wiki/kernels/fused-moe.md), [kernel-fp8-block-scale-gemm](wiki/kernels/fp8-block-scale-gemm.md), [kernel-gated-dual-gemm](wiki/kernels/gated-dual-gemm.md), [kernel-gated-delta-net](wiki/kernels/gated-delta-net.md), [kernel-sparse-mla](wiki/kernels/sparse-mla.md)

## Blackwell Appendix

- Hardware: [hw-tcgen05-mma](wiki/hardware/tcgen05-mma.md), [hw-tmem](wiki/hardware/tmem.md), [hw-clc](wiki/hardware/clc.md), [hw-2sm-cooperative](wiki/hardware/2sm-cooperative.md), [hw-nvfp4](wiki/hardware/nvfp4.md)
- Migration: [migration-wgmma-to-tcgen05](wiki/migration/wgmma-to-tcgen05.md), [migration-register-to-tmem](wiki/migration/register-to-tmem.md)
- Kernels: [kernel-flash-attention-4](wiki/kernels/flash-attention-4.md), [kernel-nvfp4-gemm](wiki/kernels/nvfp4-gemm.md), [kernel-nvfp4-gemv](wiki/kernels/nvfp4-gemv.md)
- Languages: [lang-cuda-cpp](wiki/languages/cuda-cpp.md), [lang-cute-dsl](wiki/languages/cute-dsl.md), [lang-ptx](wiki/languages/ptx-sm100.md), [lang-triton](wiki/languages/triton-blackwell.md)

## Sources this repository adds

- [doc-ptx-isa-sm90](sources/docs/ptx-isa-sm90.md) — the PTX ISA sections behind the sm90 pages
- [doc-flash-attention-3](sources/docs/flash-attention-3.md), [doc-mirage-mpk](sources/docs/mirage-persistent-kernel.md), [blog-hazyresearch-megakernels](sources/blogs/hazyresearch-megakernels.md), [blog-learn-cuda-megakernel](sources/blogs/learn-cuda-megakernel.md)
- The sm90 templates bundle: [README](artifacts/kernels/sm90-templates/variants/README.md), its source page [doc-kernel-design-templates](sources/docs/kernel-design-templates.md), and [queries/by-template.md](queries/by-template.md)
- Upstream methods behind the bundle: [blog-marlin](sources/blogs/marlin.md), [blog-humming](sources/blogs/humming.md), [blog-flashinfer-glue-kernels](sources/blogs/flashinfer-glue-kernels.md), [doc-cutlass-hopper](sources/docs/cutlass-hopper.md)
- Sibling skills: [doc-hardware-unit-test](sources/docs/hardware-unit-test.md), [doc-benchmark-kernel](sources/docs/benchmark-kernel.md), [doc-ncu-report](sources/docs/ncu-report.md)
