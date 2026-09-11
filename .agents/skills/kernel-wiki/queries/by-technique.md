# Query: By Technique

> Auto-generated. Do not edit manually.

| Technique | Tags | Architectures | Confidence | Reproducibility | Sources |
|-----------|------|--------------|------------|-----------------|---------|
| [A reduction is its own task kind, never "split 0 folds"](../wiki/techniques/reduction-own-task-kind.md) | task-loop, split-k, reduction, pdl | sm90 | measured | snippet | 3 |
| [Apply per-K factors to the register fragment, not in shared memory](../wiki/techniques/scale-on-register-fragment.md) | wgmma, ldmatrix, register-fragments, per-k-specialization | sm90 | measured | snippet | 2 |
| [CCCL CUB SM100 Scan Tuning](../wiki/techniques/cccl-memory-primitives.md) | cuda-cpp, parallel-scan, vectorized-loads, tile-scheduling | sm100 | source-reported | snippet | 1 |
| [Chunk-Based Parallelism for Linear Recurrent Models](../wiki/techniques/chunk-parallelism.md) | chunk-parallelism, linear-attention, triton | sm90 | source-reported | snippet | 2 |
| [Cluster barriers: place them, count them, scope them](../wiki/techniques/cluster-barrier-placement.md) | cluster, dsmem, mbarrier, barrier-placement | sm90 | measured | snippet | 4 |
| [Double/Multi-Buffering Patterns](../wiki/techniques/double-buffering.md) | double-buffering, tmem, pipeline-stages | sm100, sm90 | source-reported | snippet | 3 |
| [Epilogue fusion](../wiki/techniques/epilogue-fusion.md) | epilogue-fusion, tmem, warp-specialization | sm100, sm90 | source-reported | snippet | 2 |
| [External Source-Map Research For Kernel Edits](../wiki/techniques/external-source-map-research.md) | cuda-cpp, cute-dsl, tma, wgmma | sm100, sm90 | source-reported | snippet | 5 |
| [Fine-grained FP8/FP4 scaling](../wiki/techniques/fine-grained-quantization.md) | fine-grained-quantization, fp8, fp4, nvfp4 | sm100, sm90 | source-reported | snippet | 3 |
| [Fuse the producer chain; overlap the consumer with PDL](../wiki/techniques/producer-fusion-pdl.md) | pdl, griddepcontrol, producer-fusion, kernel-fusion | sm90 | measured | snippet | 4 |
| [Glue ops are bound by row traversals and launches: fuse to remove a traversal, bracket with PDL](../wiki/techniques/row-traversal-fusion.md) | kernel-fusion, vectorized-loads, pdl, griddepcontrol | sm90 | measured | snippet | 3 |
| [Kernel fusion](../wiki/techniques/kernel-fusion.md) | kernel-fusion, fused-kernel, tmem | sm100, sm90 | source-reported | snippet | 4 |
| [One 3-D TMA box loads a row-major deep-K tile](../wiki/techniques/tma-3d-box-row-major.md) | tma, tma-box-geometry, swizzling, wgmma | sm90 | measured | snippet | 3 |
| [PTX Cache Policy Differentiation](../wiki/techniques/cache-policy.md) | cache-policy, vectorized-loads | sm100, sm90 | source-reported | snippet | 6 |
| [Persistent Kernels with Cluster Launch Control](../wiki/techniques/persistent-kernels.md) | persistent-kernel, clc, tile-scheduling | sm100 | source-reported | snippet | 4 |
| [Ping-Pong Scheduling](../wiki/techniques/ping-pong-scheduling.md) | ping-pong-scheduling, warp-specialization, tmem, pipeline-stages | sm100 | source-reported | snippet | 2 |
| [Prefetch across the dependency, not across the slot](../wiki/techniques/prefetch-across-dependency.md) | task-loop, frame-level-dependency, data-reuse, tma | sm90 | measured | snippet | 2 |
| [Publish a partial with one bulk store from a staged shared-memory image](../wiki/techniques/bulk-store-publish.md) | cp-async-bulk, bulk-store, proxy-fence, task-loop | sm90 | measured | snippet | 3 |
| [Register budgeting](../wiki/techniques/register-budgeting.md) | register-budgeting, register-reuse | sm100, sm90 | source-reported | snippet | 3 |
| [Release ring frames on wgmma retirement, before the epilogue](../wiki/techniques/release-on-retirement.md) | wgmma, mbarrier, pipeline-stages, double-buffering | sm90 | measured | snippet | 4 |
| [Same-process A/B/A, min under unpinned clocks, and the in-graph regime](../wiki/techniques/same-process-aba.md) | measurement, ablation, persistent-kernel, cache-policy | sm90 | measured | snippet | 4 |
| [Shared Memory Swizzling](../wiki/techniques/swizzling.md) | swizzling, shared-memory-optimization, tma | sm100, sm90 | source-reported | snippet | 3 |
| [Software Pipelining and Multi-Stage Buffering](../wiki/techniques/pipeline-stages.md) | pipeline-stages, double-buffering, tma, mbarrier | sm100, sm90 | source-reported | snippet | 4 |
| [Software-Emulated Exponential](../wiki/techniques/software-exp.md) | software-exp, attention | sm100 | source-reported | snippet | 2 |
| [The PDL wait is derived; the trigger has to be swept](../wiki/techniques/pdl-placement.md) | pdl, griddepcontrol, pdl-placement, ablation | sm90 | source-reported | snippet | 4 |
| [Tile Scheduling Strategies](../wiki/techniques/tile-scheduling.md) | tile-scheduling, clc, persistent-kernel | sm100, sm90 | source-reported | snippet | 3 |
| [Two A fragments by stage parity, fully unrolled: the fix for ptxas C7518](../wiki/techniques/wgmma-rs-fragment-parity.md) | wgmma, register-fragments, loop-unrolling, ldmatrix | sm90 | measured | snippet | 4 |
| [Warp Specialization on Hopper and Blackwell](../wiki/techniques/warp-specialization.md) | warp-specialization, tcgen05, tmem | sm100, sm90 | source-reported | snippet | 4 |
| [Wide Vectorized Loads and Cache Policies](../wiki/techniques/vectorized-loads.md) | vectorized-loads, cache-policy, register-budgeting | sm100, sm90 | source-reported | snippet | 4 |
| [sm90 has no sub-8-bit tensor core: unpack INT4 and FP4 weights in registers](../wiki/techniques/register-unpack-sub-byte.md) | quantization, fine-grained-quantization, mma-sync, register-fragments | sm90 | source-reported | snippet | 2 |
