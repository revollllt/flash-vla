# Query: By Repository

> Auto-generated. Do not edit manually.

<a id="dao-ailabflash-attention"></a>
## Dao-AILab/flash-attention
2 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#2441](../sources/prs/flash-attention/PR-2441.md) | [Cute,Sm100,Fwd] add MLA 64/512 with topk sparsity for MQA 128 heads | 2026-04-06 | persistent-kernel, pipeline-stages | mla, tcgen05, topk |
| [#1236](../sources/prs/flash-attention/PR-1236.md) | FA3 kvcache + split kv + gqa parallelization | 2024-09-18 | tile-scheduling | fp8, gemm, tma |

<a id="nvidiatensorrt-llm"></a>
## NVIDIA/TensorRT-LLM
2 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#13340](../sources/prs/TensorRT-LLM/PR-13340.md) | [None][feat] Integrate FP4 indexer for DSA on Blackwell | 2026-04-22 | kernel-fusion | attention, fp4, fp8 |
| [#11897](../sources/prs/TensorRT-LLM/PR-11897.md) | [TRTLLM-10990][feat] Fuse SwiGLU and quant into shared expert | 2026-03-04 | kernel-fusion | fp4, fp8, gemm |

<a id="nvidiacccl"></a>
## NVIDIA/cccl
2 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#3559](../sources/prs/cccl/PR-3559.md) | Add b200 tunings for scan.exclusive.sum | 2025-01-28 |  |  |
| [#3517](../sources/prs/cccl/PR-3517.md) | Fix the vectorized loading of BlockLoad | 2025-01-24 |  | topk |

<a id="nvidiacutlass"></a>
## NVIDIA/cutlass
5 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#3106](../sources/prs/cutlass/PR-3106.md) | [CLI] add cutedsl fp16 gemm tutorial from 2 to 6 | 2026-03-13 | pipeline-stages, swizzling, tile-scheduling | clc, gemm, quantization |
| [#3021](../sources/prs/cutlass/PR-3021.md) | [Cute-DSL] Add option for issue_clc_query without multicast | 2026-02-11 | swizzling | clc |
| [#2881](../sources/prs/cutlass/PR-2881.md) | new example with TMA prefetch feature targeting for DRAM latency boun… | 2025-12-16 | pipeline-stages, swizzling, tile-scheduling | fp8, gemm, quantization |
| [#2161](../sources/prs/cutlass/PR-2161.md) | Blockwise Improvement and Programmatic Dependent Launch | 2025-03-10 |  | gemm |
| [#2139](../sources/prs/cutlass/PR-2139.md) | Blockwise and Groupwise GEMM for Blackwell and Improvements for Hopper | 2025-02-26 | swizzling, tile-scheduling, warp-specialization | block-scale, clc, fp8 |

<a id="deepseek-aideepgemm"></a>
## deepseek-ai/DeepGEMM
1 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#304](../sources/prs/DeepGEMM/PR-304.md) | [Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes | 2026-04-16 | pipeline-stages, swizzling | block-scale, fp4, fp8 |

<a id="flashinfer-aiflashinfer"></a>
## flashinfer-ai/flashinfer
1 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#1039](../sources/prs/flashinfer/PR-1039.md) | [nvidia] initial support for blackwell kernels | 2025-04-24 |  | attention, flash-attention, tma |

<a id="sgl-projectsglang"></a>
## sgl-project/sglang
2 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#22079](../sources/prs/sglang/PR-22079.md) | [nvidia] Gemma4 nvfp4 fix | 2026-04-03 |  | attention, fp4, fp8 |
| [#21019](../sources/prs/sglang/PR-21019.md) | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | 2026-03-20 | kernel-fusion |  |

<a id="vllm-projectvllm"></a>
## vllm-project/vllm
2 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#34597](../sources/prs/vllm/PR-34597.md) | [Kernel] Add FP8 KV cache support to Triton MLA decode attention | 2026-02-16 |  | attention, fp8, mla |
| [#16032](../sources/prs/vllm/PR-16032.md) | [NVIDIA] Support Cutlass MLA for Blackwell GPUs | 2025-04-03 | tile-scheduling | attention, flash-attention, fp4 |

