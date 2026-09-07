# Query: By Template

> Auto-generated. Do not edit manually. A row lists the wiki pages whose
> body names a file of the sm90-templates bundle (an excerpt, a STATUS
> block, or a design check); the bundle README points here instead of
> keeping its own column.

| Template | Pages that cite it |
|----------|--------------------|
| `03_wgmma_mainloop.cu` | [Asynchronous warpgroup MMA (wgmma)](../wiki/hardware/wgmma.md), [Release ring frames on wgmma retirement, before the epilogue](../wiki/techniques/release-on-retirement.md) |
| `04_epilogue_persistent.cu` | [Publish a partial with one bulk store from a staged shared-memory image](../wiki/techniques/bulk-store-publish.md) |
| `11_fp8_two_level_accum.cu` | [sm90 has no sub-8-bit tensor core: unpack INT4 and FP4 weights in registers](../wiki/techniques/register-unpack-sub-byte.md) |
| `12_attention_online_softmax.cu` | [FlashAttention-3 (Hopper): warp specialization and pingpong, and when to reach for it](../wiki/kernels/flash-attention-3.md) |
| `13_mla_decode_split_kv.cu` | [Prefetch across the dependency, not across the slot](../wiki/techniques/prefetch-across-dependency.md), [A reduction is its own task kind, never "split 0 folds"](../wiki/techniques/reduction-own-task-kind.md) |
| `20_marlin_w4a16.cu` | [sm90 has no sub-8-bit tensor core: unpack INT4 and FP4 weights in registers](../wiki/techniques/register-unpack-sub-byte.md) |
| `22_w4a8_gemm.cu` | [sm90 has no sub-8-bit tensor core: unpack INT4 and FP4 weights in registers](../wiki/techniques/register-unpack-sub-byte.md) |
| `40_megakernel_interpreter.cu` | [Megakernel forms: planner interpreter, task-graph runtime, Mega MoE, flag barrier](../wiki/kernels/megakernel-forms.md) |
| `42_hazy_llama_megakernel.cu` | [Megakernel forms: planner interpreter, task-graph runtime, Mega MoE, flag barrier](../wiki/kernels/megakernel-forms.md) |
| `43_mpk_task_graph_runtime.cu` | [Megakernel forms: planner interpreter, task-graph runtime, Mega MoE, flag barrier](../wiki/kernels/megakernel-forms.md) |
| `44_megamoe_sm90.cu` | [Megakernel forms: planner interpreter, task-graph runtime, Mega MoE, flag barrier](../wiki/kernels/megakernel-forms.md) |
| `45_flag_barrier_megakernel.cu` | [Megakernel forms: planner interpreter, task-graph runtime, Mega MoE, flag barrier](../wiki/kernels/megakernel-forms.md), [PTX Cache Policy Differentiation](../wiki/techniques/cache-policy.md) |
| `elementwise_sm90.cuh` | [Glue ops are bound by row traversals and launches: fuse to remove a traversal, bracket with PDL](../wiki/techniques/row-traversal-fusion.md) |
| `quant_sm90.cuh` | [sm90 has no sub-8-bit tensor core: unpack INT4 and FP4 weights in registers](../wiki/techniques/register-unpack-sub-byte.md) |
| `sm90_common.cuh` | [Thread-block clusters and distributed shared memory (DSMEM)](../wiki/hardware/cluster-dsmem.md), [Megakernel forms: planner interpreter, task-graph runtime, Mega MoE, flag barrier](../wiki/kernels/megakernel-forms.md), [Cluster barriers: place them, count them, scope them](../wiki/techniques/cluster-barrier-placement.md), [Fuse the producer chain; overlap the consumer with PDL](../wiki/techniques/producer-fusion-pdl.md), [Apply per-K factors to the register fragment, not in shared memory](../wiki/techniques/scale-on-register-fragment.md), [One 3-D TMA box loads a row-major deep-K tile](../wiki/techniques/tma-3d-box-row-major.md) |
