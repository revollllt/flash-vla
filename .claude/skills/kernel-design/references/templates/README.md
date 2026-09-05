# sm90 Templates — compilable skeletons for the choreography

Five tiers. **01-04** are the mechanism ladder: each adds one primitive to the
previous one, and together they are a whole sm90 pipeline. **10-14** are kernel
archetypes: the shape a real high-performance kernel of that family has, with
the decisions that make it fast stated as rules. **20-23** are the
mixed-precision family, where the quantization format drives the kernel, and
**30-33** are the memory-bound glue ops between the GEMMs, and **40-45** are
the fusion endgame. Start
at the template nearest the task, drop to the ladder when a mechanism in it is
unfamiliar.

## Tier 1 — the mechanism ladder

| Template | Adds | Wiki entries it makes concrete |
|---|---|---|
| `01_tma_mbarrier_ring.cu` | tensor map, ring of frames, full/empty barriers, derived phase parity | `tma-3d-box-row-major` |
| `02_warp_specialization.cu` | producer/math warpgroup split, `setmaxnreg` handoff, named barriers | `ext-fa3-pingpong`, `release-on-retirement` |
| `03_wgmma_mainloop.cu` | swizzle agreement, CuTe descriptors, fence/arrive/commit/wait batch | `wgmma-tile-n-floor`, `c7518-wgmma-serialization` |
| `04_epilogue_persistent.cu` | cross-proxy publish, bulk-store groups, persistent task striding | `bulk-store-publish`, `measuring-persistent-kernels`, `serial-epilogue-owner` |

## Tier 2 — kernel archetypes

Distilled from the pinned upstream sources in `third_party/` (DeepGEMM sm90
GEMM and scheduler, FlashMLA sm90 decode, CUTLASS Hopper collectives) and from
the FlashAttention-3 paper. Distilled, not copied: each is written for one idea
at a shape that compiles here, and names the upstream file a reader should open
for the full production version.

| Template | Archetype | The decisions it carries |
|---|---|---|
| `10_persistent_ws_gemm.cu` | DeepGEMM-style GEMM | persistent grid, L2-aware raster width, cluster multicast, per-warp barrier arrivals, dynamic-smem opt-in |
| `11_fp8_two_level_accum.cu` | sm90 fp8 GEMM | wgmma partial + CUDA-core fp32 promotion, per-row/per-column dequant folded into the promotion, why BLOCK_K is 128 |
| `12_attention_online_softmax.cu` | FlashAttention-3 | P stays in registers (A and C fragments coincide), 2-shuffle row reduction, exp2 with folded log2(e), MN-major V |
| `13_mla_decode_split_kv.cu` | FlashMLA decode | split along keys, fp32 partials + LSE, no-split fast path, PDL chain into the combine kernel |
| `14_grouped_moe_gemm.cu` | fused MoE | one launch for all experts, 3-D weight map indexed by expert, contiguous vs masked layouts, skip-before-arm |

## Tier 3 — mixed-precision GEMM

Distilled from Marlin (IST-DASLab, Apache-2.0) as carried by vLLM/SGLang, from
humming (inclusionAI, Apache-2.0) whose fused prmt LUT is used directly for
e2m1 -> e4m3, and from the CUTLASS type traits that define each format's scale.
`quant_sm90.cuh` holds the unpack vocabulary.

| Template | Formats | The decisions it carries |
|---|---|---|
| `20_marlin_w4a16.cu` | INT4A16 | mma.sync over wgmma at small batch, cp.async over TMA for a pre-permuted weight blob, lop3 dequant, per-group scales |
| `21_fp4_block_scaled_gemm.cu` | NVFP4A16, MXFP4A16 | ue8m0 vs ue4m3 block scales, 32- vs 16-element blocks, NVFP4's second per-tensor level, bias correction folded into the scale |
| `22_w4a8_gemm.cu` | INT4A8, MXFP4A8 | scales leave the inner loop and meet on the accumulator, int8 exact accumulation, e2m1 -> e4m3 by a prmt LUT with the block scale built into the table, fp8 promotion per scale block |
| `23_offline_weight_repack.cu` | all of the above | the packer half: why Marlin's pack order is what it is, AWQ's order as its inverse, permute-into-a-layout vs normalize-then-JIT, dense sub-byte packing across word boundaries |

A quantized kernel and its packer are **one artifact with two halves**. Template
23 is not optional reading for 20-22: the dequant in each is cheap only because
the weights were permuted offline, and a kernel shipped without its matching
packer produces wrong numbers rather than a slowdown.

The fact that shapes all three: **sm90 has no sub-8-bit tensor core.**
`mma_sm90_gmma.hpp` contains zero e2m1 atoms, so every INT4 and FP4 kernel here
unpacks in registers before the MMA. Native block-scaled MMA is sm100 and
sm120 only — porting these to sm120 deletes the unpack rather than translating
it, and template 21's header says where.

## Tier 4 — memory-bound glue ops

Distilled from FlashInfer (Apache-2.0) `include/flashinfer/{norm,activation,
pos_enc}.cuh` and SGLang's per-token quantization kernels.
`elementwise_sm90.cuh` holds the vectorization and reduction vocabulary.

| Template | Op | The decisions it carries |
|---|---|---|
| `30_rmsnorm_residual.cu` | norm | fp32 accumulation, two traversals as the roofline, residual add fused in, `weight_bias` serving two model families |
| `31_swiglu_fp8_quant.cu` | swiglu + quantization | four traversals collapsed to one, the result held in registers across the amax reduction, amax/448 with the zero-row guard, satfinite conversion |
| `32_rope_layouts.cu` | rope | interleaved pairs are register neighbours (`v[j^1]`), half pairs cost a second load, and the layout is buyable offline |
| `33_softmax_rowwise.cu` | softmax | online vs naive is 2 traversals against 3, not correctness -- unlike attention's, where online is forced |

Two facts run through all four. They are bound by **traversals of the row**, so
the only real optimization is fusing to remove one. And they are short enough
that the launch ramp is a comparable term [launch.lat.dev.ramp], which is why
FlashInfer brackets every one of these kernels with PDL rather than only the
interesting ones.

## Tier 5 — the fusion endgame

Where launch cost stops being tunable and becomes structural. Four upstream
megakernels, one template each, plus the MoE align/finalize pass around the
grouped GEMM. The megakernel entries are whole machines, not sketches: each
builds, checks itself against a double-precision reference, times itself under
a CUDA graph, and carries its measurements in a STATUS block next to the
upstream's published numbers. The wiki entry
[`ext-mpk-megakernel`](../wiki/ext-mpk-megakernel.md) holds the rules they
have in common.

| Template | Subject | The decisions it carries |
|---|---|---|
| `40_megakernel_interpreter.cu` | megakernel VM, reduced | the instruction ring, pages tied to the ring stage, a built-in profiler -- and the ledger of why it loses to three launches: it waits on the dependency before loading anything |
| `42_hazy_llama_megakernel.cu` | HazyResearch low-latency Llama | static per-SM instruction streams from a host planner (round robin or cost-weighted list scheduling), pages released in an op-declared order across instruction boundaries, weights prefetched under the dependency wait, one global counter per (layer, op, slice), split attention with an LSE reduction op; a whole Llama-1B decode step |
| `43_mpk_task_graph_runtime.cu` | Mirage MPK runtime | tasks and events with trigger counts, worker CTAs with per-worker queues, scheduler warps with event queues, cumulative counters that carry the kernel across decode steps, pre-launch (AOT) against launch-on-fire (JIT), event granularity as a compiler decision, cross-task pre-loading; the same decode step as 42 |
| `44_megamoe_sm90.cu` | DeepGEMM Mega MoE, sm90 port | dispatch into an expert-major pool, a two-stage scheduler whose L1 warmup waves keep the dependent L2 stage deadlock-free, a ring of pool blocks with four use-scaled counters, swapped-AB wgmma with tokens as N, granularity-8 gate/up interleave so SwiGLU is a register pair, the top-k weight folded into the L1 epilogue |
| `45_flag_barrier_megakernel.cu` | flag-barrier megakernel | the minimal form: one launch, phases split by a release/acquire flag the last CTA resets, redundant norms in place of a hop, `L1::no_allocate` and `L2::cache_hint` weight loads with the sm90 spellings that survive ptxas, and what a phase boundary costs against a launch |
| `41_moe_align_finalize.cu` | MoE around the grouped GEMM | why the align pass pads to the GEMM's block size, contention bounded by expert count not token count, the inverse permutation built once, the shared expert riding the gather |

Templates 42 and 43 run one workload so the two schedulers can be compared;
on this machine the static schedule reaches a higher fraction of the
bandwidth floor than the dynamic one, and both sit in the range their
upstream projects report. The single largest lever in every one of them is
where the dependency wait sits relative to the weight stream -- read the
wiki entry's first rule before fusing anything.

Template 44 is a port: DeepGEMM ships Mega MoE for sm100 only, and the sm90
build SGLang uses has no public source, so the sm90 file here reproduces the
sm100 design's scheduler and counters over wgmma, single-rank and bf16.

To run one on this cluster (the login node has no GPU):

```bash
sbatch --export=ALL,TEMPLATE=42_hazy_llama_megakernel.cu,RUN_ARGS="partials=8" sbatch/kernel_template.sh
```

`sm90_common.cuh` holds the raw primitives every template shares.

## What these are, and what they are not

They are **toolkit-only skeletons**: 01, 02 and 04 need nothing but CUDA; the
rest need CuTe because the wgmma matrix descriptor and operand fragments are
CuTe's to build. They carry no project types, so they are portable experience
rather than a copy of this repo's kernels.

The archetypes are **simplified on purpose**. They fix a shape, drop the tail
and predication handling, and leave out autotuning, so the structure stays
readable. A production kernel of the same family is two to five times longer,
and the upstream file named in each header is where that lives.

They are **not** the code to type here. Production kernels compose
`tile/sm90/*.cuh`, whose README owns the selection contract and the caller
invariants; the templates show what those primitives are doing and what the
hardware requires, which is what a reviewer and an agent both need when the
library does not cover a case.

They are **structurally verified only**. The checker proves each declared
instruction survives codegen — it does not prove any value is correct.
Numerical authority is a parity harness, per `../parity.md`.

## Checking them

```bash
source /usr/share/Modules/init/bash && module load cuda/13.0 gcc/13.3
export CXX="$(command -v g++)"          # nvcc's default host compiler is GCC 8, too old
python3 .claude/skills/kernel-design/scripts/check_templates.py
```

Login node, no GPU. Each template declares what must appear in its PTX:

```
// CHECK-ARCH: sm_90a               target, default sm_90a
// CHECK-INCLUDE: third_party/x     repo-relative -I, repeatable
// CHECK-PTX: wgmma\.mma_async      regex that must match
// CHECK-PTX-COUNT: 4 wgmma\.       regex that must match at least N times
```

A template with no assertion fails: compiling proves nothing on its own, since
a dead-code-eliminated mainloop still exits zero.

## Rules for adding one

- **One mechanism or one archetype per template**, named in the first line. A
  ladder template stacks on the previous one; an archetype states the decisions
  that distinguish its family and reuses the ladder for everything else.
- **Assert the instructions that are the point.** If the template exists to show
  a wgmma batch, assert the wgmma, the fence, the commit and the wait.
- **Declare every PDL site.** A template using PDL carries
  `// PDL-WAIT: <the read it precedes> -- DERIVED ...` and
  `// PDL-TRIGGER: <position> -- SWEPT: <result, or "not yet measured">`; the
  checker fails one that does not. The wait follows from data dependencies, the
  trigger only from measurement, and the annotation is what tells a later reader
  which of the two a given position is. See `../wiki/pdl-placement.md`.
- **A number appears only as a rule**, cited by hardware-unit-test tag
  (`[wgmma.stages.wg.knee]`), never as a measurement from a run.
- **Comments explain the decision, not the line.** ASCII only, `->` and `--`.
  The header comment carries the rules; the body carries only what is not
  recoverable from the code — which barrier orders what, why a wait is not
  zero, what breaks if a fence moves.
- **No project coupling**: no `flash_vla::` types, no paths into `src/`, and no
  numbers from a specific job. Name the owning primitive in `tile/sm90/` so a
  reader can cross over.
- Add the template to the table above and to the wiki entry it makes concrete;
  run the checker before committing.
