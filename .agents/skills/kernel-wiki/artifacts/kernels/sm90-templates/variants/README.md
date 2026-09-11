# sm90 Templates — compilable skeletons for the choreography

This directory is a derived artifact bundle of the `kernel-wiki` skill
(`artifacts/kernels/sm90-templates/variants`; `PROVENANCE.yaml` names the
sources each template distils and pins every file by sha256). The wiki's
pages excerpt these files by name and cite their `STATUS` blocks; the
workflow that uses them is the `kernel-design` skill.

Five tiers. **01-04** are the mechanism ladder: each adds one primitive to the
previous one, and together they are a whole sm90 pipeline. **10-14** are kernel
archetypes: the shape a real high-performance kernel of that family has.
**20-23** are the mixed-precision family, where the quantization format drives
the kernel. **30-33** are the memory-bound glue ops between the GEMMs. **40-45**
are the fusion endgame. Start at the template nearest the task, drop to the
ladder when a mechanism in it is unfamiliar.

The decisions a template carries are in its own header, as the reasons for
that file; the rules they instantiate are `kernel-wiki` pages, which cite the
template by file for their code excerpts and STATUS numbers. The inverse map,
template to the pages that cite it, is generated: `queries/by-template.md`
at the wiki root.

## Catalog

| Template | Tier | Subject | Grade |
|---|---|---|---|
| `01_tma_mbarrier_ring.cu` | ladder | tensor map, ring of frames, full/empty barriers, derived phase parity | structural |
| `02_warp_specialization.cu` | ladder | producer/math warpgroup split, `setmaxnreg` handoff, named barriers | structural |
| `03_wgmma_mainloop.cu` | ladder | swizzle agreement, CuTe descriptors, fence/arrive/commit/wait batch | structural |
| `04_epilogue_persistent.cu` | ladder | cross-proxy publish, bulk-store groups, persistent task striding | structural |
| `10_persistent_ws_gemm.cu` | archetype | DeepGEMM-style persistent warp-specialized GEMM with cluster multicast | structural |
| `11_fp8_two_level_accum.cu` | archetype | sm90 fp8 GEMM: wgmma partial plus CUDA-core fp32 promotion | structural |
| `12_attention_online_softmax.cu` | archetype | FlashAttention-3-style warp-specialized attention, online softmax | structural |
| `13_mla_decode_split_kv.cu` | archetype | FlashMLA-style split-KV decode with a separate combine on a PDL chain | structural |
| `14_grouped_moe_gemm.cu` | archetype | fused MoE grouped GEMM: one launch for all experts, 3-D weight map | structural |
| `20_marlin_w4a16.cu` | mixed precision | INT4A16: `mma.sync` over a pre-permuted `cp.async` blob, `lop3` dequant | structural |
| `21_fp4_block_scaled_gemm.cu` | mixed precision | NVFP4A16, MXFP4A16: ue8m0 vs ue4m3 block scales | structural |
| `22_w4a8_gemm.cu` | mixed precision | INT4A8, MXFP4A8: scales meet on the accumulator, e2m1 to e4m3 by `prmt` LUT | structural |
| `23_offline_weight_repack.cu` | mixed precision | the packer half: Marlin's pack order, AWQ's as its inverse | structural |
| `30_rmsnorm_residual.cu` | glue | RMSNorm with fused residual, PDL trigger as a swept knob | structural |
| `31_swiglu_fp8_quant.cu` | glue | SwiGLU with per-token fp8 quantization in one traversal | structural |
| `32_rope_layouts.cu` | glue | rope: interleaved vs half pairs, the layout as an offline purchase | structural |
| `33_softmax_rowwise.cu` | glue | row-wise softmax: online vs naive as a traversal count | structural |
| `40_megakernel_interpreter.cu` | endgame | a reduced megakernel VM, and the ledger of why it loses to three launches | reference |
| `41_moe_align_finalize.cu` | endgame | the MoE align and finalize passes around the grouped GEMM | structural |
| `42_hazy_llama_megakernel.cu` | endgame | HazyResearch low-latency Llama megakernel: planner, pages, per-op counters | reference |
| `43_mpk_task_graph_runtime.cu` | endgame | Mirage MPK runtime: tasks, events, workers, scheduler warps, across decode steps | reference |
| `44_megamoe_sm90.cu` | endgame | DeepGEMM Mega MoE, sm90 port: two dependent stages over one pool | reference |
| `45_flag_barrier_megakernel.cu` | endgame | flag-barrier megakernel with cache-policy loads | reference |
| `sm90_common.cuh` | shared | the raw sm90 primitives every template uses | — |
| `quant_sm90.cuh` | shared | the register-side unpack vocabulary (`technique-register-unpack-sub-byte`) | — |
| `elementwise_sm90.cuh` | shared | the vectorization and reduction vocabulary (`technique-row-traversal-fusion`) | — |

Templates 42 and 43 run one workload so the two schedulers can be compared;
template 44 is a port, since DeepGEMM ships Mega MoE for sm100 only. The
comparison and the rules the five endgame templates share are
`kernel-megakernel-forms` in `kernel-wiki`.

To run a reference template, use the nvcc build command in its header, then
execute the resulting binary on the intended GPU. For example, template 42
accepts `partials=8` as a runtime argument.

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

## Two grades

Every template declares which of two things it is, as `// CHECK-GRADE:`, and
the checker holds it to that claim.

| Grade | What it is | What it may claim |
|---|---|---|
| `structural` | a skeleton: it compiles, and each declared instruction is proven to survive codegen | the mechanism and the decisions behind it — nothing measured |
| `reference` | a whole machine: it runs, checks itself against a double-precision reference with the device's rounding points, and times itself under a CUDA graph | its own numbers, in a `STATUS` block naming the machine, the toolchain and the build line that reproduces them |

The split is not a quality ranking. A ladder template exists to show one
mechanism, and a harness wrapped around it would bury the thing it is there to
show. The grade is a statement about evidence, and it is enforced in both
directions: a structural template carrying a `STATUS` block fails, because it
would be quoting numbers nothing in the file can re-take, and a reference
template without one fails, because it claims to have been run and shows
nothing for it.

Numerical authority for a structural template is a parity harness, per
`../parity.md`. For a reference template it is the in-file check, which needs a
GPU.

## Checking them

```bash
python3 .agents/skills/kernel-wiki/scripts/check_templates.py
```

Use nvcc and a compatible host compiler; no GPU is needed to compile. Each template declares its grade and what must appear in
its PTX:

```
// CHECK-GRADE: structural          structural | reference, required
// CHECK-ARCH: sm_90a               target, default sm_90a
// CHECK-INCLUDE: third_party/x     repo-relative -I, repeatable
// CHECK-PTX: wgmma\.mma_async      regex that must match
// CHECK-PTX-COUNT: 4 wgmma\.       regex that must match at least N times
```

A template with no assertion fails: compiling proves nothing on its own, since
a dead-code-eliminated mainloop still exits zero. A template with no grade
fails too — a reader has no way to tell a skeleton from a machine by looking.

The wiki's validator covers the other half:

```bash
.venv/bin/python .agents/skills/kernel-wiki/scripts/validate.py
```

Every machine-constant tag a template header cites in brackets must resolve
through `hardware-unit-test`, every page id it cites must name a live wiki
page, every excerpt a page takes from a template must appear verbatim in the
named file, and every file's sha256 must match `PROVENANCE.yaml`. After
editing a template, re-pin the bundle:

```bash
.venv/bin/python .agents/skills/kernel-wiki/scripts/pin_artifacts.py artifacts/kernels/sm90-templates/variants
```

## Rules for adding one

- **One mechanism or one archetype per template**, named in the first line. A
  ladder template stacks on the previous one; an archetype states the decisions
  that distinguish its family and reuses the ladder for everything else.
- **The header carries this file's reasons, not the wiki's rules.** What the
  file is, the machine as upstream builds it, what was changed and why, the
  grade and its evidence. A rule that holds beyond this file is a `kernel-wiki`
  page; cite it by id in brackets and let the page cite the file back.
- **Declare the grade, and earn it.** `structural` says so in the header and
  reports no measurement. `reference` carries a `main`, a reference check the
  reader can run, a build line, and a `STATUS` block stating the machine, the
  toolchain, whether clocks were pinned, and the floor the numbers are a
  fraction of. Reporting a loss is a result; a reference template that only
  ever wins teaches the wrong thing.
- **Assert the instructions that are the point.** If the template exists to show
  a wgmma batch, assert the wgmma, the fence, the commit and the wait.
- **Declare every PDL site.** A template using PDL carries
  `// PDL-WAIT: <the read it precedes> -- DERIVED ...` and
  `// PDL-TRIGGER: <position> -- SWEPT: <result, or "not yet measured">`; the
  checker fails one that does not. The wait follows from data dependencies, the
  trigger only from measurement (`technique-pdl-placement`).
- **A number appears only as a rule**, cited by hardware-unit-test tag
  (`[wgmma.stages.wg.knee]`), never as a measurement from a run — except in a
  `STATUS` block.
- **Comments explain the decision, not the line.** ASCII only, `->` and `--`.
- **No project coupling**: no `flash_vla::` types, no paths into `src/`, and no
  numbers from a specific job. Name the owning primitive in `tile/sm90/` so a
  reader can cross over.
- Add the template to the catalog above; give the wiki page it makes concrete
  an excerpt named by file, so `queries/by-template.md` lists it; re-pin the
  bundle; run `check_templates.py` and `validate.py` before committing.
