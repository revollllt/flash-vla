---
id: doc-kernel-design-templates
title: "sm90 templates: compilable, graded skeletons and reference machines (artifacts/kernels/sm90-templates)"
url: ../../artifacts/kernels/sm90-templates/variants/README.md
source_category: official-doc
architectures: [sm90, sm90a]
tags: [tma, mbarrier, wgmma, cluster, pdl, ldmatrix, cp-async-bulk, setmaxnreg, mma-sync]
retrieved_at: 2026-09-07
---

# sm90 templates

Twenty-three templates in five tiers, a derived artifact bundle of this
wiki at `artifacts/kernels/sm90-templates/variants/` (its `PROVENANCE.yaml`
names the sources each distils and pins every file by sha256): the mechanism ladder
(01-04), kernel archetypes (10-14), the mixed-precision family (20-23), the
memory-bound glue ops (30-33) and the fusion endgame (40-45), over a shared
primitives header `sm90_common.cuh`. Every template declares a grade:
`structural` compiles and has each declared PTX instruction asserted to
survive codegen; `reference` also runs, checks itself against a
double-precision reference and times itself under a CUDA graph, reporting the
machine, toolchain and build line in a `STATUS` block.

`python3 .claude/skills/kernel-wiki/scripts/check_templates.py` enforces
the grade and the PTX assertions on the login node; `validate.py` checks the
digests, the header citations and that every page excerpt appears verbatim in
the named file; `pin_artifacts.py` re-pins after an edit.

## How the wiki cites this source

A technique or hardware page's code snippet is a contiguous excerpt of one
template or of `sm90_common.cuh`, named by file, and the validator fails an
excerpt that is not found verbatim in that file. A performance claim cites a
`reference` template's `STATUS` block by file as its locator; those blocks
are the only numbers a template may carry, and they name H100 SXM5, the CUDA
version, unpinned clocks and the shape.

Reference grade at the time of writing: `40_megakernel_interpreter.cu`,
`42_hazy_llama_megakernel.cu`, `43_mpk_task_graph_runtime.cu`,
`44_megamoe_sm90.cu`, `45_flag_barrier_megakernel.cu`. The other eighteen
are structural: their header rules are design statements, not measurements.
