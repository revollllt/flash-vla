---
name: kernel-wiki
description: Find GPU kernel techniques, architecture-specific pitfalls and reusable implementation examples in the project knowledge base. Use for design research, not live profiling, timing or workflow execution.
allowed-tools: "Bash Read Grep Glob"
---

# Kernel knowledge lookup

Own reference retrieval. [kernel-design](../kernel-design/SKILL.md) decides and
implements the candidate; profiling and timing have separate owners.

1. Search by the observed symptom, technique or GPU architecture.
2. Read only the relevant pages and linked code examples. Check the architecture
   and confidence labels before transferring a technique.
3. Follow the cited source for a claim. Use
   [hardware-unit-test](../hardware-unit-test/SKILL.md) for current machine values
   and [benchmark-kernel](../benchmark-kernel/SKILL.md) to verify a candidate.
4. Return the useful pattern, its applicability limits and a concrete code/page
   location. Existing measured results are evidence for their stated conditions.

## Example

From the project root:

```bash
python .claude/skills/kernel-wiki/scripts/query.py --symptom fusion-regression --compact
python .claude/skills/kernel-wiki/scripts/get_page.py technique-release-on-retirement
```

For a broad question read [the topic map](references/primer.md); for query options
read [examples](references/examples.md). The [template index](queries/by-template.md)
locates CUDA skeletons. Validate or compile affected pages/templates when editing
this knowledge base; ordinary lookup does not run corpus-wide checks.

For concrete Hopper implementations, read [DeepGEMM](references/deepgemm-sm90.md)
for producer/consumer GEMM or [FlashMLA](references/flashmla-sm90.md) for cooperating
warp groups and split-KV decode. These are pinned-source readings, not current project requirements.
