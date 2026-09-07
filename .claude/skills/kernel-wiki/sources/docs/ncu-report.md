---
id: doc-ncu-report
title: "ncu-report: Nsight Compute on sm90, the metric vocabulary and the diagnosis playbook"
url: ../../../ncu-report/SKILL.md
source_category: official-doc
architectures: [sm90]
tags: [wgmma, tma, mbarrier]
retrieved_at: 2026-09-07
---

# ncu-report

The sibling skill captures a kernel with Nsight Compute 2025.4.1 on an
ncu-capable node and reads the report into a named bottleneck. Its
`references/08-sm90-metric-names.md` is the verified sm90 metric and
stall-reason vocabulary (aggregate ratios name `gmma`, the per-PC sampler
names `warpgroup_arrive`); its `references/06-diagnosis-playbook.md` carries
patterns A through V, of which O through V are the Hopper-specific ones that
say when to overrule an NCU rule on a persistent warp-specialized kernel.

The wiki's `symptoms:` vocabulary is spelled after this skill's stall reasons
and rule names, so a report's finding and the by-problem index share one
language.
