---
name: ncu-report
description: Diagnose a selected CUDA kernel from Nsight Compute counters, or choose a focused capture that answers a kernel-level question. Use after timeline analysis has identified the kernel; local timing belongs to benchmark-kernel.
---

# Kernel counter diagnosis

Own the question, NCU capture settings and interpretation. Whole-model timelines
belong to [gpu-profiler-analysis](../gpu-profiler-analysis/SKILL.md); its runner is
also available as a capture executor. Neither profiler replaces benchmark timing.

1. Name the kernel, shape, execution mode and question. Reuse an applicable report.
2. If needed, collect only the matching launch and sections/metrics that answer
   the question. Choose replay, cache and clock controls for the workload; check
   the installed NCU's supported options. A full-set pass is optional.
3. Read launch geometry, compute/memory activity and relevant stalls. Use source
   lines only when the report has line information. Treat rule speedups as leads,
   and account for persistent-kernel designs before applying occupancy advice.
4. Return the observed bottleneck, supporting metrics, uncertainty and one useful
   next experiment. Confirm speed with [benchmark-kernel](../benchmark-kernel/SKILL.md).

## Example

Analyze an existing report from the project root; no GPU is needed:

```bash
python .claude/skills/ncu-report/scripts/report_query.py summary artifacts/profile/kernel.ncu-rep
python .claude/skills/ncu-report/scripts/report_query.py hotspots artifacts/profile/kernel.ncu-rep --top 10
```

Use `NCU_PYTHON_DIR` for a matching installation's `extras/python` directory when
it is outside standard locations. See [collection](references/03-collection.md),
[harnesses](references/02-harness-guide.md),
[analysis dimensions](references/05-analysis-dimensions.md) and
[diagnosis patterns](references/06-diagnosis-playbook.md) for the relevant detail.
Environment permissions and scheduler commands belong in user-local guidance.
