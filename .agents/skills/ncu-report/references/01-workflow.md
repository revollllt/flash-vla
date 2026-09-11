# Focused NCU workflow

1. Start from the kernel selected by model/module profiling. State a question:
   bandwidth, tensor activity, pipeline stalls, launch geometry or another
   concrete uncertainty. Reuse a report when its conditions answer that question.
2. Choose the [existing driver or a small harness](02-harness-guide.md). Keep the
   actual shapes, layout and execution path; use initialized representative inputs.
   Run it without instrumentation first if that path has not yet been exercised.
3. [Capture](03-collection.md) the matching launch and relevant counters. Begin
   with a small set; collect additional sections only when a finding needs them.
4. Read [the relevant analysis dimensions](05-analysis-dimensions.md) and
   [diagnosis patterns](06-diagnosis-playbook.md). Distinguish observations from
   hypotheses, missing counters and profiler perturbation.
5. Return the supporting metrics and one useful next experiment. Verify a
   speedup with [benchmark-kernel](../../benchmark-kernel/SKILL.md), outside NCU.

Example: low tensor activity plus long producer waits suggests checking TMA
issue/transfer activity before changing the matrix tile. If those counters are
absent, capture just the needed section rather than repeating every metric.

A short result is sufficient; the [report outline](07-report-template.md) is
optional for more involved investigations. No new approval or publication step
is implied by profiling.
