---
name: benchmark-kernel
description: Measure and compare a kernel or fused call site under representative shapes, buffers and cache conditions. Use for local latency/throughput evidence, not profiler diagnosis or model-level performance claims.
---

# Kernel timing

Own local timing. [ncu-report](../ncu-report/SKILL.md) explains hardware counters;
`python -m benchmarks latency` measures the deployed model.

1. Reuse the selected call site's actual inputs and plan. For a standalone kernel,
   use the existing `flash_vla.bench.bench_gpu_time` helper.
2. Keep shapes, dtype, layout, cache policy and timer the same on both sides.
   Warm up JIT/allocation work before timing; use the graph's capture stream.
3. Select a timer matching the question. The Target CLI defaults to amortized
   in-graph timing; CUPTI isolates a single launch, while events include launch
   overhead. Compare complete fused chains at their call-site boundary.
4. Report the command, conditions, raw samples and median; include throughput
   when its FLOP/byte count is meaningful. Uncertain gains need a focused repeat.

## Example

From the project root, append the active workload's asset options:

```bash
python -m benchmarks kernels --target h100/pi05 --plan shipped \
  --segment action_expert --site action_expert_attention --timer cupti
```

An omitted checkpoint may select synthetic inputs/weights; use the workload
actually being optimized. See [timing recipes](references/timing.md) for the
standalone Python API, cache handling and timer limitations. Resource allocation
and machine troubleshooting belong to user-local environment guidance.
