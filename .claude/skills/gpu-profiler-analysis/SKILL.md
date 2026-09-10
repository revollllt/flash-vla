---
name: gpu-profiler-analysis
description: Capture or analyze model/module timelines with Torch Profiler or Nsight Systems to locate costly stages, launch gaps and synchronization. Kernel-counter diagnosis belongs to ncu-report.
---

# Model and module profiling

Own the timeline: where time goes across the real forward. Use
[ncu-report](../ncu-report/SKILL.md) after selecting a kernel, and the latency
harness for uninstrumented end-to-end performance.

1. Reuse an applicable trace. Otherwise capture the full forward with the model's
   benchmark inputs and execution configuration.
2. Inspect the GPU timeline for costly modules, host/launch gaps and dependencies.
   CPU annotations show submission scopes, not GPU duration.
3. Profile only the selected module in more detail. Without correlation or stack
   evidence, state that source mapping is unavailable.
4. Return the dominant cost, supporting trace locations and the next focused
   question. Profiler time is diagnostic; final timing runs separately.

## Example

From the project root, with the active model environment and assets configured:

```bash
python -m tools.profiling.model --target h100/lingbot_vla --plan shipped --seed 42 \
  --overview --trace-dir artifacts/profile/overview
python .claude/skills/gpu-profiler-analysis/scripts/analyze_trace.py \
  --input artifacts/profile/overview/overview.json --output-dir artifacts/profile/summary
```

For a generic command, use [run_local_profile.py](scripts/run_local_profile.py).
Read [capture modes](references/capture-modes.md),
[Nsight Systems](references/nsys-backend.md) or
[trace schema](references/trace-schema.md) only as needed. The shared runner can
execute NCU, but its capture settings and interpretation are owned by ncu-report.
