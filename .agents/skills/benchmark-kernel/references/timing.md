# Timing recipes

Commands run from the project root in the active workload environment.

## Choose the measurement boundary

| Path | Measurement | Relevant limit |
|---|---|---|
| Target CLI `--timer cudagraph` (default) | Recorded call-site invocations captured in one graph, replay time divided by invocation count | Represents amortized execution; preserve real buffer/layout and dependency groups |
| Target CLI `--timer cupti` | First recorded invocation, cold L2, GPU activity timestamps | Requires CUPTI; unavailable CUPTI falls back to events with a warning |
| Target CLI `--timer events` | Events around the first invocation, cold L2 | Can include host launch gaps; not interchangeable with CUPTI |
| `bench_gpu_time()` | Standalone callable, CUPTI by default | Pass tensor arguments explicitly so the helper can implement its cache policy |

Choose the same timer and cache policy on both sides. A local speedup does not
establish a model speedup; integration and deployment timing belong to the
[model workflow](../../../../docs/optimization.md).

## Target call site

Use the actual checkpoint/asset options of the workload. The default plan is
`shipped`; select a candidate plan explicitly.

```bash
python -m benchmarks kernels --target h100/pi05 --plan shipped \
  --segment action_expert --site action_expert_attention --timer cudagraph
```

The runner records call-site arguments from the engine. Dependent producer and
consumer calls declared as an atomic group form one benchmark case; timing a
consumer alone would not represent a valid invocation. FLOP/byte rates use the
Target's derived costs and are meaningful only for those stated counts.

## Standalone kernel or fusion

```python
import torch
from flash_vla.bench import bench_gpu_time, KernelResult

# Replace matmul with the candidate wrapper, keeping its actual shape and dtype.
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
out = torch.empty_like(a)

def kernel(a, b, out):
    torch.matmul(a, b, out=out)

samples = bench_gpu_time(kernel, input_args=(a, b, out),
                         enable_cupti=False, use_cuda_graph=True,
                         num_iters_within_graph=10)
result = KernelResult("candidate", samples)
print(result.perf_line())
# Retain samples alongside the configuration for comparison.
```

- `enable_cupti=True` selects CUPTI; `enable_cupti=False, use_cuda_graph=False`
  selects events. The helper's precedence differs from the Target CLI default.
- `dry_run_iters` and `repeat_iters` select fixed counts; otherwise the helper
  sizes warmup/repeats from `dry_run_time_ms` and `repeat_time_ms`.
- `cold_l2_cache=True` is the helper default. Graph mode rotates buffers from
  `input_args`/`input_kwargs`; hiding tensors in a closure prevents that rotation.
  CUPTI/event modes flush between iterations. Use `cold_l2_cache=False` only
  when the intended cache-resident workload calls for it.
- If the callable launches several kernels, CUPTI reports their enclosing GPU
  time span, including gaps. Label it as a fused chain/call-site span, not one
  kernel or a sum of kernel durations.

Initialize inputs and warm up compilation/allocations. For each new comparison
capture a fresh graph and replay it on its capture stream. Use independent
processes for model latency measurements, as specified in the model workflow.

Report raw samples and median with shape, dtype, execution/cache conditions and
timer. When an apparent gain is near observed variability, repeat the relevant
comparison with controlled resource contention before assigning a cause.
