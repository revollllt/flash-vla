# Latency benchmarks

Measure how long deployed inference or a kernel/fusion chain takes. Commands run
from the project root with the active workload's environment and asset options.

```bash
python -m benchmarks latency --target h100/pi0 --plan shipped --out artifacts/current.json
python -m benchmarks kernels --target h100/pi05 --plan shipped \
  --segment action_expert --site action_expert_attention --timer cudagraph --csv artifacts/kernels.csv
```

`latency` measures the deployed forward, including input staging, host work and
final synchronization, after loading and capture. Each version runs in a fresh
process using its first capture: five warmups, 100 samples, no soak, with median
and raw samples retained. Compare versions under matching workload and device
conditions; use a focused repeat when the difference is uncertain.

`kernels` times a call site or required fusion chain with the model's actual
buffers and layouts. Select graph, event or CUPTI timing for the relevant regime;
local kernel gains still need deployed end-to-end confirmation.

Detailed traces, NCU analysis and hardware-limit estimates belong to
[profiling tools](../tools/README.md). `--breakdown` adds segment timings and
`--attribution` explicitly enables diagnostic telemetry; ordinary latency imports
and measurements do not activate profiling. Accuracy belongs to [eval](../eval/README.md).
