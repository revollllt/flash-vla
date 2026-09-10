# CUDA software-range tracing

This diagnostic example observes scalar and synchronous WMMA warp roles on one
GPU. Ranges describe software scopes, not hardware-active cycles. Unobserved
time is unknown, not idle. The example is not a production performance test.

From the repository root:

```bash
python -m unittest discover -s eval/tests -p 'test_kernel_trace.py' -v
python -m benchmarks.kernel_trace.export_perfetto raw.json trace.perfetto.json --summary summary.json
```

On a configured CUDA machine the standalone build is:

```bash
nvcc -O3 -std=c++17 -lineinfo -arch=sm_90 -Isrc lab/examples/kernel_trace/profile_example.cu -o /tmp/profile_example
/tmp/profile_example /tmp/trace.raw.json 64 128
```

The raw format accepts one GPU-local clock domain in integer nanoseconds.
Chrome JSON timestamps are microseconds relative to an integer origin.
CTA/warp/launch/replay identity owns a track; SM ID does not identify a writer.
Cross-device clocks cannot be merged. Endpoint SM changes exclude a range from
per-SM statistics. Dropped records fail export by default; `--allow-partial`
permits visualization but omits duration and overlap summaries.

Keep capture artifacts under an ignored experiment directory. Use trace-on/off
parity to check instrumentation correctness; raw GPU traces and synthetic CPU
fixtures are different evidence. No GPU timing is inferred from the CPU tests.

Timer resolution and statistical perturbation remain uncalibrated. Perfetto UI
import, focused sampling, asynchronous TMA/WGMMA completion pairing, and CUDA
Graph replay isolation are not validated by this synchronous example. Use an
uninstrumented workload for performance conclusions.
