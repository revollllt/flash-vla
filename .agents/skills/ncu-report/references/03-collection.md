# Collect the counters needed for the question

Run in an environment with the intended GPU, CUDA workload and Nsight Compute
available. Machine access and counter permissions come from user-local setup.
Use the installed `ncu --help`, `--list-sections` and `--query-metrics` to select
supported options; names vary by NCU version and GPU.

## Select a launch

From the project root, for an existing standalone harness:

```bash
mkdir -p artifacts/profile/attention
ncu --set basic --kernel-name 'regex:attention' --launch-count 1 \
  --export artifacts/profile/attention/kernel ./harness
```

Replace the regex with the kernel's actual demangled name. Use `--launch-skip`
when the driver includes warmup launches, and record which launch was captured.
The same NCU prefix can wrap an existing project Python driver; see
[harness choices](02-harness-guide.md). For graph workloads, use the installed
NCU's graph/replay mode appropriate to the actual dependencies.

## Add detail when needed

For a memory question, request `MemoryWorkloadAnalysis`; for occupancy or
scheduler questions, select their corresponding sections. Source/PC analysis
needs its sampling section and `-lineinfo` for CUDA source attribution.
`--set full` is an option when a broad counter collection is justified; it is
not a prerequisite for diagnosis and can require many replay passes.

Choose replay and cache controls deliberately: isolated cold-cache replay may
not represent a kernel inside a pipeline. Keep those settings aligned when
comparing reports. If clock control is unavailable, `--clock-control=none` uses
the available device clocks; record that setting and measure actual variability.
There is no universal percentage below which a change is noise.

## Read the result

```bash
python .agents/skills/ncu-report/scripts/report_query.py summary artifacts/profile/attention/kernel.ncu-rep
ncu --import artifacts/profile/attention/kernel.ncu-rep --page details
```

Use the writer's NCU version, or a compatible parser. Keep failed captures
visible and correct their cause before interpreting output. Profiler durations
support diagnosis; use an uninstrumented benchmark for latency claims.
