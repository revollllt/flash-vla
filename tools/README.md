# Development tools

Profile the whole forward first, then inspect only its expensive module/kernel.
Use the same workload and execution settings as the latency benchmark.

```bash
python -m tools.profiling.model --target h100/pi0 --plan shipped \
  --overview --trace-dir artifacts/profile/overview
python -m tools.profiling.model --target h100/pi0 --plan shipped \
  --segment action_expert --trace-dir artifacts/profile/detail
python -m tools.profiling.floor --target h100/pi0 --plan shipped
```

`profiling/model.py` captures and attributes model timelines. `kernel_trace/`
queries and exports kernel instrumentation; `floor.py` compares costs with
hardware peaks and measured primitive limits. NCU capture/analysis examples live
in the [ncu-report skill](../.claude/skills/ncu-report/SKILL.md).
Profiled durations explain bottlenecks; measure final speed separately with
[benchmarks](../benchmarks/README.md).

Two occasional utilities remain separate from evaluation:

```bash
python -m tools.check_checkpoint --help
python -m tools.calibrate --reports artifacts/correctness.json
```

Checkpoint inspection does not prove numerical accuracy. Calibration reports
existing rounding dispersion; it does not silently change the evaluation limits.
