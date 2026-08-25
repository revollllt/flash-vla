# Torch backend

The Torch backend is the portable timeline path. A profile-aware workload should
create the profiler context itself and call `export_chrome_trace` or
`tensorboard_trace_handler(..., use_gzip=True)` after the active replay.

Recommended timeline settings:

```python
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
record_shapes = False
with_stack = False
profile_memory = False
```

Enable `record_shapes` or `with_stack` only for a mapping capture and record the
choice in `manifest.json`. Use a warmup replay before the profiled replay so JIT
compilation and scratch-pool growth do not contaminate the timeline.

The generic local runner does not inject `torch.profiler` into arbitrary code.
The workload adapter owns graph construction, warmup, capture, and trace export;
the runner owns environment, logs, artifact discovery, and metadata.
