# Chrome trace contract

The portable timeline artifact is a Chrome trace JSON file or a gzip-compressed
Chrome trace JSON file. Accept either a top-level object with `traceEvents` or a
top-level event list. Event timestamps and durations use microseconds.

The analyzer treats events with `ph: "X"` and a positive `dur` as complete
events. GPU kernels are identified from a `Kernel`/CUDA category when possible;
the classifier is intentionally conservative and reports counts so a new trace
producer can be checked.

Optional fields used by the analyzer:

```json
{
  "name": "kernel name",
  "cat": "Kernel",
  "ph": "X",
  "ts": 123.0,
  "dur": 4.5,
  "pid": 1,
  "tid": 7,
  "args": {
    "stream": 3,
    "stage": "decoder",
    "python_scope": "pipeline.decoder",
    "source_file": "pipeline.py",
    "source_line": 160
  }
}
```

`stage`, source fields, and stream identifiers are optional. Missing source
fields mean `mapping_status` is `partial` or `unavailable`; kernel names are
never treated as source locations.

The sidecar `manifest.json` records workload, tool, host, git, capture, and
artifact metadata. Paths inside the manifest are relative to the run directory.
