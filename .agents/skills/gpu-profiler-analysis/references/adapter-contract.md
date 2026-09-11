# Local adapter contract

An adapter produces a JSON `CapturePlan` for the generic runner:

```json
{
  "command": ["python", "profile_workload.py"],
  "env": {"PROFILE_MODE": "timeline"},
  "expected_artifacts": ["trace.json.gz"],
  "workload": {
    "name": "example",
    "stage": "decoder",
    "shape_profile": {"m": 50, "n": 2048}
  },
  "capture": {
    "warmup_steps": 3,
    "active_steps": 1,
    "cuda_graph": true,
    "activities": ["CPU", "CUDA"]
  }
}
```

The command is an argv list, not a shell string. The adapter must create the
trace or native profiler artifact; it must not hide compilation, warmup, or
allocation failures. The runner injects `GPU_PROFILE_OUTPUT_DIR` for the Torch
backend and records the command, environment-independent metadata, logs, exit
code, and artifact hashes.

For `flash-vla`, adapters should expose the real target shape and stage rather
than inventing a synthetic workload. Keep model-specific imports in the adapter,
not in the unified analyzer.
