---
name: gpu-profiler-analysis
description: Analyze GPU profiling artifacts and run local Torch Profiler, Nsight Systems, or Nsight Compute captures. Use for offline Chrome trace/JSON.GZ analysis, Perfetto-ready reports, CUDA Graph timeline triage, or targeted H100 kernel diagnosis; do not use it as the source of truth for benchmark latency.
---

# GPU Profiler Analysis

## Overview

Use this skill to capture or analyze one local GPU workload while preserving the
workload shape, tool configuration, git revision, and host metadata. Keep the
portable timeline format as Chrome trace JSON or JSON.GZ; keep Nsight's native
reports native when hardware-counter detail is required.

## Choose the path

1. Start from an existing trace or profile directory when one already exists.
2. Use the local runner when a command or adapter can reproduce the workload.
3. Use Torch for Perfetto and routine triage, Nsight Systems for graph/stream
   timelines, and Nsight Compute for a small set of kernels.
4. Use CUDA-event/CUPTI or CUDA-Graph benchmark results for latency decisions;
   profiler timings are diagnostic and may include instrumentation or replay cost.

## Analyze an existing artifact

```bash
python scripts/analyze_trace.py \
  --input /path/to/trace.json.gz \
  --output-dir artifacts/profile/run-001
```

The input may be a Chrome trace file or a directory containing one. The script
writes `summary.json` and `report.md`; it does not modify the input trace. Read
`references/trace-schema.md` when a producer uses a non-standard Chrome event
layout.

The report always separates measured facts from inferred labels. If a trace has
no CPU/GPU correlation or Python stack, report source mapping as unavailable;
never guess a source location from a kernel name.

## Capture a local workload

Use `--` to pass a command without shell interpretation:

```bash
python scripts/run_local_profile.py \
  --backend nsys \
  --output-dir artifacts/profile/nsys-001 \
  -- python -m benchmarks profile-pi05 --steps 10
```

For a workload-specific adapter, provide a JSON plan with `command`, `env`,
`expected_artifacts`, `workload`, and `capture` fields:

```bash
python scripts/run_local_profile.py \
  --backend torch \
  --plan /path/to/pi05-plan.json \
  --output-dir artifacts/profile/torch-001
```

The runner writes `manifest.json`, captures stdout/stderr, validates artifacts,
and returns the workload's exit code. The Torch command or adapter must create
the Chrome trace itself; the generic runner does not inject a profiler into an
arbitrary Python process. For the Torch backend it provides
`GPU_PROFILE_OUTPUT_DIR` as a convention for profile-aware workloads; Nsight
backends do not set that variable unless the plan explicitly asks for it.

## Backend rules

- Torch: use CPU + CUDA activities for timeline mode, and export Chrome JSON or
  JSON.GZ. Keep `record_shapes` and `with_stack` opt-in because they add cost.
- Nsight Systems: capture CUDA Graph, NVTX, and OS runtime events as needed;
  retain `.nsys-rep` and optionally export SQLite/JSON Lines. See
  `references/nsys-backend.md`.
- Nsight Compute: select a small number of kernels and record the exact filter,
  replay mode, graph mode, and section set in metadata. See
  `references/ncu-backend.md`.

## Reporting contract

Every run should preserve:

- workload shape and stage
- warmup and active steps
- profiler activities and expensive flags
- git revision and dirty state
- host/GPU/tool versions
- artifact paths relative to the run directory

Use `references/adapter-contract.md` for adapter design and
`references/capture-modes.md` to choose between summary, timeline, mapping, and
kernel-detail captures.
