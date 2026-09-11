# Capture artifacts

Reuse an existing applicable report in place. For a new capture, an ignored
`artifacts/profile/<run>/` directory is convenient; no fixed directory tree or
separate report bundle is required.

Keep the native `.ncu-rep` and the command, selected launch, workload/build,
GPU/tool versions, replay/cache/clock settings needed to interpret it. A short
experiment entry can hold this context and the conclusion. Export text or CSV
only when useful to a consumer.

```bash
mkdir -p artifacts/profile/attention
ncu --set basic --kernel-name 'regex:attention' --launch-count 1 \
  --export artifacts/profile/attention/kernel ./harness
```

`analyze_reports.py --run-dir` writes its derived analysis files under that
chosen directory. See [helper examples](../scripts/README.md).
