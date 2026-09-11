# Nsight Compute backend

Use Nsight Compute only after a timeline or benchmark identifies a small set of
important kernels. Record:

- kernel name/filter and launch skip/count
- graph profiling mode (`node` or `graph`)
- replay mode
- section set or explicit metrics
- clock/cache policy when available

Prefer a reproducible isolated kernel or a narrowly filtered graph capture. NCU
replays metric passes and may serialize or otherwise perturb the workload, so
its report explains microarchitectural limits but does not replace the latency
benchmark.

Keep `.ncu-rep` native and summarize selected metrics separately if a Markdown
report is needed.
