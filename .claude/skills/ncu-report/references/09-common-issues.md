# Common NCU issues

| Symptom | Focused check |
|---|---|
| `ERR_NVGPUCTRPERM` | Counter access is restricted. Use the machine's authorized profiling setup; node selection and administrator changes belong to user-local guidance. |
| Clock-control permission error | Use an available authorized clock policy. `--clock-control=none` avoids profiler clock locking; record the choice. |
| CUDA initialization failure | Check the actual driver, runtime and loaded compatibility libraries before attributing the error. Local node anecdotes are not general CUDA rules. |
| Report will not open | Compare writer and parser versions. Set `NCU_PYTHON_DIR` to a matching installation's `extras/python`; use a Python version supported by that module. |
| Kernel filter matches nothing | Check the demangled name and verify that the uninstrumented driver actually launches that path. |
| Multiple actions disagree | Select the intended launch with `--action N`; record its mapping to workload/mode. |
| Source lines missing | Build the selected kernel with `-lineinfo` before capture. JIT flags must reach the actual compiler; otherwise use SASS/PC evidence. |
| Capture is slow | Reduce launch count and requested sections. Broad sets may need many replays; do not reduce to an unrepresentative shape merely for convenience. |

## Version and architecture details

- In Nsight Compute 2025.4.1, rule dictionaries use `rule_identifier`,
  `rule_message` and `speedup_estimation`; helpers also accept upstream variants.
- Sampling series in those reports may be `warpsampling:*` and
  `*.TriageCompute.*`, rather than `pmsampling:*`. Use `plot_timeline.py --list`
  to inspect the report. Sparse samples from a short kernel cannot justify a
  detailed timeline conclusion.
- On sm90, ops-path counters can miss wgmma work. Read the tensor/GMMA activity
  metrics in [the sm90 vocabulary](08-sm90-metric-names.md).
- Low occupancy may be deliberate in a persistent kernel. Rule speedup estimates
  need interpretation against its launch geometry and dependencies.
- Profiler replay changes execution conditions. If a kernel fails only under
  NCU, inspect replay support, dependencies and initialized state; retain the
  failure instead of accepting a partial report.

Compare structural counters as diagnostic evidence and confirm latency with a
separate benchmark. Estimate uncertainty from actual measurements rather than
assuming a fixed 5% noise floor.
