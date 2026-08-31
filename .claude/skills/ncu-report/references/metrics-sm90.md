# sm90 metric vocabulary — verified names, and how to re-verify

**Validity**: every name below holds for full-set (`--set=full`) sm90
reports produced by the host toolchain's ncu, Nsight Compute **2025.4.1**,
verified by enumerating `action.metric_names()` (a full-set report carries
~2.3 K metrics). Names drift between ncu versions; on any other toolchain,
re-run the enumeration before trusting this page:

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "/data/apps/cuda/13.1/nsight-compute-2025.4.1/extras/python")
import ncu_report
a = ncu_report.load_report("<rep>").range_by_idx(0).action_by_idx(0)
for n in sorted(a.metric_names()):
    print(n)
EOF
```

## Names that work here (all verified present)

The families the dimensions doc uses, plus the traps:

- **Launch/occupancy**: `launch__grid_size`, `launch__block_size`,
  `launch__waves_per_multiprocessor`, `launch__registers_per_thread`,
  `launch__shared_mem_per_block`, `launch__occupancy_limit_{blocks,registers,
  shared_mem,warps}`, `device__attribute_multiprocessor_count`,
  `sm__maximum_warps_per_active_cycle_pct`,
  `sm__warps_active.avg.pct_of_peak_sustained_active`.
- **SOL/timing**: `gpu__time_duration.sum` (ns),
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`,
  `l1tex__throughput.avg.pct_of_peak_sustained_active`,
  `lts__throughput.avg.pct_of_peak_sustained_elapsed`,
  `dram__bytes_{read,write}.sum` + `.pct_of_peak_sustained_elapsed` +
  `.per_second`.
- **Pipes**: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_{elapsed,active}`,
  `sm__inst_executed_pipe_{fma,lsu}.avg.pct_of_peak_sustained_active`.
- **Memory pattern**: `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`,
  `l1tex__t_{sectors,requests}_pipe_lsu_mem_global_op_ld.sum`,
  `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio`,
  `smsp__sass_inst_executed_op_{global,local,shared}_{ld,st}.sum` (the
  `sass_` spelling is the live one on this toolchain),
  `smsp__thread_inst_executed_per_inst_executed.ratio`.

## Stall reasons

Aggregate form `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`,
19 reasons verified:

```
barrier  branch_resolving  dispatch_stall  drain  gmma  imc_miss  lg_throttle
long_scoreboard  math_pipe_throttle  membar  mio_throttle  misc  no_instruction
not_selected  selected  short_scoreboard  sleeping  tex_throttle  wait
```

`gmma` is Hopper-specific (wgmma wait). Per-PC form
`smsp__pcsamp_warps_issue_stalled_<reason>` with paired `<reason>_not_issued`
variants (38 metrics) and `smsp__pcsamp_sample_count`. **Trap**: the
aggregate says `no_instruction`, the pcsamp says `no_instructions`.

## What a full-set report does and does not contain (2025.4.1)

- Per-PC stall samples: **present** even without `--section SourceCounters`
  in the wrapper — `hotspots` works on existing reports, PC/SASS-addressed.
- Source-line mapping: only with a `-lineinfo` build (the host's
  extra-nvcc-flags hook carries it); otherwise `source_info(pc)` is None.
  Verified end-to-end on this host: hotspots resolve to kernel source lines,
  headers included.
- `pmsampling:*` time-series: **absent from every report captured so far** —
  the host wrappers never request the sections. 2025.4.1 does list
  `PmSampling` / `PmSampling_WarpStates` (verified via `ncu
  --list-sections`), so the recipe is `--section PmSampling --section
  PmSampling_WarpStates` on top of the set — but whether the resulting
  metrics populate on this GPU/driver is a GAP until someone captures and
  checks `num_instances() > 0`. Until then the timeline lens falls back to
  nsys.
- Rule-engine results: present; their dict shape on this version is in
  `python-api.md`.
