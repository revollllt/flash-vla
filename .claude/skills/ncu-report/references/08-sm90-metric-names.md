# H100 (sm90) Metric Name Reference

Every name below holds for **full-set (`--set full`) sm90 reports written by Nsight Compute 2025.4.1** (the `cuda/13.1` module on this cluster), verified by enumerating `action.metric_names()` on this repo's own reports (2 320 names each) and reading the values. Names drift between ncu versions and GPU generations: the upstream B200 list (`08-b200-metric-names.md`) is close but not identical, and older docs use names that exist on neither. On any other toolchain, re-run the enumeration before trusting a name:

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "/data/apps/cuda/13.1/nsight-compute-2025.4.1/extras/python")
import ncu_report
a = ncu_report.load_report("<rep>").range_by_idx(0).action_by_idx(0)
print(a.metric_names().__len__())
for n in sorted(a.metric_names()):
    print(n)
EOF
# or, for a chip without a report yet:
ncu --query-metrics --chip gh100 | grep -i <pattern>
```

---

## What differs from the upstream B200 / sm100 list

Every name in the upstream `B200_KEY_METRICS` list exists on sm90 / 2025.4.1 and returns a value — the *curated* list is portable. The differences are around it:

| Upstream (B200 doc) | sm90 / 2025.4.1 |
|---|---|
| `pmsampling:<metric>` timeline names | **absent.** Timelines are `warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>` (warp states) and `<UNIT>.TriageCompute.<metric>` (PM counters) — see below |
| `sm__pipe_tensor_subpipe_{hmma,imma,dmma}_cycles_active.*` | **absent.** Use `sm__pipe_tensor_op_{hmma,imma,dmma}_cycles_active.*` |
| `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg` | exists but **reads 0 on wgmma kernels** — no `gmma` ops-path counter exists. Use the pipe-cycles metrics |
| `lts__t_sectors_op_{atom,red}.sum` | **absent** from the full set (L2 atomics are read from the SASS-side `ATOM`/`RED` hotspots) |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.sum` | **absent.** Use `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` (+ `_write`) and `dram__bytes.sum.per_second` |
| `smsp__inst_executed_op_{global,local,shared}_{ld,st}.sum` | **absent.** The live spelling is `smsp__sass_inst_executed_op_*` (same trap as B200) |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | **absent.** Compute `sectors.sum / requests.sum` |
| `sm__inst_executed_pipe_fmaheavy.*` | only as the PM-sampled `TPC.TriageCompute.sm__inst_executed_pipe_fmaheavy_realtime...` |
| `lts__t_bytes.sum`, `lts__t_sectors_op_{read,write}.sum` | **absent.** Available: `lts__t_sectors.sum`, `lts__t_sectors_srcunit_tex_op_read.sum`, `lts__t_sector_op_{read,write}_hit_rate.pct` |
| stall reason list | sm90 adds **`gmma`** (aggregate) and **`warpgroup_arrive`** (per-PC), plus `imc_miss` in both forms |

And sm90 has families B200 docs never mention: TMA, wgmma, cluster / DSMEM, named barriers, spill counters, bank-conflict counters — listed below.

---

## Canonical sm90 metric set (curated, all verified present)

### Device
```
device__attribute_display_name                 # 'NVIDIA H100 80GB HBM3'
device__attribute_compute_capability_major/minor   # 9 / 0
device__attribute_multiprocessor_count         # 132
device__attribute_l2_cache_size                # 52428800 (50 MB)
device__attribute_max_shared_memory_per_multiprocessor   # 233472 (228 KB)
device__attribute_max_shared_memory_per_block_optin      # 232448 (227 KB)
device__attribute_max_warps_per_multiprocessor / _max_warps_per_scheduler   # 64 / 16
device__attribute_max_registers_per_multiprocessor / _per_thread            # 65536 / 255
device__attribute_cluster_launch                                            # cluster support flag
```

### Launch geometry / occupancy
```
launch__grid_size, launch__grid_dim_{x,y,z}, launch__user_grid_size
launch__block_size, launch__block_dim_{x,y,z}, launch__thread_count
launch__waves_per_multiprocessor
launch__registers_per_thread, launch__registers_per_thread_allocated
launch__shared_mem_per_block, _static, _dynamic, _driver, _allocated, launch__shared_mem_config_size
launch__occupancy_limit_{blocks,registers,shared_mem,warps,barriers}      # barriers is sm90's named-barrier limit
launch__occupancy_per_{block_size,register_count,shared_mem_size,barrier_count,cluster_size}   # curves
launch__cluster_size, launch__cluster_dim_{x,y,z}, launch__cluster_max_active, launch__cluster_max_potential_size
launch__cluster_scheduling_policy                       # string, e.g. 'PolicySpread'
launch__occupancy_cluster_pct, launch__occupancy_cluster_gpu_pct
launch__sm_count, launch__tpc_count, launch__persisting_l2_cache_size, launch__stack_size
launch__uses_cdp, launch__uses_green_context, launch__uses_mps, launch__uses_vgpu
sm__maximum_warps_per_active_cycle_pct                  # theoretical occupancy %
smsp__maximum_warps_avg_per_active_cycle                # theoretical warps per scheduler (max 16)
sm__warps_active.avg.pct_of_peak_sustained_active       # achieved occupancy %
sm__warps_active.{avg,max,min}.per_cycle_active
smsp__warps_active.avg.per_cycle_active, smsp__warps_eligible.{avg,max,min}.per_cycle_active
```

### SOL / throughput
```
sm__throughput.avg.pct_of_peak_sustained_elapsed            (+ breakdown:sm__throughput.* sub-metrics)
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed   (+ breakdown:*)
gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_active
lts__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum[.pct_of_peak_sustained_elapsed | .per_second]
dram__bytes_write.sum[.pct_of_peak_sustained_elapsed | .per_second]
dram__bytes.{avg,sum}.per_second, dram__sectors_{read,write}.sum
```

### Timing / cycles / issue
```
gpu__time_duration.sum                                  # ns (diagnostic only — never a latency claim)
gpc__cycles_elapsed.{avg,max,min,sum}
sm__cycles_active.{avg,max,min,sum}, sm__cycles_elapsed.*          # per-SM spread → balance
smsp__cycles_active.{avg,max,min,sum}, smsp__cycles_elapsed.*
sm__inst_executed.avg.per_cycle_active                  # IPC
sm__inst_issued.avg.per_cycle_active
smsp__issue_active.avg.{per_cycle_active,pct_of_peak_sustained_active}
smsp__average_warp_latency_per_inst_issued.ratio        # warp cycles per issued instruction
sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed
```

### Compute pipelines
```
sm__inst_executed_pipe_{adu,alu,cbu,fma,fma_type_fp16,fp64,lsu,tex,tma,uniform,xu}.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_tensor_op_{hmma,imma,dmma,gmma}.avg.pct_of_peak_sustained_{active,elapsed}   # gmma = wgmma issue share
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_{active,elapsed}
sm__pipe_tensor_op_{hmma,imma,dmma}_cycles_active.avg.pct_of_peak_sustained_{active,elapsed}         # wgmma counted under hmma
sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.*                          # all tensor types
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_{active,elapsed}                                # TMA unit busy
sm__ops_path_tensor_op_hmma_src_{bf16,fp16,tf32}_dst_{fp16,fp32}_sparsity_{off,on}.*                 # mma.sync only — 0 on wgmma
```

### TMA / bulk copies / DSMEM (sm90)
```
smsp__inst_executed_op_tma_{ld,st}.sum
smsp__sass_inst_executed_op_tma_{ld,st,red}.sum
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum[.pct_of_peak_sustained_elapsed | .per_second]   # bytes loaded by TMA
l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum
l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_{st,red}.sum[...]                                    # bytes stored / reduced by TMA
l1tex__m_{xbar2l1tex_read,l1tex2xbar_write}_sectors_mem_dshared_op_tma_{st,red}.sum                     # DSMEM via TMA
smsp__sass_inst_executed_op_dshared{,_ld,_st,_atom,_redas,_stas,_tma_red,_tma_st}.sum                   # DSMEM instructions
l1tex__tmain_requests.avg.pct_of_peak_sustained_elapsed
```

### Cache hit rates
```
l1tex__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_{ld,st,atom,red}_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_local_op_{ld,st}_hit_rate.pct
lts__t_sector_hit_rate.pct, lts__t_sector_op_{read,write}_hit_rate.pct
smsp__imc_request_hit_rate.pct, sm__icc_request_hit_rate.pct, idc__request_hit_rate.pct
```

### Memory access counts, sectors, coalescing
```
smsp__sass_inst_executed_op_global_{ld,st}.sum
smsp__sass_inst_executed_op_shared{,_ld,_st,_gmma}.sum        # shared_gmma = wgmma reading smem operands
l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum, ..._ld_lookup_{hit,miss}.sum
l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st}.sum          # sectors/request = sectors.sum / requests.sum (ideal 4)
smsp__sass_average_data_bytes_per_sector_mem_global_op_{ld,st}.ratio   # max 32
derived__memory_l2_theoretical_sectors_global_excessive       # the UncoalescedGlobalAccess rule's count
lts__t_sectors.sum, lts__t_sectors_srcunit_tex_op_read.sum[.per_second]
l1tex__data_pipe_lsu_wavefronts_mem_shared{,_op_ld,_op_st,_op_atom}.sum[.pct_of_peak_sustained_elapsed]
l1tex__data_bank_conflicts_pipe_lsu_mem_shared{,_op_ld,_op_st,_op_atom,_op_ldgsts}.sum
```

### Register spill / local memory
```
smsp__sass_inst_executed_op_local_{ld,st}.sum
sass__inst_executed_register_spilling_mem_local{,_op_read,_op_write}
l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum, l1tex__t_requests_pipe_lsu_mem_local_op_{ld,st}.sum
```

### Divergence / branches
```
smsp__thread_inst_executed_per_inst_executed.ratio            # 32 = no divergence
smsp__sass_average_branch_targets_threads_uniform.pct         # "Branch Efficiency"
smsp__sass_branch_targets_threads_divergent.{avg,sum}
smsp__inst_executed_op_branch.sum, derived__smsp__inst_executed_op_branch_pct
```

### Stall reasons — aggregate ratios (19, sm90)
```
smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio
  barrier  branch_resolving  dispatch_stall  drain  gmma  imc_miss  lg_throttle
  long_scoreboard  math_pipe_throttle  membar  mio_throttle  misc  no_instruction
  not_selected  selected  short_scoreboard  sleeping  tex_throttle  wait
```
`gmma` (warps waiting on wgmma completion) is Hopper-specific. `selected` is the productive baseline (= 1.0).

### Stall reasons — per-PC samples (in `--set full` here; 45 metrics)
```
smsp__pcsamp_sample_count
smsp__pcsamp_warps_issue_stalled_<reason>               (+ _not_issued twin)
  barrier  branch_resolving  dispatch_stall  drain  imc_miss  lg_throttle  long_scoreboard
  math_pipe_throttle  membar  mio_throttle  misc  no_instructions  not_selected  selected
  short_scoreboard  sleeping  tex_throttle  wait  warpgroup_arrive
smsp__pcsamp_aggregated_passes, _buffer_overflow, _buffer_size_bytes, _dropped_bytes, _interval, _interval_cycles
```
Each has `num_instances() > 0` with `correlation_ids()` = program counters; `action.source_info(pc)` maps them to `(file, line)` on a `-lineinfo` build. **Traps:** the sampler says `warpgroup_arrive` where the aggregate says `gmma` (neither name exists on the other side); `no_instruction` (aggregate) vs `no_instructions` (sampler).

### Timelines (PM sampling — in `--set full` here)
```
warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>       # 19 reasons + samples_data_dropped; ~140 timestamped instances
warpsampling:smsp__pcsamp_samples_data_dropped
TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
TPC.TriageCompute.sm__pipe_tensor_op_{hmma,imma}_cycles_active_realtime.avg
TPC.TriageCompute.sm__inst_executed_realtime.avg.{pct_of_peak_sustained_elapsed,per_cycle_active,per_cycle_elapsed}
TPC.TriageCompute.sm__inst_executed_pipe_{alu,fma,fmaheavy,fmalite}_realtime.avg.pct_of_peak_sustained_elapsed
TPC.TriageCompute.sm__pipe_fp64_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed
SM_A.TriageCompute.l1tex__throughput.avg.pct_of_peak_sustained_elapsed
SM_A.TriageCompute.sm__inst_executed_pipe_{uniform,xu}_realtime.avg.pct_of_peak_sustained_elapsed
SM_B.TriageCompute.l1tex__t_sector_hit_rate.pct, l1tex__t_sectors{,_lookup_hit,_lookup_miss}.sum
SM_C.TriageCompute.smsp__pipe_tensor_op_dmma_cycles_active.avg
LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed
LTS.TriageCompute.lts__average_t_sector_hit_rate{,_srcunit_tex}_realtime.pct
LTS.TriageCompute.lts__t_sector_throughput_{aperture_device,srcunit_gcc,srcunit_tex}.avg.pct_of_peak_sustained_elapsed
FBSP.TriageCompute.dramc__{read_,write_,}throughput.avg.pct_of_peak_sustained_elapsed
FE_B.TriageCompute.gr__ctas_launched_realtime.{avg,sum}.per_cycle_elapsed
GPC_B.TriageCompute.gpc__cgas_{active_realtime,launched}.*.per_cycle_elapsed
profiler__pmsampler_{buffer_size_bytes,interval_time,merged_samples,pass_groups}     # per pass group
profiler__pmsampler_ctxsw_<g>, profiler__timestamp_workload_{start,end}_<g>          # window bounds per pass group
```
Correlation ids are GPU timestamps (ns). The window is longer than the kernel; samples outside `profiler__timestamp_workload_{start,end}_<g>` read 0. At the default 1.5 µs interval a 12 µs kernel gets under ten in-kernel samples.

### Rule engine
`action.rule_results_as_dicts()` → keys `focus_metrics, name, result_tables, rule_identifier, rule_message, section_identifier, speedup_estimation` (shape in `04-python-api.md`). Rule identifiers seen on this repo's reports: `SOLBottleneck, IssueSlotUtilization, CPIStall, HighPipeUtilization, FPInstructions, LaunchConfiguration, MemoryCacheAccessPattern, MemoryL2Compression, PMSamplingData, SOLFPRoofline, TheoreticalOccupancy, UncoalescedGlobalAccess, WorkloadImbalance`.

---

## What a full-set report does and does not contain (2025.4.1, sm90)

- Per-PC stall samples: **present** without `--section SourceCounters`; `hotspots` works on every existing report.
- Source-line mapping: only with a `-lineinfo` build; otherwise `source_info(pc)` is `None` and sites are PCs.
- Warp-state and PM-counter timelines: **present** (`warpsampling:`, `*.TriageCompute.*`) — the `pmsampling:` names of the B200 docs are not.
- L2 atomic sector counters, `dram__throughput`, tensor `subpipe` names: **absent**.
- Rule-engine results: present, dict-shaped as above.

---

## Gotchas

1. **Metric exists in `--query-metrics` but not in the report**: it wasn't collected — add the section or use `--metrics`.
2. **Value is `0.0`**: the counter is genuinely zero (no tensor activity), or the metric is derived from something not collected, or — for `sm__ops_path_tensor_op_hmma_*` — the work was wgmma, which it does not count.
3. **`.avg` vs `.sum` vs `.max`**: separate names; `.avg` for rates/percentages, `.sum` for counts, `.max`/`.min` for spread.
4. **`pct_of_peak_sustained_elapsed` vs `_active`**: `_elapsed` normalizes against total kernel time (idle SMs and the ramp count against you); `_active` against cycles the unit was running. `_elapsed` is more honest for under-filled kernels; `_active` for "how hard was the busy unit working". Quote which.
5. **Percent of *what* peak**: NCU's peak is the datasheet frame. The reachable ceiling on this machine is the tagged constant in `hardware-unit-test` — compare against that before calling a number bad.
