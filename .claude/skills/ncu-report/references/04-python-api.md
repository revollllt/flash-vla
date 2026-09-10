# `ncu_report` Python API

The helpers support offline report analysis without a GPU. Examples below use
Nsight Compute 2025.4.1 report shapes; inspect available metrics on other versions.

## Module location

The module ships in a Nsight Compute installation's `extras/python` directory.
Set `NCU_PYTHON_DIR` to that directory for a compatible installation. The helper
also searches standard install locations. Use a Python version supported by the
module's native extension; a newer parser may read an older report, but verify
that compatibility rather than assuming it.

## Basic loading

```python
import os, sys
sys.path.insert(0, os.environ["NCU_PYTHON_DIR"])
import ncu_report

report = ncu_report.load_report("artifacts/profile/<run>/reports/full_<tag>.ncu-rep")
print(report.get_version())            # version of the *parsing module*, not the writer's — the writer is in the job log (`ncu --version`)

rng = report.range_by_idx(0)           # one range per profiled region; with -c 1 there is one
n = rng.num_actions()                  # one action per profiled launch
action = rng.action_by_idx(0)          # a parity script with several modes → several actions

print(action.name())                   # 'ffn_taskloop_kernel'
print(len(action.metric_names()))      # ~2320 on a full-set sm90 report
```

A multi-launch capture is one range holding one action per launch, in launch order — `--action N` in every helper. Record the index → mode mapping in the report.

---

## Reading a single metric

```python
def safe(action, name, default=None):
    """Metric value, or default if missing / not collected."""
    try:
        return action[name].value()
    except Exception:
        return default

sm_util   = safe(action, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
dram_rd   = safe(action, "dram__bytes_read.sum.per_second")          # bytes/s
dur_ns    = safe(action, "gpu__time_duration.sum")                    # ns (check .unit())
tma_bytes = safe(action, "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum")
```

**Always wrap reads.** A name that exists on one GPU or ncu version raises `KeyError` on another (`08-sm90-metric-names.md` lists what holds here).

---

## Enumerating available metrics

```python
for name in sorted(action.metric_names()):
    if "issue_stalled" in name and name.endswith(".ratio"):
        print(name, "=", safe(action, name))
```

This is how the sm90 vocabulary in `08-sm90-metric-names.md` was built — enumerate, don't guess.

---

## Instanced metrics: the three families in a 2025.4.1 full-set report

`value()` returns the aggregate; `num_instances()` > 0 means per-instance samples exist, and `correlation_ids()` says what each instance is keyed by.

| Family | Name shape | Correlation id | Typical count |
|---|---|---|---|
| Per-PC warp sampling | `smsp__pcsamp_warps_issue_stalled_<reason>` (+ `_not_issued`), `smsp__pcsamp_sample_count` | **program counter** (uint64) | ~200 PCs |
| Warp-state timeline | `warpsampling:smsp__pcsamp_warps_issue_stalled_<reason>` | **GPU timestamp (ns)** | ~140 samples |
| PM-sampling timeline | `<UNIT>.TriageCompute.<metric>` e.g. `TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed` | GPU timestamp (ns) | ~150 samples |

The timeline series are what the B200 docs address as `pmsampling:*`; on this version the same data lives under `warpsampling:` and the `Triage` groups. Their sample window is the PM-sampling window, longer than the kernel — instances before and after the launch read 0. `profiler__timestamp_workload_start_<g>` / `_end_<g>` (one instance each, per pass group `g`) bound the kernel; `scripts/plot_timeline.py` uses them to trim.

```python
m = action["warpsampling:smsp__pcsamp_warps_issue_stalled_barrier"]
cor = m.correlation_ids()
series = [(cor.as_uint64(i), m.as_double(i)) for i in range(m.num_instances())]   # (t_ns, warps)
```

---

## Per-PC → per-source-line mapping

```python
def per_pc_stalls(action, stall_metric):
    m = action[stall_metric]
    if m.num_instances() == 0 or not m.has_correlation_ids():
        return []
    cor, out = m.correlation_ids(), []
    for i in range(m.num_instances()):
        pc = cor.as_uint64(i)
        si = action.source_info(pc)                   # None without -lineinfo
        site = (si.file_name(), si.line()) if si else ("?", pc)
        out.append((site, m.as_uint64(i)))
    return out
```

Aggregate by `(file, line)` and sort by total — `scripts/extract_stall_hotspots.py` is the complete implementation, `report_query.py hotspots` the terminal view. Verified end-to-end on this host: with `-lineinfo` the hotspots resolve to kernel source lines, CUTLASS headers included (`cutlass/arch/barrier.h:426` is the ring wait in a task loop).

Per-PC lookups on 2025.4.1:

```python
action.sass_by_pc(pc)      # SASS text for one PC — takes the address; the no-arg dict form does not exist here
action.ptx_by_pc(pc)       # PTX, only when the build kept it
action.source_files()      # map file name → contents when embedded (empty on -lineinfo-only builds)
action.source_markers()
```

---

## Discovering value kind

```python
def metric_val(m, i=None):
    if i is None:
        return m.value()
    k = m.kind()
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    try:
        return m.as_uint64(i)
    except Exception:
        return m.as_double(i)
```

Per-PC counts are uint64; the timeline series are doubles. `ncu_utils.metric_value_at` does this.

---

## Useful `action` / `metric` methods (2025.4.1)

```python
# Action (= one kernel launch)
action.name()                   # kernel name (NameBase_FUNCTION / _DEMANGLED / _MANGLED selectable)
action.metric_names()
action.metric_by_name(name)     # same as action[name]
action.source_info(pc)          # SourceInfo(file_name(), line()) or None
action.sass_by_pc(pc) / action.ptx_by_pc(pc)
action.rule_results_as_dicts()  # rule engine, list of dicts (shape below)
action.workload_type()          # WorkloadType_KERNEL / _GRAPH / _RANGE / _CMDLIST
action.nvtx_state()

# Metric
m.value(); m.unit(); m.description(); m.kind(); m.rollup_operation()
m.num_instances(); m.has_correlation_ids(); m.correlation_ids()
m.as_uint64(i) / m.as_double(i) / m.as_string(i)
```

---

## Extracting NCU's rule suggestions

The rule engine — the `OPT Est. Speedup: X%` bullets of the details page — is structured data. **Its dict shape is version-specific.** On 2025.4.1:

```python
for r in action.rule_results_as_dicts():
    ident   = r["rule_identifier"]            # 'CPIStall', 'TheoreticalOccupancy', ...
    section = r["section_identifier"]         # 'WarpStateStats', 'Occupancy', ...
    msg     = r.get("rule_message") or {}     # {'title': 'Barrier Stalls', 'message': '...'}  (a dict here;
                                              #  some builds return it as a *stringified* dict)
    est     = r.get("speedup_estimation")     # {'type': 1|2, 'speedup': 31.4} or None
    focus   = r.get("focus_metrics")          # [{'name':..., 'value':..., 'severity':..., 'info':...}, ...]
```

Keys on this version: `focus_metrics, name, result_tables, rule_identifier, rule_message, section_identifier, speedup_estimation`. The upstream B200 helpers expect `rule_name` / `message_for_display` / `estimated_speedup_pct`, which **do not exist here** — `ncu_utils.rule_speedups()` accepts both shapes and parses stringified payloads with `ast.literal_eval`. `type` distinguishes NCU's global and local estimates (the details page labels the local ones `Est. Local Speedup`). Introspect `sorted(rules[0].keys())` on any new version before relying on a field.

Sort by estimate descending for the highest-impact suggestions first; never add estimates together.

---

## Comparing two reports programmatically

```python
def compare(a, b, metrics):
    print(f"{'metric':<72} {'A':>12} {'B':>12} {'delta':>8}")
    for m in metrics:
        va, vb = safe(a, m), safe(b, m)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            print(f"{m:<72} {va:>12.4g} {vb:>12.4g} {100*(vb-va)/va:>+7.1f}%")
        else:
            print(f"{m:<72} {str(va):>12} {str(vb):>12}")
```

`report_query.py compare` does this for the headline metrics plus the top stall ratios; `analyze_reports.py` with two tags writes the full curated table. Compare counts and ratios for diagnosis; determine timing uncertainty with repeated uninstrumented measurements.

---

## Saving everything for later

```python
import json
from pathlib import Path

def dump_all(action, outpath):
    rows = []
    for name in sorted(action.metric_names()):
        try:
            m = action[name]
            rows.append({"name": name, "value": m.value(), "unit": m.unit()})
        except Exception as e:
            rows.append({"name": name, "error": str(e)})
    Path(outpath).write_text(json.dumps(rows, indent=1, default=str))
```

`analyze_reports.py` writes this as `metrics_all_<tag>.json`; re-analysis then needs no `.ncu-rep`.

---

## Gotchas

- **`KeyError` on a metric that "should" exist**: different name on this GPU / version. Check `08-sm90-metric-names.md`, then enumerate.
- **`num_instances() == 0` where per-instance data was expected**: the producing section wasn't collected (on 2025.4.1 `--set full` has all three families; a `--metrics`-only capture has none).
- **`source_info(pc)` returns None**: no `-lineinfo` in the build (check the hook variable was exported *in the job*).
- **Rule payload is a string**: an older/newer build stringified it — `ast.literal_eval`, not `json.loads`.
- **A newer module than the writer**: reads fine (2026.2.1 on a 2025.4.1 report verified); the reverse fails to load — the helpers try the next install.
- **Value is a string like `PolicySpread`**: an enum (`launch__cluster_scheduling_policy`); use `as_string()` / `value()` and expect text.
