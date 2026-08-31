# ncu_report parsing patterns (verified on 2025.4.1)

`report_query.py` covers the routine cases; these are the building blocks
for anything bespoke. All verified on the host's login node with the repo
venv (python 3.12) — no GPU needed.

## Module location

The module ships inside every Nsight Compute install; the one matching the
producer parses most reliably. On this host:

```
/data/apps/cuda/13.1/nsight-compute-2025.4.1/extras/python    # what the wrappers capture with
/usr/local/cuda-13.{2,3}/nsight-compute-2026.*/extras/python  # newer, for newer reports
```

`report_query.py` scans newest-first and falls through on load failure; set
`NCU_PYTHON_DIR` to pin one. The `_ncu_report*.so` beside the .py is built
per python version — if import fails under one interpreter, try the venv's.

## Load and address

```python
import sys
sys.path.insert(0, "/data/apps/cuda/13.1/nsight-compute-2025.4.1/extras/python")
import ncu_report
rng = ncu_report.load_report(path).range_by_idx(0)
action = rng.action_by_idx(i)          # one profiled kernel launch
```

A multi-launch capture (a wrapper profiling several modes, say) is one range
holding one action per launch, in launch order. Always wrap metric reads:

```python
def safe(action, name, default=None):
    try: return action[name].value()
    except Exception: return default   # absent = not collected on this version/set
```

## Per-PC attribution

```python
m = action["smsp__pcsamp_warps_issue_stalled_barrier"]
if m.num_instances() and m.has_correlation_ids():
    cor = m.correlation_ids()
    for i in range(m.num_instances()):
        pc, samples = cor.as_uint64(i), m.as_uint64(i)
        si = action.source_info(pc)            # None without -lineinfo
        site = f"{si.file_name()}:{si.line()}" if si else hex(pc)
```

On this version `sass_by_pc(address)` takes the PC as an argument (per-PC
lookup); it is not the no-arg dict some newer docs describe.

## Rule engine (version-specific shape)

On 2025.4.1, `action.rule_results_as_dicts()` yields keys
`rule_identifier`, `name`, `section_identifier`, `rule_message`,
`focus_metrics`, and (only when estimated) `speedup_estimation` — and the
last three are **stringified python dicts**, so parse with
`ast.literal_eval`, not json:

```python
import ast
for r in action.rule_results_as_dicts():
    msg = ast.literal_eval(r.get("rule_message", "{}"))       # {'title':…, 'message':…}
    est = ast.literal_eval(r.get("speedup_estimation", "{}")) # {'type':…, 'speedup':…}
```

Newer ncu versions rename these fields; treat the shape as per-version and
introspect `sorted(rules[0].keys())` before relying on it.

## The shape of a verified read

The end-to-end pass that validated all of the above, run on a
warp-specialized persistent kernel from the host's report store: occupancy
pinned to 1 CTA/SM by registers + smem (by design), barrier and
long_scoreboard dominating the stall ratios and concentrated on one or two
ring-wait PCs, tensor pipe far below its ceiling at a tiny shape, zero
spill, LSU nearly idle because the traffic is TMA — and with a `-lineinfo`
build, the same hotspots resolving to kernel source lines. Reproduce on any
full-set report with
`report_query.py summary|rules|hotspots <report>.ncu-rep --action N`.
