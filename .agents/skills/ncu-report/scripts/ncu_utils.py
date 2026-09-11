"""Shared helpers for parsing Nsight Compute reports.

Usage:
    from ncu_utils import load_action, safe, key_metrics_for, rule_speedups

Design-time only: needs the `ncu_report` module that ships inside any Nsight
Compute install (no GPU, no torch), so analysis can run on a CPU-only host.  The module
is located via NCU_PYTHON_DIR, else by scanning the Nsight Compute installs
under the roots below, newest first; a report written by a newer ncu than the
module can fail to load, so `load_report` tries the next candidate rather than
giving up.  A newer module reads older reports (2026.2.1 on a 2025.4.1 report
is verified).

API shapes that differ between ncu versions (rule dicts, `sass_by_pc`) are
handled here so the scripts above stay version-agnostic.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys
from pathlib import Path

# --- Locate ncu_report -------------------------------------------------------

_SEARCH_ROOTS = (
    "/usr/local/cuda-*/nsight-compute-*",
    "/usr/local/cuda/nsight-compute",
    "/opt/nvidia/nsight-compute-*",
    "/opt/nvidia/nsight-compute/*",
    "/opt/cuda/nsight-compute-*",
)


def _version_key(path: str) -> tuple:
    m = re.search(r"nsight-compute-(\d+)\.(\d+)\.(\d+)", path)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def ncu_python_candidates() -> list[str]:
    """Directories that hold an `ncu_report.py`, newest ncu version first."""
    override = os.environ.get("NCU_PYTHON_DIR")
    if override:
        return [override]
    dirs: set[str] = set()
    for pat in _SEARCH_ROOTS:
        dirs.update(glob.glob(pat))
    out = []
    for d in sorted(dirs, key=_version_key, reverse=True):
        py = os.path.join(d, "extras", "python")
        if os.path.isfile(os.path.join(py, "ncu_report.py")):
            out.append(py)
    return out


def _import_ncu_report(candidate: str):
    sys.path.insert(0, candidate)
    sys.modules.pop("ncu_report", None)
    try:
        import ncu_report  # noqa: PLC0415
        return ncu_report
    except Exception:
        sys.path.remove(candidate)
        raise


def load_report(path, verbose: bool = True):
    """Load a .ncu-rep, trying ncu_report modules newest-first.

    Returns the report object.  Keep it alive while using its actions.
    """
    errors = []
    for cand in ncu_python_candidates():
        try:
            mod = _import_ncu_report(cand)
            report = mod.load_report(str(path))
            if verbose:
                print(f"[ncu_report from {cand}]", file=sys.stderr)
            return report
        except Exception as exc:  # try the next install
            errors.append(f"  {cand}: {type(exc).__name__}: {exc}")
    raise SystemExit(
        "no usable ncu_report module could load this report "
        "(set NCU_PYTHON_DIR to the extras/python dir of a matching Nsight Compute):\n"
        + "\n".join(errors)
    )


def action_of(report, index: int = 0):
    """The `index`-th profiled launch of the first range, with a helpful error."""
    rng = report.range_by_idx(0)
    n = rng.num_actions()
    if not 0 <= index < n:
        names = ", ".join(rng.action_by_idx(i).name()[:40] for i in range(n))
        raise SystemExit(f"--action {index} out of range; report has {n} action(s): {names}")
    return rng.action_by_idx(index)


def load_action(path, action_idx: int = 0, verbose: bool = True):
    """(report, action) for one launch.  The report is returned so it stays alive."""
    report = load_report(path, verbose=verbose)
    return report, action_of(report, action_idx)


def module_version(report) -> str:
    """Version of the ncu_report module that loaded the report -- NOT the writer's.

    `IContext.get_version()` reports the parsing module (verified: the same
    2025.4.1 report answers 2025.4.1 or 2026.2.1 depending on the module).
    The writer's version comes from the capture job's log (`ncu --version`).
    """
    try:
        return str(report.get_version())
    except Exception:
        return "?"


# --- Safe metric access ------------------------------------------------------

def safe(action, name: str, default=None):
    """Metric value, or `default` if the metric is missing / not collected."""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names, default=None) -> dict:
    return {n: safe(action, n, default) for n in names}


def metric_or_none(action, *candidates):
    """First candidate name that yields a value -- for names that differ by GPU / ncu version."""
    for n in candidates:
        v = safe(action, n, None)
        if v is not None:
            return v
    return None


def has_metric(action, name: str) -> bool:
    try:
        action[name]
        return True
    except Exception:
        return False


# --- Value-kind robust accessors ----------------------------------------------

def metric_value_at(m, i: int):
    """The i-th instance value regardless of value kind."""
    try:
        k = m.kind()
        if k == m.ValueKind_UINT64:
            return m.as_uint64(i)
        if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
            return m.as_double(i)
        if k == m.ValueKind_STRING:
            return m.as_string(i)
    except Exception:
        pass
    for fn in ("as_uint64", "as_double", "as_string"):
        try:
            return getattr(m, fn)(i)
        except Exception:
            continue
    return None


def per_instance_values(action, metric_name: str):
    """List of per-instance values, or None if the metric has none."""
    try:
        m = action[metric_name]
        n = m.num_instances()
    except Exception:
        return None
    if n == 0:
        return None
    return [metric_value_at(m, i) for i in range(n)]


def per_instance_series(action, metric_name: str):
    """List of (correlation_id, value) -- PCs for per-PC metrics, GPU timestamps (ns) for timelines."""
    try:
        m = action[metric_name]
        n = m.num_instances()
    except Exception:
        return []
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        try:
            cid = cor.as_uint64(i)
        except Exception:
            try:
                cid = int(cor.as_double(i))
            except Exception:
                cid = None
        out.append((cid, metric_value_at(m, i)))
    return out


# --- PC -> source line ---------------------------------------------------------

def per_pc_values(action, metric_name: str):
    """[(pc, value)] for a per-PC metric (correlation ids are program counters)."""
    return per_instance_series(action, metric_name)


def pc_to_source_line(action, pc):
    """(file, line) for a PC, or ('?', 0) without -lineinfo."""
    try:
        si = action.source_info(pc)
        if si is None:
            return "?", 0
        return si.file_name(), si.line()
    except Exception:
        return "?", 0


def sass_at(action, pc) -> str:
    """SASS text for one PC.  2025.4.1 takes the address; older docs describe a no-arg dict."""
    try:
        return str(action.sass_by_pc(pc))
    except TypeError:
        try:
            return str(action.sass_by_pc().get(pc, ""))
        except Exception:
            return ""
    except Exception:
        return ""


# --- Stall reasons -------------------------------------------------------------

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"
PCSAMP_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
WARPSAMP_PREFIX = "warpsampling:smsp__pcsamp_warps_issue_stalled_"


def stall_ratios(action) -> list[tuple[str, float]]:
    """[(reason, warps stalled per issue)] sorted worst first (aggregate ratios)."""
    rows = []
    for name in action.metric_names():
        if name.startswith(STALL_PREFIX) and name.endswith(STALL_SUFFIX):
            v = safe(action, name)
            if v is not None:
                rows.append((name[len(STALL_PREFIX):-len(STALL_SUFFIX)], v))
    return sorted(rows, key=lambda r: r[1], reverse=True)


def pcsamp_metrics(action, include_not_issued: bool = False) -> list[str]:
    """The per-PC stall-sample metric names present in this report."""
    out = []
    for name in action.metric_names():
        if not name.startswith(PCSAMP_PREFIX):
            continue
        if name.endswith("_not_issued") and not include_not_issued:
            continue
        out.append(name)
    return sorted(out)


# --- Timelines (PM sampling / warp-state sampling) ------------------------------

def timeline_metrics(action) -> list[str]:
    """Timeline series present: `warpsampling:*`, `pmsampling:*` and `<UNIT>.TriageCompute.*`."""
    out = []
    for name in action.metric_names():
        if name.startswith(("warpsampling:", "pmsampling:")) or ".Triage" in name:
            out.append(name)
    return sorted(out)


def workload_windows(action) -> list[tuple[int, int]]:
    """[(start_ns, end_ns)] per PM-sampling pass group, from profiler__timestamp_workload_{start,end}_<g>.

    The timestamps are the *correlation ids* of those one-instance metrics.
    """
    out = []
    g = 0
    while True:
        s = per_instance_series(action, f"profiler__timestamp_workload_start_{g}")
        e = per_instance_series(action, f"profiler__timestamp_workload_end_{g}")
        if not s or not e or s[0][0] is None or e[0][0] is None:
            break
        out.append((int(s[0][0]), int(e[0][0])))
        g += 1
    return out


def timeline_series(action, metric_name: str, trim: bool = True):
    """(timestamps_ns, values) for a timeline metric, trimmed to the pass group's workload window.

    Returns (None, None) when the metric has no instances.  If no window bounds
    are recorded, or none overlaps the series, leading/trailing zeros are trimmed.
    """
    series = per_instance_series(action, metric_name)
    if not series:
        return None, None
    series = [(t, (v if v is not None else 0.0)) for t, v in series if t is not None]
    if not series:
        return None, None
    if trim:
        t0, t1 = series[0][0], series[-1][0]
        for ws, we in workload_windows(action):
            if we >= t0 and ws <= t1:
                inside = [(t, v) for t, v in series if ws <= t <= we]
                if inside:
                    series = inside
                    break
        else:
            lead = 0
            while lead < len(series) and not series[lead][1]:
                lead += 1
            trail = len(series)
            while trail > lead and not series[trail - 1][1]:
                trail -= 1
            series = series[lead:trail] or series
    return [t for t, _ in series], [float(v) for _, v in series]


# --- Archive all metrics -------------------------------------------------------

def dump_all_metrics(action, outfile) -> int:
    """Every metric name + value (+ unit) to JSON.  Returns the entry count."""
    out = []
    for n in sorted(action.metric_names()):
        rec = {"name": n}
        try:
            m = action[n]
            try:
                rec["value"] = m.value()
            except Exception as e:
                rec["error"] = str(e)
            try:
                rec["unit"] = m.unit()
            except Exception:
                pass
            try:
                ni = m.num_instances()
                if ni:
                    rec["instances"] = ni
            except Exception:
                pass
        except Exception as e:
            rec["error"] = str(e)
        out.append(rec)
    Path(outfile).write_text(json.dumps(out, indent=1, default=str))
    return len(out)


# --- Curated metric sets --------------------------------------------------------
#
# KEY_METRICS_COMMON is the upstream B200 list; every name in it is verified to
# exist on sm90 with ncu 2025.4.1 as well, so it is the portable core.
# KEY_METRICS_SM90 adds the Hopper families (wgmma, TMA, cluster/DSMEM, named
# barriers, spill, bank conflicts) -- see ../references/08-sm90-metric-names.md.
# KEY_METRICS_SM100 adds the tensor sub-pipe names the B200 doc lists.
# `key_metrics_for(action)` picks by compute capability.

KEY_METRICS_COMMON = [
    # Launch geometry
    "launch__grid_size",
    "launch__block_size",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__block_dim_x",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__thread_count",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "device__attribute_multiprocessor_count",
    "device__attribute_max_warps_per_multiprocessor",
    # Timing
    "gpu__time_duration.sum",
    "smsp__cycles_active.avg",
    # SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.max.per_cycle_active",
    "sm__warps_active.min.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_eligible.max.per_cycle_active",
    # IPC
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
    # Tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg",
    # DRAM
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    # Caches
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
    # Memory instruction counts
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # Sectors / requests (coalescing)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    # Stall reasons -- aggregate ratios
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    # Stall reasons -- per-PC totals
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_membar",
]

KEY_METRICS_SM90 = [
    # Device / occupancy extras
    "device__attribute_display_name",
    "device__attribute_compute_capability_major",
    "device__attribute_l2_cache_size",
    "launch__registers_per_thread_allocated",
    "launch__shared_mem_per_block_dynamic",
    "launch__occupancy_limit_barriers",
    "smsp__maximum_warps_avg_per_active_cycle",
    "launch__cluster_size",
    "launch__cluster_dim_x",
    "launch__occupancy_cluster_pct",
    # Spread across SMs (balance)
    "sm__cycles_active.avg",
    "sm__cycles_active.max",
    "sm__cycles_active.min",
    # Issue
    "smsp__average_warp_latency_per_inst_issued.ratio",
    # Tensor pipe, wgmma
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
    "smsp__sass_inst_executed_op_shared_gmma.sum",
    # TMA / bulk
    "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tma.avg.pct_of_peak_sustained_active",
    "smsp__sass_inst_executed_op_tma_ld.sum",
    "smsp__sass_inst_executed_op_tma_st.sum",
    "smsp__sass_inst_executed_op_tma_red.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum.pct_of_peak_sustained_elapsed",
    "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_red.sum",
    "smsp__sass_inst_executed_op_dshared.sum",
    # L2 traffic / hit split
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors.sum",
    "lts__t_sector_op_read_hit_rate.pct",
    "lts__t_sector_op_write_hit_rate.pct",
    # Coalescing extras
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
    "derived__memory_l2_theoretical_sectors_global_excessive",
    # Shared memory conflicts
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Spill
    "sass__inst_executed_register_spilling_mem_local",
    # Divergence
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    # sm90 stall names
    "smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio",
    "smsp__pcsamp_warps_issue_stalled_warpgroup_arrive",
    "smsp__pcsamp_warps_issue_stalled_imc_miss",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
]

KEY_METRICS_SM100 = [
    "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_subpipe_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
]

# Backwards-compatible alias for the upstream name.
B200_KEY_METRICS = KEY_METRICS_COMMON


def compute_capability(action) -> tuple[int, int]:
    major = safe(action, "device__attribute_compute_capability_major", 0) or 0
    minor = safe(action, "device__attribute_compute_capability_minor", 0) or 0
    return int(major), int(minor)


def key_metrics_for(action) -> list[str]:
    """The curated key-metric list for this report's GPU generation."""
    major, _ = compute_capability(action)
    if major == 9:
        return KEY_METRICS_COMMON + KEY_METRICS_SM90
    if major >= 10:
        return KEY_METRICS_COMMON + KEY_METRICS_SM100
    return KEY_METRICS_COMMON


# --- NCU rule results -----------------------------------------------------------

def _literal(text, default=None):
    """Rule payloads are dicts on 2025.4.1 and stringified dicts on some other builds."""
    if isinstance(text, (dict, list)):
        return text
    if text is None:
        return default
    try:
        return ast.literal_eval(text)
    except Exception:
        return default


def rule_results(action) -> list[dict]:
    try:
        return list(action.rule_results_as_dicts())
    except Exception:
        return []


def rule_speedups(action) -> list[tuple]:
    """[(est_speedup_pct or None, identifier, title, message, section)] sorted by estimate desc.

    Tolerates both dict shapes: 2025.4.1 (`rule_identifier`, `rule_message`
    {'title','message'}, `speedup_estimation` {'type','speedup'}) and the
    upstream B200 shape (`rule_name`, `message_for_display`,
    `estimated_speedup_pct`).
    """
    out = []
    for r in rule_results(action):
        ident = r.get("rule_identifier") or r.get("rule_name") or r.get("name") or "?"
        section = r.get("section_identifier") or ""
        msg = _literal(r.get("rule_message"), {}) or {}
        title = msg.get("title", "") if isinstance(msg, dict) else ""
        message = msg.get("message", "") if isinstance(msg, dict) else str(msg)
        if not message:
            message = r.get("message_for_display") or ""
        est = None
        se = _literal(r.get("speedup_estimation"), None)
        if isinstance(se, dict) and se.get("speedup") is not None:
            est = float(se["speedup"])
        elif r.get("estimated_speedup_pct") is not None:
            try:
                est = float(r["estimated_speedup_pct"])
            except Exception:
                est = None
        out.append((est, ident, title, " ".join(str(message).split()), section))
    out.sort(key=lambda x: (x[0] if x[0] is not None else -1.0), reverse=True)
    return out


# --- Report header --------------------------------------------------------------

def describe(report, action) -> str:
    """One line of provenance for report headers."""
    cc = compute_capability(action)
    return (f"kernel={action.name()} device={safe(action, 'device__attribute_display_name', '?')} "
            f"cc={cc[0]}.{cc[1]} sms={safe(action, 'device__attribute_multiprocessor_count', '?')} "
            f"ncu_report_module={module_version(report)} metrics={len(action.metric_names())}")
