#!/usr/bin/env python3
"""Query an Nsight Compute report from the command line, without the GUI.

Design-time tool: needs only the `ncu_report` module that ships inside any
Nsight Compute install (no GPU, no torch), so it runs on the login node.

    report_query.py summary  <report.ncu-rep> [--action N]
    report_query.py rules    <report.ncu-rep> [--action N]
    report_query.py hotspots <report.ncu-rep> [--action N] [--top K]
    report_query.py compare  <a.ncu-rep> <b.ncu-rep> [--action N]

The module is located via NCU_PYTHON_DIR, else by scanning the Nsight Compute
installs under the host-specific roots below, newest first (NCU_PYTHON_DIR is
the portable override when porting). A report written by a newer ncu than the
module can fail to load; the scanner tries the next candidate rather than
giving up.
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import sys
from collections import defaultdict

_SEARCH_ROOTS = ("/data/apps/cuda/*/nsight-compute-*", "/usr/local/cuda-*/nsight-compute-*")


def _module_candidates() -> list[str]:
    override = os.environ.get("NCU_PYTHON_DIR")
    if override:
        return [override]
    dirs: list[str] = []
    for pat in _SEARCH_ROOTS:
        dirs.extend(sorted(glob.glob(pat), reverse=True))
    return [os.path.join(d, "extras", "python") for d in dirs
            if os.path.isfile(os.path.join(d, "extras", "python", "ncu_report.py"))]


def load(path: str):
    """Import ncu_report and load the report, trying installs newest-first."""
    errors = []
    for cand in _module_candidates():
        sys.path.insert(0, cand)
        try:
            import ncu_report  # noqa: PLC0415
            report = ncu_report.load_report(path)
            print(f"[ncu_report from {cand}]")
            return report
        except Exception as exc:  # try the next install
            errors.append(f"  {cand}: {type(exc).__name__}: {exc}")
            sys.path.pop(0)
            sys.modules.pop("ncu_report", None)
    raise SystemExit("no usable ncu_report module could load this report:\n" + "\n".join(errors))


def action_of(report, index: int):
    rng = report.range_by_idx(0)
    n = rng.num_actions()
    if not 0 <= index < n:
        names = ", ".join(rng.action_by_idx(i).name()[:40] for i in range(n))
        raise SystemExit(f"--action {index} out of range; report has {n} action(s): {names}")
    return rng.action_by_idx(index)


def safe(action, name: str, default=None):
    try:
        return action[name].value()
    except Exception:
        return default


def fmt(value, width: int = 10) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:{width}.3f}"
    return f"{value:{width}}"


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
HEADLINE = [
    ("duration us", "gpu__time_duration.sum", 1e-3),
    ("SM SOL %", "sm__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("mem SOL %", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("DRAM rd %", "dram__bytes_read.sum.pct_of_peak_sustained_elapsed", 1),
    ("DRAM wr %", "dram__bytes_write.sum.pct_of_peak_sustained_elapsed", 1),
    ("tensor %", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 1),
    ("occ ach %", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
]

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


def stall_ratios(action) -> list[tuple[str, float]]:
    rows = []
    for name in action.metric_names():
        if name.startswith(STALL_PREFIX) and name.endswith(STALL_SUFFIX):
            reason = name[len(STALL_PREFIX):-len(STALL_SUFFIX)]
            value = safe(action, name)
            if value is not None:
                rows.append((reason, value))
    return sorted(rows, key=lambda r: r[1], reverse=True)


def cmd_summary(args) -> None:
    action = action_of(load(args.report), args.action)
    print(f"kernel: {action.name()}")

    print("\n-- headline --")
    for label, metric, scale in HEADLINE:
        value = safe(action, metric)
        print(f"  {label:<12} {fmt(value * scale if value is not None else None)}")

    print("\n-- launch / occupancy --")
    sms = safe(action, "device__attribute_multiprocessor_count")
    print(f"  grid {safe(action, 'launch__grid_size')} x block {safe(action, 'launch__block_size')}"
          f"  on {sms} SMs; waves/SM {safe(action, 'launch__waves_per_multiprocessor')}")
    print(f"  regs/thread {safe(action, 'launch__registers_per_thread')}"
          f"  smem/block {safe(action, 'launch__shared_mem_per_block')}")
    limits = {k: safe(action, f"launch__occupancy_limit_{k}")
              for k in ("blocks", "registers", "shared_mem", "warps")}
    print(f"  occupancy limits (blocks/SM by resource): {limits}")
    print(f"  theoretical occ % {fmt(safe(action, 'sm__maximum_warps_per_active_cycle_pct'))}"
          f"  achieved % {fmt(safe(action, 'sm__warps_active.avg.pct_of_peak_sustained_active'))}")

    print("\n-- stalls (warps stalled per issue, worst first) --")
    for reason, value in stall_ratios(action)[:8]:
        print(f"  {reason:<22} {value:8.3f}")

    print("\n-- memory pattern --")
    sectors = safe(action, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    requests = safe(action, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    per_req = sectors / requests if sectors and requests else None
    print(f"  global-ld sectors/request {fmt(per_req)}   (LSU path only; TMA traffic not counted)")
    print(f"  L1 hit % {fmt(safe(action, 'l1tex__t_sector_hit_rate.pct'))}"
          f"   L2 hit % {fmt(safe(action, 'lts__t_sector_hit_rate.pct'))}")
    print(f"  store bytes/sector {fmt(safe(action, 'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio'))} (max 32)")
    print(f"  local ld/st (spill) {safe(action, 'smsp__sass_inst_executed_op_local_ld.sum')}"
          f" / {safe(action, 'smsp__sass_inst_executed_op_local_st.sum')}")
    print(f"  threads/inst (divergence, max 32) {fmt(safe(action, 'smsp__thread_inst_executed_per_inst_executed.ratio'))}")


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------
def _literal(text, default=None):
    """Rule payloads arrive as stringified python dicts on ncu 2025.4.1."""
    if isinstance(text, (dict, list)):
        return text
    try:
        return ast.literal_eval(text)
    except Exception:
        return default


def cmd_rules(args) -> None:
    action = action_of(load(args.report), args.action)
    rows = []
    for rule in action.rule_results_as_dicts():
        message = _literal(rule.get("rule_message"), {}) or {}
        speedup = _literal(rule.get("speedup_estimation"), {}) or {}
        rows.append((speedup.get("speedup"), rule.get("rule_identifier"),
                     message.get("title") or "", message.get("message") or ""))
    rows.sort(key=lambda r: r[0] if r[0] is not None else -1, reverse=True)
    for speedup, ident, title, message in rows:
        head = f"est +{speedup:.1f}%" if speedup is not None else "info    "
        print(f"[{head}] {ident}: {title}")
        print(f"    {' '.join(message.split())[:300]}")


# --------------------------------------------------------------------------
# hotspots
# --------------------------------------------------------------------------
PCSAMP_PREFIX = "smsp__pcsamp_warps_issue_stalled_"


def cmd_hotspots(args) -> None:
    action = action_of(load(args.report), args.action)
    per_site: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name in action.metric_names():
        if not name.startswith(PCSAMP_PREFIX):
            continue
        reason = name[len(PCSAMP_PREFIX):]
        metric = action[name]
        if not metric.num_instances() or not metric.has_correlation_ids():
            continue
        correlation = metric.correlation_ids()
        for i in range(metric.num_instances()):
            pc = correlation.as_uint64(i)
            info = action.source_info(pc)
            site = f"{info.file_name()}:{info.line()}" if info else f"pc {pc:#x}"
            per_site[site][reason] += metric.as_uint64(i)
    if not per_site:
        raise SystemExit("no per-PC samples in this report (collect with --set full, "
                         "and build with -lineinfo for source lines)")
    ranked = sorted(per_site.items(), key=lambda kv: sum(kv[1].values()), reverse=True)
    for site, reasons in ranked[: args.top]:
        total = sum(reasons.values())
        worst = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
        detail = ", ".join(f"{r} {v}" for r, v in worst if v)
        print(f"{total:6d}  {site}   [{detail}]")


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
def cmd_compare(args) -> None:
    a = action_of(load(args.report), args.action)
    b = action_of(load(args.other), args.action)
    print(f"{'metric':<44} {'A':>12} {'B':>12} {'delta':>8}")
    metrics = [m for _, m, _ in HEADLINE] + [STALL_PREFIX + r + STALL_SUFFIX
                                             for r, _ in stall_ratios(a)[:5]]
    for metric in metrics:
        va, vb = safe(a, metric), safe(b, metric)
        short = metric.replace("smsp__average_warps_issue_stalled_", "stall:") \
                      .replace("_per_issue_active.ratio", "")
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            print(f"{short:<44} {va:>12.3f} {vb:>12.3f} {100 * (vb - va) / va:>+7.1f}%")
        else:
            print(f"{short:<44} {str(va):>12} {str(vb):>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("summary", cmd_summary), ("rules", cmd_rules),
                     ("hotspots", cmd_hotspots), ("compare", cmd_compare)):
        p = sub.add_parser(name)
        p.add_argument("report")
        if name == "compare":
            p.add_argument("other")
        p.add_argument("--action", type=int, default=0,
                       help="kernel-launch index inside the report (default 0)")
        if name == "hotspots":
            p.add_argument("--top", type=int, default=15)
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
