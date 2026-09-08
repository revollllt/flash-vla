#!/usr/bin/env python3
"""Convert the starter's completed software ranges to Perfetto Chrome Trace JSON.
Standard library only. No GPU required. Not an IKET JSON parser or a hardware
utilization calculator. Raw integer timestamps are preserved in input artifacts.
"""
from __future__ import annotations
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SEMANTICS = {"software_scope", "issue_scope", "wait_scope", "completion_observed_window"}


def integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("reversed interval")
        if end == start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def union_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in merge_intervals(intervals))


def overlap_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    left, right = merge_intervals(a), merge_intervals(b)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def convert(raw: dict[str, Any], *, allow_partial: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported raw trace schema")
    if raw.get("clock_domain") != "gpu-local" or raw.get("timer_unit") != "ns":
        raise ValueError("this prototype supports one GPU-local nanosecond clock domain")
    if type(raw.get("synthetic")) is not bool:
        raise ValueError("raw trace must declare synthetic: true or false")
    dropped = integer(raw.get("dropped_records"), "dropped_records")
    if dropped and not allow_partial:
        raise ValueError("dropped records; use --allow-partial only for partial visualization")
    ranges = raw.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("ranges must be a nonempty list")
    seen: set[tuple] = set()
    devices: set[str] = set()
    warnings: list[str] = []
    if dropped:
        warnings.append("partial trace: full-coverage duration/overlap conclusions are disabled")
    checked: list[dict[str, Any]] = []
    for source in ranges:
        if not isinstance(source, dict):
            raise ValueError("range must be an object")
        r = dict(source)
        for field in ("device_id", "role", "stage"):
            if not isinstance(r.get(field), str) or not r[field]:
                raise ValueError(f"{field} must be a nonempty string")
        if r.get("semantic") not in SEMANTICS:
            raise ValueError("unknown semantic; hardware-active inference is not supported")
        cta = r.get("cta")
        if not isinstance(cta, list) or len(cta) != 3:
            raise ValueError("cta must have three integer coordinates")
        for v in cta:
            integer(v, "cta coordinate")
        for field in ("launch_id", "replay_id", "warp_id", "token", "sm_begin", "sm_end", "start_ns", "end_ns"):
            r[field] = integer(r.get(field), field)
        if r["end_ns"] < r["start_ns"]:
            raise ValueError("end precedes start")
        key = (r["device_id"], r["launch_id"], r["replay_id"], tuple(cta),
               r["warp_id"], r["role"], r["stage"], r["token"], integer(r.get("generation", 0), "generation"))
        if key in seen:
            raise ValueError("duplicate dynamic range identity")
        seen.add(key)
        devices.add(r["device_id"])
        checked.append(r)
    if len(devices) != 1:
        raise ValueError("cross-device traces need explicit clock calibration; split this input")
    origin = min([r["start_ns"] for r in checked] + [integer(m["timestamp_ns"], "marker timestamp") for m in raw.get("markers", [])])
    if max([r["end_ns"] for r in checked] + [m["timestamp_ns"] for m in raw.get("markers", [])]) - origin >= 2**53:
        raise ValueError("time window too large for this JSON exporter; split the capture")
    events: list[dict[str, Any]] = []
    processes: dict[tuple, int] = {}
    track_slots: dict[tuple, list[list[int]]] = defaultdict(list)
    stage_stats: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    role_intervals: dict[tuple, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    next_tid = 1
    migrations = 0
    for r in sorted(checked, key=lambda x: (x["start_ns"], -x["end_ns"])):
        stable_sm = r["sm_begin"] == r["sm_end"]
        sm = r["sm_begin"] if stable_sm else "unknown-migration"
        if not stable_sm:
            migrations += 1
        pkey = (r["device_id"], sm)
        if pkey not in processes:
            pid = len(processes) + 1
            processes[pkey] = pid
            events.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                           "args": {"name": f"{r['device_id']} / SM {sm} (software observations)"}})
        pid = processes[pkey]
        logical = (pid, r["launch_id"], r["replay_id"], tuple(r["cta"]), r["warp_id"], r["role"])
        # Crossing or nested complete ranges get separate subtracks; X events on
        # one physical track must not silently create an invalid nesting order.
        slot = next((s for s in track_slots[logical] if s[0] <= r["start_ns"]), None)
        if slot is None:
            slot = [0, next_tid]
            next_tid += 1
            track_slots[logical].append(slot)
            events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": slot[1],
                           "args": {"name": f"launch {r['launch_id']} replay {r['replay_id']} "
                                    f"CTA {r['cta']} warp {r['warp_id']} {r['role']} "
                                    f"lane {len(track_slots[logical])-1}"}})
        slot[0] = r["end_ns"]
        args = {k: v for k, v in r.items() if k not in ("start_ns", "end_ns")}
        args.update({"synthetic": raw["synthetic"], "hardware_active": "not_measured",
                     "timestamp_meaning": "software_observation"})
        event = {"ph": "X", "name": r["stage"], "cat": r["semantic"],
                 "pid": pid, "tid": slot[1], "ts": (r["start_ns"] - origin) / 1000.0,
                 "dur": (r["end_ns"] - r["start_ns"]) / 1000.0, "args": args}
        events.append(event)
        if stable_sm:
            group = (r["device_id"], sm, r["launch_id"], r["replay_id"])
            stage_stats[group + (r["role"], r["stage"], r["semantic"])].append((r["start_ns"], r["end_ns"]))
            # Different semantic kinds must never be combined as engine activity.
            if r["semantic"] == "software_scope":
                role_intervals[group][r["role"]].append((r["start_ns"], r["end_ns"]))
    if migrations:
        warnings.append(f"{migrations} ranges have different endpoint SM IDs; excluded from per-SM summaries")
    stages = []
    for key, intervals in sorted(stage_stats.items()):
        device, sm, launch, replay, role, stage, semantic = key
        stages.append({"device_id": device, "sm_id": sm, "launch_id": launch, "replay_id": replay,
                       "role": role, "stage": stage, "semantic": semantic, "count": len(intervals),
                       "sum_duration_ns": sum(e-s for s,e in intervals), "interval_union_ns": union_ns(intervals)})
    overlaps = []
    for key, roles in sorted(role_intervals.items()):
        for a, b in itertools.combinations(sorted(roles), 2):
            both = overlap_ns(roles[a], roles[b])
            whole = union_ns(roles[a] + roles[b])
            overlaps.append({"device_id": key[0], "sm_id": key[1], "launch_id": key[2], "replay_id": key[3],
                             "roles": [a, b], "observed_scope_overlap_ns": both,
                             "observed_scope_union_ns": whole,
                             "overlap_over_scope_union": both/whole if whole else None,
                             "interpretation": "software-scope intersection, NOT hardware pipe utilization"})
    if raw.get("markers"):
        from .query import validate_tokens
        pairing = validate_tokens(raw)
        marker_pid = len(processes) + 1
        events.append(dict(ph="M", name="process_name", pid=marker_pid,
                           args={"name": "async software observations; SM not inferred"}))
        marker_tracks = {}
        for marker in raw["markers"]:
            if marker["device_id"] not in devices:
                raise ValueError("marker device differs from range clock domain")
            track = (marker["launch_id"], marker["replay_id"], tuple(marker["cta"]), marker["role"])
            if track not in marker_tracks:
                marker_tracks[track] = len(marker_tracks) + 1
                events.append(dict(ph="M", name="thread_name", pid=marker_pid, tid=marker_tracks[track],
                                   args={"name": str(track)}))
            events.append(dict(ph="i", s="t", name=marker["operation"] + ":" + marker["kind"],
                               pid=marker_pid, tid=marker_tracks[track],
                               ts=(marker["timestamp_ns"]-origin)/1000.0, args=marker))
        if pairing["incomplete"]:
            warnings.append("incomplete async operation pairs; completion is unknown")
    metadata = {k: v for k, v in raw.items() if k != "ranges"}
    metadata.update({"origin_ns": str(origin), "warnings": warnings, "production_gate_eligible": False})
    trace = {"displayTimeUnit": "ns", "traceEvents": events, "metadata": metadata}
    summary = {"schema_version": 1, "synthetic": raw["synthetic"], "range_count": len(checked),
               "dropped_records": dropped, "warnings": warnings, "production_gate_eligible": False,
               "counter_evidence": None, "coverage": raw.get("coverage", "unknown"),
               "timer_resolution_ns": raw.get("timer_resolution_ns"),
               "scope_statistics": stages if not dropped else [],
               "observed_scope_overlaps": overlaps if not dropped else [],
               "unobserved_intervals": "unknown, not idle",
               "cross_run_or_cross_device_alignment": "not_performed"}
    return trace, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        trace, summary = convert(raw, allow_partial=args.allow_partial)
        args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        if args.summary:
            args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"export error: {error}\n")

if __name__ == "__main__":
    main()
