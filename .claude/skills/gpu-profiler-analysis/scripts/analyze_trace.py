#!/usr/bin/env python3
"""Analyze a Chrome trace emitted by torch.profiler or a compatible producer."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


COPY_TOKENS = ("memcpy", "memset", "copy_", "copy ", "cuda memcpy")
STAGE_NAMES = ("vision", "prefix", "encoder", "decoder", "full")


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.name.endswith(".gz") else path.open(
        "r", encoding="utf-8"
    )


def load_json(path: Path) -> Any:
    with _open_text(path) as handle:
        return json.load(handle)


def discover_traces(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    ignored = {"manifest.json", "summary.json", "metadata.json"}
    paths = [
        path
        for path in input_path.rglob("*")
        if path.is_file()
        and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
        and path.name not in ignored
        and not path.name.endswith(".report.json")
    ]
    preferred = [path for path in paths if "trace" in path.name.lower()]
    return sorted(preferred or paths)


def load_manifest(input_path: Path) -> dict[str, Any]:
    directory = input_path if input_path.is_dir() else input_path.parent
    manifest = directory / "manifest.json"
    if not manifest.exists():
        return {}
    value = load_json(manifest)
    return value if isinstance(value, dict) else {}


def trace_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("traceEvents", payload.get("events", []))
    else:
        values = []
    return [event for event in values if isinstance(event, dict)]


def event_name(event: dict[str, Any]) -> str:
    return str(event.get("name") or event.get("cat") or "<unnamed>")


def event_args(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args")
    return args if isinstance(args, dict) else {}


def duration_us(event: dict[str, Any]) -> float:
    try:
        return max(float(event.get("dur", 0.0)), 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_kernel(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    name = event_name(event).lower()
    if "kernel" in category:
        return True
    if "cuda" in category and not any(token in category for token in ("runtime", "api")):
        return not any(token in name for token in ("launch", "event", "synchronize"))
    return False


def is_cpu_annotation(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    return any(token in category for token in ("cpu", "python", "user_annotation", "operator"))


def is_copy(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in COPY_TOKENS)


def canonical_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def stage_for(event: dict[str, Any], manifest: dict[str, Any]) -> str:
    args = event_args(event)
    for key in ("stage", "stage_name", "workload_stage"):
        if args.get(key):
            return str(args[key])
    workload = manifest.get("workload")
    if isinstance(workload, dict) and workload.get("stage"):
        return str(workload["stage"])
    lowered = event_name(event).lower()
    for stage in STAGE_NAMES:
        if stage in lowered:
            return stage
    return "all"


def stream_for(event: dict[str, Any]) -> str:
    args = event_args(event)
    for key in ("stream", "stream_id", "streamId"):
        if args.get(key) is not None:
            return str(args[key])
    pid = event.get("pid", "?")
    tid = event.get("tid", "?")
    return f"{pid}:{tid}"


def source_for(event: dict[str, Any]) -> dict[str, Any] | None:
    args = event_args(event)
    scope = args.get("python_scope") or args.get("python") or args.get("scope")
    file_name = args.get("source_file") or args.get("file")
    line = args.get("source_line") or args.get("line")
    if scope or file_name or line:
        return {"scope": scope, "file": file_name, "line": line}
    return None


def union_us(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def analyze(paths: list[Path], manifest: dict[str, Any]) -> dict[str, Any]:
    kernels: list[dict[str, Any]] = []
    cpu_count = 0
    event_count = 0
    for path in paths:
        for event in trace_events(load_json(path)):
            event_count += 1
            dur = duration_us(event)
            if dur <= 0 or event.get("ph", "X") not in ("X", "B"):
                continue
            if is_kernel(event):
                source = source_for(event)
                kernels.append(
                    {
                        "name": canonical_name(event_name(event)),
                        "duration_us": dur,
                        "stage": stage_for(event, manifest),
                        "stream": stream_for(event),
                        "ts": float(event.get("ts", 0.0) or 0.0),
                        "source": source,
                        "copy": is_copy(event_name(event)),
                    }
                )
            elif is_cpu_annotation(event):
                cpu_count += 1

    total_us = sum(item["duration_us"] for item in kernels)
    intervals = [(item["ts"], item["ts"] + item["duration_us"]) for item in kernels]
    span_us = union_us(intervals)
    by_name: dict[str, dict[str, Any]] = {}
    by_stage: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    by_stream: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    source_count = 0
    for item in kernels:
        row = by_name.setdefault(
            item["name"],
            {"kernel": item["name"], "calls": 0, "total_us": 0.0, "stages": {}, "copy": item["copy"]},
        )
        row["calls"] += 1
        row["total_us"] += item["duration_us"]
        row["stages"][item["stage"]] = row["stages"].get(item["stage"], 0) + 1
        stage = by_stage[item["stage"]]
        stage["calls"] += 1
        stage["total_us"] += item["duration_us"]
        stream = by_stream[item["stream"]]
        stream["calls"] += 1
        stream["total_us"] += item["duration_us"]
        source_count += int(item["source"] is not None)

    for row in by_name.values():
        row["share_pct"] = (row["total_us"] / total_us * 100.0) if total_us else 0.0
        row["avg_us"] = row["total_us"] / row["calls"] if row["calls"] else 0.0
    mapping_status = "mapped" if source_count == len(kernels) and kernels else "partial"
    if not kernels or (not source_count and not cpu_count):
        mapping_status = "unavailable"
    return {
        "input_traces": [str(path) for path in paths],
        "event_count": event_count,
        "kernel_count": len(kernels),
        "cpu_annotation_count": cpu_count,
        "total_kernel_us": total_us,
        "kernel_span_us": span_us,
        "estimated_overlap_us": max(total_us - span_us, 0.0),
        "estimated_gpu_busy_ratio": (total_us / span_us) if span_us else 0.0,
        "copy_us": sum(item["duration_us"] for item in kernels if item["copy"]),
        "mapping_status": mapping_status,
        "kernels": sorted(by_name.values(), key=lambda row: row["total_us"], reverse=True),
        "stages": dict(sorted(by_stage.items(), key=lambda pair: -pair[1]["total_us"])),
        "streams": dict(sorted(by_stream.items(), key=lambda pair: -pair[1]["total_us"])),
        "manifest": manifest,
    }


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(summary: dict[str, Any], top: int, min_share: float) -> str:
    lines = ["# GPU Profiler Report", "", f"Source mapping: `{summary['mapping_status']}`", ""]
    lines += [
        "## Summary",
        "",
        f"- traces: {len(summary['input_traces'])}",
        f"- kernels: {summary['kernel_count']}",
        f"- total kernel self-time: {summary['total_kernel_us'] / 1000.0:.3f} ms",
        f"- kernel timeline span: {summary['kernel_span_us'] / 1000.0:.3f} ms",
        f"- estimated overlap: {summary['estimated_overlap_us'] / 1000.0:.3f} ms",
        f"- copy/memset time: {summary['copy_us'] / 1000.0:.3f} ms",
        "",
        "## Kernel table",
        "",
        "| Kernel | Calls | Total ms | Share | Avg us | Stage(s) | Kind |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    visible = [row for row in summary["kernels"] if row["share_pct"] >= min_share][:top]
    if not visible:
        lines.append("| No kernels met the reporting threshold | - | - | - | - | - | - |")
    for row in visible:
        kind = "copy" if row["copy"] else "compute"
        stages = ", ".join(sorted(row["stages"]))
        lines.append(
            f"| {md_cell(row['kernel'])} | {row['calls']} | {row['total_us'] / 1000.0:.3f} | "
            f"{row['share_pct']:.1f}% | {row['avg_us']:.2f} | {md_cell(stages)} | {kind} |"
        )
    lines += ["", "## Stage table", "", "| Stage | Calls | Total ms |", "| --- | ---: | ---: |"]
    for stage, row in summary["stages"].items():
        lines.append(f"| {md_cell(stage)} | {row['calls']} | {row['total_us'] / 1000.0:.3f} |")
    lines += [
        "",
        "## Perfetto",
        "",
        "Open the Chrome trace artifact at https://ui.perfetto.dev. This report is a summary; use the trace for event-level conclusions.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Chrome trace file or directory")
    parser.add_argument("--output-dir", type=Path, help="Directory for summary.json and report.md")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-share", type=float, default=1.0)
    args = parser.parse_args(argv)

    paths = discover_traces(args.input)
    if not paths:
        parser.error(f"no .json or .json.gz trace found under {args.input}")
    output_dir = args.output_dir or (args.input if args.input.is_dir() else args.input.parent / "profile-analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(paths, load_manifest(args.input))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(
        render_report(summary, max(args.top, 0), max(args.min_share, 0.0)), encoding="utf-8"
    )
    print(json.dumps({"summary": str(output_dir / "summary.json"), "report": str(output_dir / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
