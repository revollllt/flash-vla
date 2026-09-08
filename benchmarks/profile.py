"""Per-call-site time inside the captured graphs of any Target.

    python -m benchmarks profile --target h100/pi05
    python -m benchmarks profile --target h100/pi05 --plan reference --plan shipped   # A/B
    python -m benchmarks profile --target h100/pi0 --plan reference --trace-dir artifacts/profile

Diagnostic timings from an instrumented replay, never a latency baseline
(`python -m benchmarks latency` owns that). Two runs per segment:

1. An instrumented eager run: every op-table entry is wrapped in a profiler
   annotation and bracketed by two marker kernels, and the segment's kernels
   are issued outside the graph. A launch the profiler intercepted lands
   under its call site by CPU correlation; a launch it did not (a cooperative
   launch through an API without a callback) lands under the call site whose
   markers bracket it in stream order. This yields the ordered sequence
   (call site, kernel name); the markers are stripped.
2. One replay under the profiler: the same launches in the same order with
   their in-graph durations and grid sizes, and no correlation, because the
   CPU side is a single graph launch.

The two sequences must agree name for name, position by position; then the
replay's in-graph time is attributed by position. A mismatch is reported as
such and no time is attributed by guesswork. Kernels a pipeline issues outside
any call site are reported as unattributed.

The report also checks the graph contract the routed backends declare
(kernel names that must or must not appear) and reports, per kernel, the CTA
count against the device's SM count. Legs run in the order given, as in the
latency runner, and per-call-site in-graph time is compared against the first
leg. The runner contains no model, stage or kernel names.

Under a dependent-launch chain a kernel's recorded duration includes the time
it spent launched early and waiting on its predecessor, so attributed
durations overlap and their sum exceeds the segment's wall time; the report
reports interval sums, unions and makespans from the same capture. Separate
unprofiled wall measurements are diagnostic comparisons, not an overlap
subtraction. Shares are of the duration sum, not critical-path latency.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from flash_vla.runtime.engine import segments

from .latency import _env, parse_options
from .metrics import require_cuda
from .targets import PLAN_NAMES, build, resolve
from .timeline import intervals, occurrences, region_occurrences

ANNOTATION = "callsite:"
MARKER_BEGIN = "flash_vla_profile_begin_kernel"
MARKER_END = "flash_vla_profile_end_kernel"
_MARKERS = None


def _markers():
    """Two empty kernels with unmistakable names, compiled once per job.

    They bracket every call site's launches in stream order, which is what
    attributes a launch the profiler has no CPU-side record of. Returns `None`
    when the extension cannot be built; attribution then relies on correlation
    alone and says so.
    """
    global _MARKERS
    if _MARKERS is not None:
        return _MARKERS or None
    try:
        from torch.utils.cpp_extension import load_inline
        module = load_inline(
            name="flash_vla_profile_markers",
            cpp_sources="void begin(); void end();",
            cuda_sources=f"""
            #include <cuda_runtime.h>
            #include <ATen/cuda/CUDAContext.h>
            __global__ void {MARKER_BEGIN}() {{}}
            __global__ void {MARKER_END}() {{}}
            void begin() {{ {MARKER_BEGIN}<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(); }}
            void end() {{ {MARKER_END}<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(); }}
            """,
            functions=["begin", "end"], verbose=False)
        _MARKERS = (module.begin, module.end)
    except Exception as exc:  # no nvcc, no compiler: fall back to correlation only
        print(f"[profile] marker kernels unavailable ({type(exc).__name__}: {exc}); "
              "attributing by correlation only", flush=True)
        _MARKERS = ()
    return _MARKERS or None


def is_copy(name: str) -> bool:
    lowered = name.lower()
    return ("memcpy" in lowered or "memset" in lowered or "copy_" in lowered
            or ("elementwise" in lowered and "copy" in lowered))


def _annotate(name: str, fn: Callable) -> Callable:
    markers = _markers()

    def wrapped(*args, **kwargs):
        with record_function(ANNOTATION + name):
            if markers:
                markers[0]()
            try:
                return fn(*args, **kwargs)
            finally:
                if markers:
                    markers[1]()
    return wrapped


def _trace_events(prof, keep: Path | None = None) -> list[dict]:
    """The profiler's Chrome trace events; a profiler exports once, so the
    export goes to `keep` when a path is wanted and to a temporary file otherwise."""
    if keep is not None:
        keep.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(keep))
        with open(keep) as f:
            return json.load(f).get("traceEvents", [])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            return json.load(f).get("traceEvents", [])


def _gpu_events(events: list[dict]) -> list[dict]:
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    return sorted(gpu, key=lambda e: e.get("ts", 0.0))


def eager_sequence(engine, segment: str,
                   trace_path: Path | None = None) -> list[dict[str, Any]]:
    """(call_site, name) per GPU launch of one instrumented eager run, in issue order."""
    with engine.instrument(_annotate):
        engine.run_eager(segment)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            engine.run_eager(segment)
            torch.cuda.synchronize()
    events = _trace_events(prof, trace_path)
    annotations = [e for e in events if e.get("cat") == "user_annotation"
                   and str(e.get("name", "")).startswith(ANNOTATION)]
    for invocation_id, annotation in enumerate(sorted(annotations, key=lambda e: e["ts"])):
        annotation["invocation_id"] = invocation_id
    # A launch reaches the GPU through the runtime API (torch, cuBLAS, the CUDA
    # extensions) or the driver API (TileLang's compiled host stubs); both
    # carry the correlation id the kernel event refers back to.
    runtime = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                runtime[corr] = e
    cats = Counter(e.get("cat") for e in events)
    by_tid: dict[Any, list[dict]] = defaultdict(list)
    for a in annotations:
        by_tid[a.get("tid")].append(a)
    # A launch CUPTI has no API callback for (a cooperative or attribute
    # launch through an API it does not intercept) still carries the external
    # correlation id of the CPU op active when it was issued; the annotation
    # itself, or a CPU op nested in it, carries the same id.
    by_external: dict[Any, dict] = {}
    for e in events:
        ext = (e.get("args") or {}).get("External id")
        if ext is not None and e.get("cat") in ("user_annotation", "cpu_op"):
            by_external.setdefault(ext, e)

    def owner(launch: dict) -> dict | None:
        ts = launch.get("ts", 0.0)
        best = None
        for a in by_tid.get(launch.get("tid"), ()):
            if a["ts"] <= ts <= a["ts"] + a.get("dur", 0.0):
                if best is None or a.get("dur", 0.0) < best.get("dur", 0.0):
                    best = a
        return best

    sequence = []
    brackets: dict[Any, dict | None] = {}
    markers_seen = 0
    disagreements = 0
    for e in _gpu_events(events):
        args = e.get("args") or {}
        name = e.get("name") or ""
        launch = runtime.get(args.get("correlation"))
        via = "correlation" if launch is not None else None
        if launch is None:
            launch = by_external.get(args.get("External id"))
            via = "external" if launch is not None else None
        annotation = owner(launch) if launch else None
        stream = args.get("stream")
        bracket = brackets.get(stream)
        if name.startswith(MARKER_BEGIN):
            brackets[stream] = annotation
            markers_seen += 1
            continue
        if name.startswith(MARKER_END):
            brackets[stream] = None
            markers_seen += 1
            continue
        if annotation is None and bracket is not None:
            annotation, via = bracket, "bracket"
        elif annotation is not None and bracket is not None and annotation != bracket:
            disagreements += 1
        sequence.append({"call_site": annotation["name"][len(ANNOTATION):] if annotation else None,
                         "invocation_id": annotation["invocation_id"] if annotation else None,
                         "stream": stream, "name": name, "cat": e.get("cat"),
                         "launch_seen": launch is not None, "via": via})
    sequence_cats = {"trace_categories": dict(cats),
                     "launch_seen": sum(1 for s in sequence if s["launch_seen"]),
                     "via_external": sum(1 for s in sequence if s["via"] == "external"),
                     "via_bracket": sum(1 for s in sequence if s["via"] == "bracket"),
                     "markers": markers_seen, "bracket_disagreements": disagreements}
    return sequence, sequence_cats


def replay_sequence(engine, segment: str, trace_path: Path | None = None) -> list[dict[str, Any]]:
    """(name, dur_us, ctas) per GPU launch of one replay, in issue order."""
    engine.replay(segment)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        engine.replay(segment)
        torch.cuda.synchronize()
    out = []
    for e in _gpu_events(_trace_events(prof, trace_path)):
        grid = (e.get("args") or {}).get("grid")
        ctas = int(grid[0]) * int(grid[1]) * int(grid[2]) if grid else None
        out.append({"name": e.get("name"), "cat": e.get("cat"),
                    "dur_us": float(e.get("dur", 0.0)), "ctas": ctas,
                    "start_us": e["ts"], "end_us": e["ts"] + e.get("dur", 0.0),
                    "stream": (e.get("args") or {}).get("stream"),
                    "graph_node_id": (e.get("args") or {}).get("graph node id")})
    return out


def _same_launch(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same kernel, or a copy on both sides: a graph runs a memcpy node as a kernel
    (`memcpy32_post`) where eager records a memcpy activity."""
    if a["name"] == b["name"]:
        return True
    return is_copy(a["name"] or "") and is_copy(b["name"] or "")


def attribute(engine, segment: str, sm_count: int, trace_path: Path | None = None,
              eager_trace_path: Path | None = None) -> dict[str, Any]:
    """Per-call-site in-graph time of one segment, by positional match of the two runs."""
    eager, eager_info = eager_sequence(engine, segment, eager_trace_path)
    replay = replay_sequence(engine, segment, trace_path)
    wall_us = _replay_wall_us(engine, segment)
    same_length = len(eager) == len(replay)
    mismatches = [{"position": i, "eager": a["name"], "replay": b["name"]}
                  for i, (a, b) in enumerate(zip(eager, replay)) if not _same_launch(a, b)]
    single_stream = len({e["stream"] for e in eager}) <= 1 and len({e["stream"] for e in replay}) <= 1
    valid = same_length and not mismatches and single_stream and not eager_info["bracket_disagreements"]
    for index, event in enumerate(replay):
        event["call_site"] = eager[index]["call_site"] if valid else None
        event["invocation_id"] = eager[index]["invocation_id"] if valid else None
    per_site: dict[str, dict[str, Any]] = {}
    unattributed = {"launches": 0, "dur_us": 0.0, "names": set()}
    if valid:
        for a, b in zip(eager, replay):
            site = a["call_site"]
            if site is None:
                unattributed["launches"] += 1
                unattributed["dur_us"] += b["dur_us"]
                unattributed["names"].add(b["name"])
                continue
            row = per_site.setdefault(site, {"launches": 0, "dur_us": 0.0, "names": set(),
                                             "ctas": [], "copy_us": 0.0})
            row["launches"] += 1
            row["dur_us"] += b["dur_us"]
            row["names"].add(b["name"])
            if b["ctas"] is not None:
                row["ctas"].append(b["ctas"])
            if is_copy(b["name"] or ""):
                row["copy_us"] += b["dur_us"]
    if not valid:
        unattributed = {"launches": len(replay), "dur_us": sum(e["dur_us"] for e in replay),
                        "names": {e["name"] for e in replay}}
    for site, row in per_site.items():
        row.update(intervals([e for e in replay if e["call_site"] == site]))
        row["occurrences"] = occurrences([e for e in replay if e["call_site"] == site])
        row["names"] = sorted(row["names"])
        ctas = row.pop("ctas")
        row["ctas_min"] = min(ctas) if ctas else None
        row["under_one_wave"] = sum(1 for c in ctas if c < sm_count)
    unattributed["names"] = sorted(unattributed["names"])
    total = sum(b["dur_us"] for b in replay)
    for row in per_site.values():
        row["share"] = row["dur_us"] / total if total else 0.0
    return {"valid": valid, "launches": len(replay), "eager_launches": len(eager),
            "mismatched_positions": mismatches[:10], "eager_trace": eager_info,
            "total_us": total, "wall_us": wall_us,
            "unprofiled_region_wall_us": wall_us, **intervals(replay),
            "events": replay, "occurrences": occurrences(replay),
            "mapping": "single_stream_positional" if valid else "unattributed",
            "regions": [{"call_sites": sorted(group), **region_occurrences(replay, group)}
                        for group in engine.atomic_groups],
            "copy_us": sum(b["dur_us"] for b in replay if is_copy(b["name"] or "")),
            "call_sites": dict(sorted(per_site.items(), key=lambda kv: -kv[1]["dur_us"])),
            "unattributed": unattributed,
            "kernel_names": sorted({b["name"] for b in replay if b["name"]})}


def _replay_wall_us(engine, segment: str, reps: int = 5) -> float:
    """Event-timed minimum of an unprofiled replay, the segment's real duration."""
    best = None
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        engine.replay(segment)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        best = ms if best is None else min(best, ms)
    return (best or 0.0) * 1e3


def check_contract(contract: dict[str, list[str]], names: set[str]) -> dict[str, Any]:
    violations = []
    for pattern in contract.get("forbid", ()):
        hits = sorted(n for n in names if pattern in n)
        if hits:
            violations.append({"forbid": pattern, "found": hits})
    for pattern in contract.get("require_one", ()):
        hits = sorted(n for n in names if pattern in n)
        if len(hits) != 1:
            violations.append({"require_one": pattern, "found": hits})
    return {"passed": not violations, "violations": violations, "contract": contract}


def run(target: str, plans: list[str | None], seed: int = 0, trace_dir: str | None = None,
        leg_options: list[dict[str, Any]] | None = None, eager_trace_dir: str | None = None,
        **overrides) -> dict[str, Any]:
    """Attribute one replay per segment per leg, check contracts, compare legs."""
    require_cuda()
    torch.cuda.init()
    target = resolve(target)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    plans = [plan or "shipped" for plan in plans]
    leg_options = list(leg_options or [{}] * len(plans))
    legs = []
    for index, (plan, options) in enumerate(zip(plans, leg_options)):
        print(f"== leg {index}: {target} plan={plan} options={options}", flush=True)
        engine = build(target, plan, seed=seed, **{**overrides, **options})
        inputs = engine.sample_inputs(seed)
        engine.forward(**inputs)
        torch.cuda.synchronize()
        seg_reports = {}
        names: set[str] = set()
        for name in segments(engine):
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{index}_{plan}_{name}")
            path = Path(trace_dir) / f"{slug}.json" if trace_dir else None
            eager_path = Path(eager_trace_dir) / f"{slug}_eager.json" if eager_trace_dir else None
            seg_reports[name] = attribute(engine, name, sm_count, path, eager_path)
            names |= set(seg_reports[name]["kernel_names"])
        legs.append({"leg": index, "plan": plan, "options": dict(options),
                     "identity": engine.identity.as_dict(), "segments": seg_reports,
                     "contract": check_contract(engine.graph_contract, names)})
        del engine
        torch.cuda.empty_cache()
    report = {"identity": legs[0]["identity"], "env": _env(), "sm_count": sm_count,
              "config": {"seed": seed, "plans": plans, "trace_dir": trace_dir},
              "legs": legs, "deltas": _deltas(legs) if len(legs) > 1 else None}
    return report


def _deltas(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-call-site in-graph time of every leg against the first, per segment."""
    out = []
    for leg in legs[1:]:
        rows = {}
        for seg, rep in leg["segments"].items():
            base = legs[0]["segments"].get(seg, {}).get("call_sites", {})
            for site, row in rep["call_sites"].items():
                ref = base.get(site)
                rows[f"{seg}/{site}"] = {"dur_us": row["dur_us"],
                                         "delta_us": row["dur_us"] - ref["dur_us"] if ref else None}
        out.append({"leg": leg["leg"], "plan": leg["plan"], "options": leg["options"],
                    "call_sites": rows})
    return out


def render(report: dict[str, Any], top: int = 12) -> str:
    lines = []
    for leg in report["legs"]:
        lines.append(f"== leg {leg['leg']} plan={leg['plan']} options={leg['options']} "
                     f"contract={'ok' if leg['contract']['passed'] else leg['contract']['violations']}")
        for seg, rep in leg["segments"].items():
            status = "" if rep["valid"] else (
                f"  ATTRIBUTION INVALID (eager {rep['eager_launches']} vs replay "
                f"{rep['launches']} launches, first mismatches {rep['mismatched_positions']})")
            lines.append(f"  -- {seg}: wall {rep['wall_us'] / 1e3:.3f} ms, attributed "
                         f"{rep['total_us'] / 1e3:.3f} ms, trace union "
                         f"{rep['interval_union_us'] / 1e3:.3f} ms, "
                         f"{rep['launches']} launches, copies {rep['copy_us'] / 1e3:.3f} ms{status}")
            for site, row in list(rep["call_sites"].items())[:top]:
                lines.append(f"     {row['dur_us'] / 1e3:8.3f} ms {row['share'] * 100:5.1f}%  "
                             f"x{row['launches']:<5d} {site:32s} ctas_min={row['ctas_min']} "
                             f"under_wave={row['under_one_wave']}")
            if rep["unattributed"]["launches"]:
                u = rep["unattributed"]
                lines.append(f"     {u['dur_us'] / 1e3:8.3f} ms        x{u['launches']:<5d} "
                             f"(unattributed) {u['names'][:3]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--plan", action="append", default=None,
                        help=f"one of {PLAN_NAMES}, a JSON object or a lab/plans/*.json path; "
                             "repeat for A/B (default: shipped)")
    parser.add_argument("--option", action="append", default=[],
                        help="target-local option key=value applied to every leg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--trace-dir", default=os.environ.get("GPU_PROFILE_OUTPUT_DIR"),
                        help="write one Chrome trace per segment per leg here")
    parser.add_argument("--eager-trace-dir", default=None,
                        help="also write the instrumented eager run's trace, for attribution triage")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in (("steps", args.steps), ("layers", args.layers))
                 if v is not None}
    overrides.update(parse_options(args.option))
    report = run(args.target, args.plan or [None], seed=args.seed, trace_dir=args.trace_dir,
                 eager_trace_dir=args.eager_trace_dir, **overrides)
    print(render(report, args.top))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    ok = all(leg["contract"]["passed"] and all(s["valid"] for s in leg["segments"].values())
             for leg in report["legs"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
