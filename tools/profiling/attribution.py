"""Per-leg evidence that says why a forward was late, not only that one was.

`benchmarks/latency.py` reports `min`, `median` and `p99` of every metric the
acceptance registry names. A tail is then a number with no cause: Pi0.5's
chunk latency has read 2.5 to 17.3 ms above its own `min` in one or two
forwards of a hundred, on either plan, on six different nodes, and nothing in
the report says which forward, when, or what else happened at that moment.

This module records, per leg, what turns such a sample into an attribution:

  loops         per timed loop, every per-forward sample with its offset from
                the loop's start, and per-forward deltas of the process's
                context switches and page faults (`resource.getrusage`), plus
                the loop's voluntary / nonvoluntary switch delta read from
                `/proc/self/status`
  gc            every collection Python's cyclic collector ran while the leg
                was measured, with generation, start and duration
                (`gc.callbacks`)
  gpu           a 10 Hz sample of SM clock, clock event reasons, utilization,
                temperature and power, and the device's other compute
                processes, both timestamped by `nvidia-smi` itself
  late          every sample above `min + late_ms`, with the switches, faults,
                collections and GPU samples that fall inside it

The instrument may not create the artifact it looks for. Two rules follow.

*No sampling thread.* A Python thread that wakes at 10 Hz hands the GIL over at
`sys.getswitchinterval()`, 5 ms by default, which is the magnitude under
investigation. `nvidia-smi` therefore runs as a subprocess with `--loop-ms`
writing to a file, and the file is parsed after the timed loops finish. Loops
record both a `perf_counter` origin and a wall clock so the two clocks align.

*Nothing added inside the timed region.* The per-sample counters are read after
the timed region closes, so sample *i*'s delta covers forward *i* plus the loop
bookkeeping after it. `getrusage` is one syscall and one object;
`/proc/self/status` is parsed twice per loop, never per sample, because parsing
it per forward would allocate dozens of objects and feed the collector this
module is trying to observe. Sample lists are preallocated and assigned by
index for the same reason.

Every instrument fails into a recorded error string. Missing evidence degrades
a run; it never fails one.
"""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import time
from typing import Any

try:  # POSIX only, and the whole point of the module; recorded if absent
    import resource
except ImportError:  # pragma: no cover - not this platform
    resource = None  # type: ignore[assignment]

#: Sampling period of the GPU sampler, in milliseconds.
GPU_SAMPLE_MS = 100
#: `nvidia-smi --query-gpu` fields, in order; `timestamp` first so the sampler's
#: own clock is on every row.
GPU_FIELDS = ("timestamp", "clocks.sm", "clocks.mem", "clocks_event_reasons.active",
              "utilization.gpu", "utilization.memory", "temperature.gpu", "power.draw",
              "memory.used")
#: `nvidia-smi --query-compute-apps` fields: who else is on this device.
APP_FIELDS = ("timestamp", "pid", "process_name", "used_gpu_memory")
#: The process counters read after every sample, in `getrusage` order.
COUNTERS = ("nvcsw", "nivcsw", "minflt", "majflt")
#: A sample this far above its loop's `min` is late enough to explain. It is
#: the registry's own deployment bound, so this module introduces no new number.
LATE_MS = 0.5


def _rusage() -> tuple[int, int, int, int]:
    """(voluntary, involuntary) context switches and (minor, major) faults, so far."""
    if resource is None:
        return (0, 0, 0, 0)
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_nvcsw, r.ru_nivcsw, r.ru_minflt, r.ru_majflt)


def _proc_status() -> dict[str, int]:
    """`voluntary_ctxt_switches` and `nonvoluntary_ctxt_switches` of this process."""
    out: dict[str, int] = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
                    out[key] = int(value.strip())
    except (OSError, ValueError):
        pass
    return out


class LoopTrace:
    """One timed loop's per-sample record, preallocated at construction.

    `enter`, `start` and `mark` are all the loop itself calls: `enter` once
    before the first repetition, `start` with the `perf_counter` the timed
    region opened at, `mark` once after it closed. Nothing here runs between
    `perf_counter()` and `torch.cuda.synchronize()`.
    """

    __slots__ = ("name", "reps", "t0_perf", "t0_unix", "starts", "status_before",
                 "status_after", "_ru", "_ru0")

    def __init__(self, name: str, reps: int) -> None:
        self.name = name
        self.reps = max(int(reps), 0)
        self.t0_perf = 0.0
        self.t0_unix = 0.0
        self.starts = [0.0] * self.reps          # ms from this loop's origin
        self._ru = [(0, 0, 0, 0)] * self.reps    # cumulative counters after sample i
        self._ru0 = (0, 0, 0, 0)
        self.status_before: dict[str, int] = {}
        self.status_after: dict[str, int] = {}

    def enter(self) -> None:
        self.status_before = _proc_status()
        self._ru0 = _rusage()
        self.t0_unix = time.time()
        self.t0_perf = time.perf_counter()

    def start(self, index: int, when: float) -> None:
        """Record where sample `index` began, on this loop's own clock."""
        if 0 <= index < self.reps:
            self.starts[index] = (when - self.t0_perf) * 1e3

    def mark(self, index: int) -> None:
        """Read the process counters just after sample `index`'s timed region."""
        if 0 <= index < self.reps:
            self._ru[index] = _rusage()

    def leave(self) -> None:
        self.status_after = _proc_status()

    def as_dict(self, samples: list[float]) -> dict[str, Any]:
        n = min(len(samples), self.reps)
        per: dict[str, list[int]] = {k: [0] * n for k in COUNTERS}
        previous = self._ru0
        for i in range(n):
            current = self._ru[i]
            for k, before, after in zip(COUNTERS, previous, current):
                per[k][i] = after - before
            previous = current
        delta = {k: self.status_after.get(k, 0) - value
                 for k, value in self.status_before.items()}
        return {
            "reps": n,
            "started_unix": self.t0_unix,
            "started_perf": self.t0_perf,
            "wall_ms": round(self.starts[n - 1] + samples[n - 1], 4) if n else 0.0,
            "samples_ms": [round(s, 6) for s in samples[:n]],
            "start_ms": [round(s, 4) for s in self.starts[:n]],
            "ctxt_switch_delta": delta,
            "per_sample": per,
            "totals": {k: sum(v) for k, v in per.items()},
        }


class _GpuSampler:
    """`nvidia-smi` as a detached subprocess writing timestamped rows to a file.

    Two processes: the device's own counters at `GPU_SAMPLE_MS`, and the list of
    compute processes on it once a second. Neither runs any Python while the
    measurement is in flight; both files are read once, afterwards.
    """

    def __init__(self, directory: str, index: str | int | None) -> None:
        self.directory = directory
        self.index = index
        self.procs: list[tuple[str, subprocess.Popen, Any, str]] = []
        self.error: str | None = None
        self.stopped: dict[str, Any] | None = None

    def _spawn(self, kind: str, flag: str, fields: tuple[str, ...], period_ms: int) -> None:
        path = os.path.join(self.directory, f"{kind}.csv")
        argv = ["nvidia-smi", f"--{flag}=" + ",".join(fields),
                "--format=csv,noheader,nounits", f"--loop-ms={period_ms}"]
        if self.index is not None:
            argv += ["-i", str(self.index)]
        handle = open(path, "w")
        # stderr goes to a file, not to DEVNULL: if this nvidia-smi build
        # rejects a flag the sampler is silent, and the reason has to survive
        # into the record rather than be guessed at from an empty CSV.
        errors = open(path + ".err", "w")
        proc = subprocess.Popen(argv, stdout=handle, stderr=errors)
        self.procs.append((kind, proc, (handle, errors), path))

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi not on PATH"
            return
        try:
            os.makedirs(self.directory, exist_ok=True)
            self._spawn("gpu", "query-gpu", GPU_FIELDS, GPU_SAMPLE_MS)
            self._spawn("apps", "query-compute-apps", APP_FIELDS, 10 * GPU_SAMPLE_MS)
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> dict[str, Any]:
        if self.stopped is not None:
            return self.stopped
        out: dict[str, Any] = {"interval_ms": GPU_SAMPLE_MS, "device_index": self.index,
                               "error": self.error, "gpu": [], "apps": []}
        complaints = []
        for kind, proc, handles, path in self.procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                try:
                    proc.kill()
                except OSError:
                    pass
            for handle in handles:
                try:
                    handle.close()
                except OSError:
                    pass
            out[kind] = _read_csv(path, GPU_FIELDS if kind == "gpu" else APP_FIELDS)
            if not out[kind]:
                complaints.append(f"{kind}: {_tail(path + '.err')}")
        if complaints and out["error"] is None:
            out["error"] = "; ".join(complaints)
        self.stopped = out
        return out


def _tail(path: str, limit: int = 400) -> str:
    """The last of a sampler's stderr, for the record when it produced no rows."""
    try:
        with open(path) as f:
            return f.read().strip()[-limit:] or "no rows and no error output"
    except OSError:
        return "no error output"


def _read_csv(path: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Rows of an `nvidia-smi --format=csv,noheader` file, with a unix timestamp."""
    rows: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) != len(fields) or not parts[0]:
                    continue
                row: dict[str, Any] = {}
                for key, value in zip(fields, parts):
                    if key == "timestamp":
                        row["unix"] = _parse_stamp(value)
                    else:
                        row[key] = _number(value)
                rows.append(row)
    except OSError:
        return []
    return rows


def _parse_stamp(value: str) -> float | None:
    """`nvidia-smi`'s `YYYY/MM/DD HH:MM:SS.mmm` to a unix timestamp."""
    try:
        head, _, millis = value.partition(".")
        base = time.mktime(time.strptime(head, "%Y/%m/%d %H:%M:%S"))
        return base + (int(millis) / 1e3 if millis.isdigit() else 0.0)
    except (ValueError, OverflowError):
        return None


def _number(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


class Attribution:
    """The evidence collector for one leg: `with Attribution(...) as a: ...`.

    The measurement calls `a.trace(name, reps)` for each timed loop and hands
    the loop's samples back with `a.record(trace, samples)`. The collector's
    callbacks and the GPU sampler run for the whole `with` block, which covers
    the leg's soak as well as its loops. `as_dict()` is read after the block
    closes, when the sampler has been drained.
    """

    def __init__(self, directory: str | None = None,
                 device_index: str | int | None = None, late_ms: float = LATE_MS) -> None:
        self.late_ms = late_ms
        self.loops: dict[str, Any] = {}
        self.collections: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._pending: dict[int, float] = {}
        self._t0_perf = 0.0
        self._t0_unix = 0.0
        self._gc_enabled: bool | None = None
        self._counts0: tuple[int, ...] = ()
        directory = directory or os.path.join(
            os.environ.get("TMPDIR", "/tmp"),
            f"flash-vla-attribution-{os.getpid()}-{int(time.time() * 1e3)}")
        self._sampler = _GpuSampler(directory, device_index)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Attribution":
        self._t0_unix = time.time()
        self._t0_perf = time.perf_counter()
        self._gc_enabled = gc.isenabled()
        self._counts0 = gc.get_count()
        gc.callbacks.append(self._on_gc)
        self._sampler.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        try:
            gc.callbacks.remove(self._on_gc)
        except ValueError:
            pass
        self._sampler.stop()

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        """`gc.callbacks` entry. Runs inside a collection, so it stays small."""
        now = time.perf_counter()
        generation = info.get("generation", -1)
        if phase == "start":
            self._pending[generation] = now
            return
        started = self._pending.pop(generation, now)
        self.collections.append({
            "generation": generation,
            "start_ms": (started - self._t0_perf) * 1e3,
            "duration_ms": (now - started) * 1e3,
            "collected": info.get("collected"),
            "uncollectable": info.get("uncollectable"),
        })

    # -- per-loop recording ------------------------------------------------

    def trace(self, name: str, reps: int) -> LoopTrace:
        return LoopTrace(name, reps)

    def record(self, trace: LoopTrace, samples: list[float]) -> None:
        try:
            block = trace.as_dict(samples)
            block["late"] = self._late(block, samples)
            self.loops[trace.name] = block
        except Exception as exc:  # evidence, never a failure
            self.errors.append(f"loop {trace.name}: {type(exc).__name__}: {exc}")

    def _late(self, block: dict[str, Any], samples: list[float]) -> list[dict[str, Any]]:
        """Every sample above `min + late_ms`, with what the record says happened in it.

        Offsets are carried on two clocks: `start_ms` is this loop's, `unix` is
        the wall clock the GPU sampler shares, and the collections are matched
        on the leg-wide `perf_counter` origin the callbacks use.
        """
        if not samples:
            return []
        floor = min(samples)
        offset = (block["started_perf"] - self._t0_perf) * 1e3   # loop clock -> leg clock
        out = []
        for index, value in enumerate(samples[:block["reps"]]):
            if value - floor <= self.late_ms:
                continue
            start = block["start_ms"][index]
            out.append({
                "index": index, "ms": round(value, 6), "above_min_ms": round(value - floor, 6),
                "start_ms": start, "unix": block["started_unix"] + start / 1e3,
                "switches": {k: block["per_sample"][k][index] for k in COUNTERS},
                "collections": [c for c in self.collections
                                if start + offset <= c["start_ms"] <= start + offset + value],
                "gpu": [],
            })
        return out

    # -- the record --------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """The `attribution` block. Read after the `with` closes."""
        gpu = self._sampler.stop()
        for block in self.loops.values():
            for entry in block.get("late", []):
                first, last = entry["unix"], entry["unix"] + entry["ms"] / 1e3
                entry["gpu"] = [r for r in gpu["gpu"] if r.get("unix") is not None
                                and first - GPU_SAMPLE_MS / 1e3 <= r["unix"] <= last]
        clocks = [r["clocks.sm"] for r in gpu["gpu"]
                  if isinstance(r.get("clocks.sm"), (int, float))]
        reasons = sorted({str(r.get("clocks_event_reasons.active")) for r in gpu["gpu"]})
        pids = sorted({r["pid"] for r in gpu["apps"] if isinstance(r.get("pid"), int)})
        return {
            "late_ms": self.late_ms,
            "started_unix": self._t0_unix,
            "process": {
                "pid": os.getpid(),
                "affinity": (sorted(os.sched_getaffinity(0))
                             if hasattr(os, "sched_getaffinity") else None),
                "switch_interval_s": sys.getswitchinterval(),
            },
            "loops": self.loops,
            "gc": {
                "enabled_at_entry": self._gc_enabled,
                "counts_at_entry": list(self._counts0),
                "counts_at_exit": list(gc.get_count()),
                "collections": self.collections,
                "totals": {
                    "collections": len(self.collections),
                    "full": sum(1 for c in self.collections if c["generation"] == 2),
                    "ms": round(sum(c["duration_ms"] for c in self.collections), 4),
                    "max_ms": round(max((c["duration_ms"] for c in self.collections),
                                        default=0.0), 4),
                },
            },
            "gpu": {
                "interval_ms": gpu["interval_ms"], "device_index": gpu["device_index"],
                "error": gpu["error"], "samples": gpu["gpu"], "compute_apps": gpu["apps"],
                "clocks_sm": ({"min": min(clocks), "max": max(clocks), "n": len(clocks)}
                              if clocks else None),
                "clock_event_reasons": reasons,
                "other_pids": [p for p in pids if p != os.getpid()],
            },
            "errors": self.errors,
        }


def summary(block: dict[str, Any]) -> str:
    """One human line per loop: how many samples were late, and what sat inside them."""
    lines = []
    for name, loop in block.get("loops", {}).items():
        late = loop.get("late", [])
        worst = max((e["above_min_ms"] for e in late), default=0.0)
        involuntary = sum(e["switches"].get("nivcsw", 0) for e in late)
        voluntary = sum(e["switches"].get("nvcsw", 0) for e in late)
        faults = sum(e["switches"].get("majflt", 0) for e in late)
        collections = sum(len(e["collections"]) for e in late)
        loop_switches = loop.get("ctxt_switch_delta", {})
        lines.append(
            f"    {name:32s} late={len(late):<3d} worst=+{worst:8.3f} ms | in them: "
            f"nvcsw={voluntary} nivcsw={involuntary} majflt={faults} gc={collections} | "
            f"loop: nvcsw={loop_switches.get('voluntary_ctxt_switches')} "
            f"nivcsw={loop_switches.get('nonvoluntary_ctxt_switches')}")
    totals = block.get("gc", {}).get("totals", {})
    lines.append(f"    gc: {totals.get('collections')} collections "
                 f"({totals.get('full')} full), {totals.get('ms')} ms total, "
                 f"max {totals.get('max_ms')} ms")
    gpu = block.get("gpu", {})
    lines.append(f"    gpu: clocks.sm={gpu.get('clocks_sm')} "
                 f"reasons={gpu.get('clock_event_reasons')} "
                 f"other_pids={gpu.get('other_pids')} error={gpu.get('error')}")
    if block.get("errors"):
        lines.append(f"    errors: {block['errors']}")
    return "\n".join(lines)


__all__ = ["APP_FIELDS", "Attribution", "COUNTERS", "GPU_FIELDS", "GPU_SAMPLE_MS", "LATE_MS",
           "LoopTrace", "summary"]
