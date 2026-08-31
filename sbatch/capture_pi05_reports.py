"""Run the Pi0.5 stage benchmarks and persist their reports as JSON.

`benchmarks.e2e_pi05` and `benchmarks.profile_pi05` print a report and return
it. Only the printed copy has ever survived a job, and it is interleaved with
TileLang's compiler output, so every recorded number in `models/pi05/PLAN.md`
has to be re-measured rather than cited. This captures the RETURN value instead
of parsing the log, and writes a manifest beside it so the numbers carry their
own provenance.

Both benchmarks are run in one job, e2e first: the profiler perturbs
scheduling, so a latency claim must not come from a process that has already
profiled. They build separate engines; the second build is cheap because
TileLang's cache is warm by then.

Not a benchmark itself -- it adds no timing of its own and changes none of the
knobs the benchmarks own.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git() -> dict[str, object]:
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    status = run("git", "status", "--porcelain")
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_entries": len(status.splitlines()) if status else 0}


def _host() -> dict[str, str]:
    def smi(query):
        try:
            return subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            return ""
    return {"hostname": os.uname().nodename,
            "gpu_name": smi("name"),
            "driver_version": smi("driver_version"),
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def _software() -> dict[str, object]:
    versions = {}
    for name in ("torch", "tilelang"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:                                    # noqa: BLE001
            versions[name] = "unavailable"
    return {**versions, "git": _git()}


def _write(path: Path, report: dict) -> dict[str, object]:
    """Write one report and describe it for the manifest."""
    payload = json.dumps(report, indent=2, sort_keys=True).encode()
    path.write_bytes(payload)
    return {"path": path.name, "kind": path.stem.split("_")[0], "format": "json",
            "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def run(*, out_dir: Path, tag: str, num_views: int, chunk_size: int, steps: int,
        layers: int, reps: int, top: int, seed: int, prompt: str | None,
        tokenizer: str | None, trace_dir: str | None, only: str | None,
        plan: str | None = None) -> int:
    """Run the selected benchmarks, persist each report, then write the manifest."""
    from benchmarks import e2e_pi05, profile_pi05

    out_dir.mkdir(parents=True, exist_ok=True)
    shared = dict(num_views=num_views, chunk_size=chunk_size, steps=steps, layers=layers,
                  seed=seed, tokenizer_path=tokenizer,
                  plan=e2e_pi05.parse_plan(plan) if plan else None)
    if prompt is not None:
        shared["prompt"] = prompt

    started = datetime.now(timezone.utc).isoformat()
    artifacts = []

    if only in (None, "e2e"):
        print("[capture] e2e-pi05: per-stage latency, uncontaminated by the profiler",
              flush=True)
        artifacts.append(_write(out_dir / f"e2e_{tag}.json",
                                e2e_pi05.run(reps=reps, **shared)))
        _release()

    if only in (None, "profile"):
        print("[capture] profile-pi05: per-kernel self-device-time inside each graph",
              flush=True)
        artifacts.append(_write(out_dir / f"profile_{tag}.json",
                                profile_pi05.run(top=top, trace_dir=trace_dir, **shared)))
        _release()

    manifest = {
        "schema_version": 1,
        "run_id": f"pi05-stages-{tag}",
        "created_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": _host(),
        "software": _software(),
        "tool": {"tool_name": "benchmarks.{e2e,profile}_pi05", "command": sys.argv},
        "config": {**shared, "reps": reps, "top": top, "trace_dir": trace_dir},
        "artifacts": artifacts,
    }
    (out_dir / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2,
                                                             sort_keys=True))
    print(f"[capture] wrote {len(artifacts)} report(s) + manifest to {out_dir}")
    return 0


def _release() -> None:
    """Drop the finished engine before the next one allocates its weights."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("profiles/pi05"))
    parser.add_argument("--tag", default=os.environ.get("SLURM_JOB_ID", "local"),
                        help="filename suffix; defaults to $SLURM_JOB_ID")
    parser.add_argument("--only", choices=("e2e", "profile"), default=None)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--layers", type=int, default=18)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--top", type=int, default=40,
                        help="kernels listed per stage; the default is above the "
                             "distinct-kernel count so no stage is truncated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--tokenizer", default=os.environ.get("PALIGEMMA_TOKENIZER"))
    parser.add_argument("--trace-dir", default=None,
                        help="also export per-stage Chrome traces here")
    parser.add_argument("--plan", default=None,
                        help="call-site plan, a name from e2e_pi05.PLANS or a JSON object")
    args = parser.parse_args(argv)
    return run(out_dir=args.out_dir, tag=args.tag, num_views=args.num_views,
               chunk_size=args.chunk_size, steps=args.steps, layers=args.layers,
               reps=args.reps, top=args.top, seed=args.seed, prompt=args.prompt,
               tokenizer=args.tokenizer, trace_dir=args.trace_dir, only=args.only,
               plan=args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
