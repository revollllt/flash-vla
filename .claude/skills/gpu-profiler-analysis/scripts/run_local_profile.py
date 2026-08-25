#!/usr/bin/env python3
"""Run a local profiler command and write a reproducible artifact manifest."""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS = {
    "torch": ("*.json", "*.json.gz"),
    "nsys": ("*.nsys-rep", "*.sqlite", "*.jsonlines"),
    "ncu": ("*.ncu-rep",),
}


def read_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profile plan must be a JSON object")
    return value


def command_from(args: argparse.Namespace, plan: dict[str, Any]) -> list[str]:
    if plan:
        command = plan.get("command")
    else:
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError("command must be a non-empty JSON list of strings; pass a command after `--`")
    return command


def tool_name(backend: str) -> str:
    return {"torch": "torch.profiler", "nsys": "nsys", "ncu": "ncu"}[backend]


def tool_version(tool: str) -> str | None:
    executable = shutil.which(tool)
    if not executable:
        return None
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    text = (result.stdout or result.stderr).strip().splitlines()
    return text[0] if text else None


def command_output(*command: str) -> str | None:
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0] if lines else None


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*command: str) -> str | None:
        result = subprocess.run(["git", "-C", str(repo_root), *command], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status_entries": len(status.splitlines()) if status else 0}


def wrap_command(backend: str, command: list[str], output_dir: Path) -> list[str]:
    if backend == "torch":
        if not shutil.which(command[0]):
            raise FileNotFoundError(command[0])
        return command
    if backend == "nsys":
        tool = shutil.which("nsys")
        if not tool:
            raise FileNotFoundError("nsys")
        return [
            tool,
            "profile",
            "--force-overwrite=true",
            "--trace=cuda,nvtx,osrt",
            "--cuda-graph-trace=graph",
            "--export=sqlite,jsonlines",
            "--output",
            str(output_dir / "profile"),
            *command,
        ]
    tool = shutil.which("ncu")
    if not tool:
        raise FileNotFoundError("ncu")
    return [
        tool,
        "--force-overwrite",
        "--graph-profiling",
        "node",
        "--export",
        str(output_dir / "profile.ncu-rep"),
        *command,
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_artifacts(output_dir: Path, patterns: list[str]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.name in {"manifest.json", "stdout.log", "stderr.log"}:
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            paths.append(path)
    rows = []
    for path in sorted(paths):
        if path.suffix == ".gz" and path.name.endswith(".json.gz"):
            kind, fmt = "chrome_trace", "chrome-json-gzip"
        elif path.suffix == ".json":
            kind, fmt = "chrome_trace", "chrome-json"
        elif path.suffix == ".nsys-rep":
            kind, fmt = "nsys_report", "nsys-rep"
        elif path.suffix == ".ncu-rep":
            kind, fmt = "ncu_report", "ncu-rep"
        elif path.suffix == ".sqlite":
            kind, fmt = "nsys_export", "sqlite"
        elif path.suffix == ".jsonlines":
            kind, fmt = "nsys_export", "jsonlines"
        else:
            kind, fmt = "profile_artifact", path.suffix.lstrip(".")
        rows.append(
            {
                "kind": kind,
                "path": str(path.relative_to(output_dir)),
                "format": fmt,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(DEFAULT_ARTIFACTS), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan", type=Path, help="JSON CapturePlan; mutually exclusive with the command")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run after `--`")
    args = parser.parse_args(argv)
    if args.plan and args.command:
        parser.error("use either --plan or a command after `--`, not both")

    plan = read_plan(args.plan) if args.plan else {}
    try:
        command = command_from(args, plan)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        wrapped = wrap_command(args.backend, command, output_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    env = os.environ.copy()
    plan_env = plan.get("env", {})
    if plan_env and not isinstance(plan_env, dict):
        parser.error("plan.env must be an object")
    env.update({str(key): str(value) for key, value in plan_env.items()})
    if args.backend == "torch":
        env.setdefault("GPU_PROFILE_OUTPUT_DIR", str(output_dir))

    started = dt.datetime.now(dt.timezone.utc)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            wrapped,
            cwd=str((args.cwd or args.repo_root).resolve()),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    finished = dt.datetime.now(dt.timezone.utc)

    configured_patterns = plan.get("expected_artifacts") or list(DEFAULT_ARTIFACTS[args.backend])
    if not isinstance(configured_patterns, list) or not all(isinstance(item, str) for item in configured_patterns):
        parser.error("plan.expected_artifacts must be a list of glob strings")
    artifacts = discover_artifacts(output_dir, configured_patterns)
    manifest = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "created_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "exit_code": result.returncode,
        "tool": {"backend": args.backend, "tool_name": tool_name(args.backend), "tool_version": tool_version(tool_name(args.backend)), "command": wrapped},
        "host": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": command_output("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
            "driver_version": command_output("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            "nvcc_version": command_output("nvcc", "--version"),
            "python": sys.version.split()[0],
        },
        "software": {"git": git_metadata(args.repo_root.resolve())},
        "workload": plan.get("workload", {}),
        "capture": plan.get("capture", {}),
        "artifacts": artifacts,
        "validation": {"expected_patterns": configured_patterns, "artifacts_found": bool(artifacts)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "exit_code": result.returncode, "artifacts": artifacts}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
