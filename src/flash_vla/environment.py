"""Device observations and provenance shared by evaluation and timing tools."""
from __future__ import annotations
import csv
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from typing import Any
import torch

def env_block(device=None) -> dict[str, Any]:
    """GPU / toolchain versions, for stamping result files."""
    try:
        tilelang_version = metadata.version("tilelang")
    except metadata.PackageNotFoundError:
        tilelang_version = "unavailable"

    return {
        "gpu": torch.cuda.get_device_name(device),
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tilelang": tilelang_version,
    }


def require_cuda() -> None:
    """Fail early when CUDA is unavailable in the active runtime."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with a visible GPU and a compatible PyTorch/CUDA environment")


def report_context(engine, environment):
    """Stamp measurement provenance separately from architecture identity."""
    from flash_vla.runtime.identity import MeasurementContext
    return MeasurementContext(
        weights=engine.measurement_context["weights"],
        fixture=engine.measurement_context["fixture"],
        environment={
            "gpu_sku": environment.get("gpu"), "driver": environment.get("driver"),
            "cuda_runtime": environment.get("torch_cuda"), "pytorch": environment.get("torch"),
            "tilelang": environment.get("tilelang"), "clock_policy": environment.get("clock_policy"),
            "power_policy": environment.get("power_policy"), "capture_regime": "cuda_graph",
            "clock_observation": environment.get("clock_observation"),
        },
        hostname=environment.get("node"), slurm_job_id=environment.get("job"),
        timestamp=time.time(),
        reference_provenance=engine.measurement_context.get("reference_provenance", {}),
    ).as_dict()


def collect(device=None) -> dict[str, Any]:
    observation_fields = ("clocks.sm", "clocks.mem", "pstate", "temperature.gpu",
                          "power.draw", "clocks_event_reasons.active")
    fields = ("driver_version", "power.limit", "enforced.power.limit",
              "clocks.applications.graphics", "clocks.applications.memory",
              *observation_fields)
    uuid = device_selector(device)
    result = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-gpu=" + ",".join(fields),
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise ValueError("expected one complete nvidia-smi row for the current CUDA device")
    driver, requested, enforced, graphics, memory, *observations = rows[0]
    env = env_block(device)
    env.update({
        "gpu_uuid": uuid,
        "runtime_observation": dict(zip(observation_fields, observations)),
        "node": platform.node(),
        "job": os.environ.get("SLURM_JOB_ID"),
        "driver": driver,
        "clocks": "unlocked",
        # This is the benchmark's execution policy, not a global hardware-lock
        # certificate. Application clocks remain separate observations.
        "clock_policy": {
            "benchmark_control": "inherit",
            "slurm_gpu_freq_request": os.environ.get("SLURM_GPU_FREQ"),
            "effective_locked_clocks": "unobserved",
        },
        "clock_observation": {"application_graphics_mhz": graphics,
                              "application_memory_mhz": memory},
        "power_policy": {"requested_limit_w": float(requested),
                         "enforced_limit_w": float(enforced)},
    })
    return env


def device_selector(device=None) -> str:
    """Select the actual CUDA device, including CUDA_VISIBLE_DEVICES remapping."""
    uuid = str(torch.cuda.get_device_properties(
        torch.cuda.current_device() if device is None else device).uuid)
    return uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"
