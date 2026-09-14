"""Pi0.5 expert FFN with hand-written AdaRMS and gated-activation stages.

The two GEMMs remain torch.mm. The CUDA stages preserve the torch route's
bf16 factor, two bf16 activation multiplications and bf16 GEMM outputs.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess

import torch

NAMES = ("action_expert_norm_gated_ffn",)
_HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _library():
    """Build before graph capture; retain only the host library globally."""
    source = _HERE / "fused_ffn.cu"
    directory = _HERE.parents[6] / ".cache" / "cuda_ext" / "rtx5090_pi05_fused_ffn"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "libfused_ffn.so"
    if not output.exists() or output.stat().st_mtime_ns < max(
            source.stat().st_mtime_ns, Path(__file__).stat().st_mtime_ns):
        cuda_home = os.environ.get("CUDA_HOME")
        nvcc = os.environ.get("FLASH_VLA_NVCC") or (
            str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else shutil.which("nvcc"))
        if nvcc is None:
            raise RuntimeError("Pi0.5 fused FFN requires nvcc on PATH or CUDA_HOME")
        subprocess.run([nvcc, "-O3", "-std=c++17", "--fmad=false", "--shared",
                        "-Xcompiler", "-fPIC", "-arch=sm_120", str(source),
                        "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.ada_rms_launch.argtypes = [ctypes.c_void_p] * 4 + [
        ctypes.c_int32, ctypes.c_void_p]
    lib.ada_rms_launch.restype = ctypes.c_int32
    lib.gated_activation_launch.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_int32, ctypes.c_void_p]
    lib.gated_activation_launch.restype = ctypes.c_int32
    return lib


def _check(status: int, kernel: str, rows: int) -> None:
    if status != 0:
        raise RuntimeError(f"{kernel}(rows={rows}) failed: cudaError {status}")


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind scratch-owned workspaces for contiguous CUDA bf16 Mx1024 -> Mx4096.

    Scale is (1024,), both weights are (1024, 4096), and biases are (4096,).
    The wrapper writes ``out`` and ``norm_factor``. Warmup allocates buffers
    and builds the library before scratch freezes and CUDA graph capture starts.
    Each factory call owns its workspaces; no device tensors are module globals.
    """
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"fused FFN backend does not implement {sorted(unknown)}")
    workspaces = {}
    role = f"pi05_fused_ffn_{id(workspaces)}"

    def action_expert_norm_gated_ffn(x, scale, gate_w, up_w, gate_b, up_b,
                                     out, norm_factor):
        rows = x.shape[0]
        key = (rows, x.dtype, x.device)
        if key not in workspaces:
            workspaces[key] = (
                scratch(role + "_a", (rows, 1024), x.dtype, x.device),
                scratch(role + "_gate", (rows, 4096), x.dtype, x.device),
                scratch(role + "_up", (rows, 4096), x.dtype, x.device),
            )
        a, gate, up = workspaces[key]
        lib = _library()
        stream = torch.cuda.current_stream().cuda_stream
        _check(lib.ada_rms_launch(x.data_ptr(), scale.data_ptr(), a.data_ptr(),
                                 norm_factor.data_ptr(), rows, stream), "ada_rms", rows)
        torch.mm(a, gate_w, out=gate)
        torch.mm(a, up_w, out=up)
        _check(lib.gated_activation_launch(
            gate.data_ptr(), up.data_ptr(), gate_b.data_ptr(), up_b.data_ptr(),
            out.data_ptr(), rows, stream), "gated_activation", rows)
        return out

    return {name: action_expert_norm_gated_ffn for name in names}


__all__ = ["NAMES", "make_wrappers"]
