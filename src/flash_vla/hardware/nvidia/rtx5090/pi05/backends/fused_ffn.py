"""Pi0.5 expert FFN with hand-written AdaRMS and gated-activation stages.

The two GEMMs remain torch.mm. The CUDA stages preserve the torch route's
bf16 factor, two bf16 activation multiplications and bf16 GEMM outputs.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from flash_vla.hardware.nvidia.native import NativeLibrary
from flash_vla.runtime.registry import Backend

NAMES = ("action_expert_norm_gated_ffn",)
SOURCE = Path(__file__).with_suffix(".cu")

#: The expert FFN's AdaRMS, gated-activation and gated-residual kernels.
LIBRARY = NativeLibrary(
    name="rtx5090_pi05_fused_ffn",
    sources=(SOURCE,),
    arch=("-arch=sm_120",),
    flags=("--fmad=false",))


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The loaded library with its C ABI declared; build it before graph capture."""
    kernels = LIBRARY.load()
    kernels.ada_rms_launch.argtypes = [ctypes.c_void_p] * 4 + [
        ctypes.c_int32, ctypes.c_void_p]
    kernels.ada_rms_launch.restype = ctypes.c_int32
    kernels.gated_activation_launch.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_int32, ctypes.c_void_p]
    kernels.gated_activation_launch.restype = ctypes.c_int32
    kernels.packed_gated_activation_launch.argtypes = kernels.gated_activation_launch.argtypes
    kernels.packed_gated_activation_launch.restype = ctypes.c_int32
    kernels.gated_residual_launch.argtypes = [ctypes.c_void_p] * 3 + [
        ctypes.c_int32, ctypes.c_void_p]
    kernels.gated_residual_launch.restype = ctypes.c_int32
    return kernels


def check(status: int, kernel: str, rows: int) -> None:
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
        lib = library()
        stream = torch.cuda.current_stream().cuda_stream
        check(lib.ada_rms_launch(x.data_ptr(), scale.data_ptr(), a.data_ptr(),
                                 norm_factor.data_ptr(), rows, stream), "ada_rms", rows)
        torch.mm(a, gate_w, out=gate)
        torch.mm(a, up_w, out=up)
        check(lib.gated_activation_launch(
            gate.data_ptr(), up.data_ptr(), gate_b.data_ptr(), up_b.data_ptr(),
            out.data_ptr(), rows, stream), "gated_activation", rows)
        return out

    return {name: action_expert_norm_gated_ffn for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
