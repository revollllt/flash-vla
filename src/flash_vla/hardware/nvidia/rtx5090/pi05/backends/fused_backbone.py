"""Pi0.5 backbone FFN with CUDA RMSNorm and a fused GELU/product.

The two torch BF16 GEMMs retain the torch route's rounding boundaries.
The wrapper writes its declared output and x_norm buffers; the runner owns
its gate workspace. Native launch state belongs to each wrapper factory.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import torch

NAMES = frozenset({"llm_backbone_norm_gated_ffn"})
_SOURCE = Path(__file__).with_suffix(".cu")


def _library():
    nvcc = os.environ.get("FLASH_VLA_NVCC")
    if nvcc is None:
        cuda_home = os.environ.get("CUDA_HOME")
        nvcc = str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("set CUDA_HOME or FLASH_VLA_NVCC, or put nvcc on PATH")
    directory = _SOURCE.parents[7] / ".cache/cuda_ext/rtx5090_pi05_backbone"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "libfused_backbone.so"
    if not output.exists() or output.stat().st_mtime < _SOURCE.stat().st_mtime:
        subprocess.run(
            [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
             "-gencode", "arch=compute_120,code=sm_120",
             str(_SOURCE), "-o", str(output)],
            check=True,
        )
    library = ctypes.CDLL(str(output))
    library.backbone_rms_norm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
    ]
    library.backbone_rms_norm.restype = ctypes.c_int32
    library.backbone_gelu_mul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
    ]
    library.backbone_gelu_mul.restype = ctypes.c_int32
    return library


def make_wrappers(scratch, selected_names=None) -> dict:
    """Build capture-safe wrappers for this Target's contiguous CUDA BF16 FFN.

    x/x_norm are (M, 2048), weights (2048, 16384), and out (M, 16384).
    x_norm and out are distinct writable buffers; out temporarily holds up.
    The runner must warm each shape before freezing the supplied allocator.
    """
    library = None

    def llm_backbone_norm_gated_ffn(x, gate_w, up_w, out, x_norm):
        nonlocal library
        if library is None:
            library = _library()
        rows = x.shape[0]
        normed, result = x_norm[:rows], out[:rows]
        gate = scratch("backbone_ffn_gate", result.shape, x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        status = library.backbone_rms_norm(x.data_ptr(), normed.data_ptr(), rows, stream)
        if status:
            raise RuntimeError(f"backbone_rms_norm M={rows} K=2048: cudaError {status}")
        torch.mm(normed, gate_w, out=gate)
        torch.mm(normed, up_w, out=result)
        status = library.backbone_gelu_mul(
            gate.data_ptr(), result.data_ptr(), result.numel(), stream)
        if status:
            raise RuntimeError(
                f"backbone_gelu_mul elements={result.numel()}: cudaError {status}")
        return out

    return {"llm_backbone_norm_gated_ffn": llm_backbone_norm_gated_ffn}
