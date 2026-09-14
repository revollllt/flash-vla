"""Pi0.5 RTX 5090 backbone RMSNorm, BF16 QKV GEMM and fused RoPE/scatter."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import torch

from . import fused_backbone

NAMES = ("llm_backbone_norm_qkv_rope",)
_SOURCE = Path(__file__).with_suffix(".cu")


def _library():
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = os.environ.get("FLASH_VLA_NVCC") or (
        str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc"))
    if nvcc is None:
        raise RuntimeError("set CUDA_HOME or FLASH_VLA_NVCC, or put nvcc on PATH")
    directory = _SOURCE.parents[7] / ".cache/cuda_ext/rtx5090_pi05_prefix_qkv"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "libfused_prefix_qkv.so"
    if not output.exists() or output.stat().st_mtime_ns < max(
            _SOURCE.stat().st_mtime_ns, Path(__file__).stat().st_mtime_ns):
        subprocess.run([nvcc, "-O3", "-std=c++17", "--fmad=false", "--shared",
                        "-Xcompiler", "-fPIC", "-arch=sm_120", str(_SOURCE),
                        "-o", str(output)], check=True)
    library = ctypes.CDLL(str(output))
    library.prefix_rope_scatter.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_int32, ctypes.c_void_p]
    library.prefix_rope_scatter.restype = ctypes.c_int32
    return library


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind contiguous CUDA BF16 prefix QKV: Mx2048 @ 2048x2560.

    RoPE is (M, 256); Q is (M*8, 256), K/V are (M, 256). The wrapper writes
    Q/K/V and x_norm (M, 2048). RMSNorm rounds its normalized output to BF16
    before the GEMM; the GEMM rounds to BF16 before FP32 RoPE arithmetic.
    Warmup builds native libraries and allocates scratch before the runner
    freezes it for capture. Libraries and scratch ownership stay within this
    wrapper factory; constructing the op table requires no CUDA toolchain.
    """
    names = set(NAMES) if selected_names is None else set(selected_names)
    unknown = names - set(NAMES)
    if unknown:
        raise KeyError(f"prefix QKV backend does not implement {sorted(unknown)}")
    libraries = []
    role = f"pi05_prefix_qkv_{id(libraries)}"

    def llm_backbone_norm_qkv_rope(x, weight_qkv, rope, Q, K, V, x_norm):
        if not libraries:
            libraries.extend((fused_backbone._library(), _library()))
        norm_library, rope_library = libraries
        rows = x.shape[0]
        normed = x_norm[:rows]
        projected = scratch(role, (rows, 2560), x.dtype, x.device)
        stream = torch.cuda.current_stream().cuda_stream
        status = norm_library.backbone_rms_norm(x.data_ptr(), normed.data_ptr(), rows, stream)
        if status:
            raise RuntimeError(f"backbone_rms_norm M={rows} K=2048: cudaError {status}")
        torch.mm(normed, weight_qkv, out=projected)
        status = rope_library.prefix_rope_scatter(
            projected.data_ptr(), rope.data_ptr(), Q.data_ptr(), K.data_ptr(),
            V.data_ptr(), rows, stream)
        if status:
            raise RuntimeError(f"prefix_rope_scatter M={rows} width=2560: cudaError {status}")

    return {name: llm_backbone_norm_qkv_rope for name in names}


__all__ = ["NAMES", "make_wrappers"]
