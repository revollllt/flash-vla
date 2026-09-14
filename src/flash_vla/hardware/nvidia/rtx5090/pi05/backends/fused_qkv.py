"""Pi0.5 expert QKV with fused CUDA normalization and projection epilogue.

For BF16 CUDA x(M,1024), scale(1024), weight(1024,2560), bias(2560),
rope(M,256), this writes Q(M*8,256), K/V(M,256), and norm_factor(M).
All rows are contiguous; K/V may be slices of the layer KV cache. Two scratch
buffers belong to the runner. Warmup builds the library and allocates them;
subsequent calls are CUDA Graph safe on the current stream.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache, partial
from pathlib import Path

import torch

NAMES = frozenset({"action_expert_norm_qkv_rope"})


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_suffix(".cu")
    directory = source.parents[7] / ".cache" / "cuda_ext" / "rtx5090_pi05_fused_qkv"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "fused_qkv.so"
    if not output.exists() or output.stat().st_mtime < source.stat().st_mtime:
        nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc")
        subprocess.run(
            [nvcc, "-O3", "-std=c++17", "--fmad=false", "--shared", "-Xcompiler",
             "-fPIC", "-arch=sm_120a", str(source), "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.pi05_qkv_prepare.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int32, ctypes.c_void_p]
    lib.pi05_qkv_prepare.restype = ctypes.c_int32
    lib.pi05_qkv_finish.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int32, ctypes.c_void_p]
    lib.pi05_qkv_finish.restype = ctypes.c_int32
    return lib


def action_expert_norm_qkv_rope(
    x, scale, weight_qkv, bias, rope, Q, K, V, norm_factor, *, scratch,
):
    """Apply the expert QKV chain with the BF16 roundings of torch_ops."""
    m = x.shape[0]
    scaled = scratch("pi05_qkv_scaled", (m, 1024), x.dtype, x.device)
    projected = scratch("pi05_qkv_projected", (m, 2560), x.dtype, x.device)
    lib = _library()
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.pi05_qkv_prepare(
        x.data_ptr(), scale.data_ptr(), scaled.data_ptr(), norm_factor.data_ptr(), m, stream)
    if rc:
        raise RuntimeError(f"pi05_qkv_prepare rows={m}, threads=256: CUDA error {rc}")
    torch.mm(scaled, weight_qkv, out=projected)
    rc = lib.pi05_qkv_finish(
        projected.data_ptr(), norm_factor.data_ptr(), bias.data_ptr(), rope.data_ptr(),
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), m, stream)
    if rc:
        raise RuntimeError(f"pi05_qkv_finish blocks={m * 5}, threads=256: CUDA error {rc}")


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind the selected expert QKV wrapper to this runner's scratch allocator."""
    names = NAMES if selected_names is None else selected_names
    wrappers = {"action_expert_norm_qkv_rope": action_expert_norm_qkv_rope}
    return {name: partial(wrappers[name], scratch=scratch) for name in names}


__all__ = ["NAMES", "make_wrappers"]
