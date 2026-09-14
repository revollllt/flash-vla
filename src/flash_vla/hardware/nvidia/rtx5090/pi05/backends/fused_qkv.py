"""Pi0.5 expert QKV and action output with fused CUDA pointwise stages.

For BF16 CUDA x(M,1024), scale(1024), weight(1024,2560), bias(2560),
rope(M,256), this writes Q(M*8,256), K/V(M,256), and norm_factor(M).
All rows are contiguous; K/V may be slices of the layer KV cache. Two scratch
buffers belong to the runner. Warmup builds the library and allocates them;
subsequent calls are CUDA Graph safe on the current stream. The action output
wrapper uses weight(1024,32), bias(32), and out(M,32); it privately computes the
BF16 RMS factor and leaves its norm_factor argument unchanged.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache, partial
from pathlib import Path

import torch

NAMES = frozenset({"action_expert_norm_qkv_rope", "action_expert_action_out_proj"})


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
    lib.pi05_action_out_factor_launch.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_int32, ctypes.c_void_p])
    lib.pi05_action_out_factor_launch.restype = ctypes.c_int32
    lib.pi05_action_out_update_launch.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int32, ctypes.c_void_p])
    lib.pi05_action_out_update_launch.restype = ctypes.c_int32
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


def action_expert_action_out_proj(x, weight, bias, out, norm_factor, *, scratch):
    """Update BF16 actions after FP32 factor/bias/residual; do not write norm_factor."""
    rows = x.shape[0]
    factor = scratch("pi05_action_out_factor", (rows,), x.dtype, x.device)
    projected = scratch("pi05_action_out_projected", (rows, 32), x.dtype, x.device)
    lib = _library()
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.pi05_action_out_factor_launch(x.data_ptr(), factor.data_ptr(), rows, stream)
    if rc:
        raise RuntimeError(f"pi05_action_out_factor rows={rows}, threads=256: CUDA error {rc}")
    torch.mm(x, weight, out=projected)
    rc = lib.pi05_action_out_update_launch(
        projected.data_ptr(), factor.data_ptr(), bias.data_ptr(), out.data_ptr(), rows, stream)
    if rc:
        raise RuntimeError(f"pi05_action_out_update rows={rows}, threads=256: CUDA error {rc}")
    return out


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind the selected expert projection wrappers to this runner's scratch allocator."""
    names = NAMES if selected_names is None else selected_names
    wrappers = {"action_expert_norm_qkv_rope": action_expert_norm_qkv_rope,
                "action_expert_action_out_proj": action_expert_action_out_proj}
    return {name: partial(wrappers[name], scratch=scratch) for name in names}


__all__ = ["NAMES", "make_wrappers"]
