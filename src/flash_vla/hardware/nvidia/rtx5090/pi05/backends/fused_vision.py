"""Pi0.5 vision LayerNorm and post-addmm GELU on handwritten CUDA.

BF16 CUDA x(views,256,1152), norm weight/bias(1152) normalize in FP32 and
round to BF16 before unchanged torch.addmm. QKV writes out(views,256,3456).
FFN writes out(views,256,4304), applying tanh GELU in place only after addmm
has rounded to BF16. Scratch is per runner; warmup compiles and allocates it
before CUDA Graph capture. Rows and affine parameters are contiguous.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache, partial
from pathlib import Path

import torch

NAMES = frozenset({"vision_encoder_norm_qkv", "vision_encoder_norm_ffn_up"})


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_suffix(".cu")
    directory = source.parents[7] / ".cache" / "cuda_ext" / "rtx5090_pi05_fused_vision"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "fused_vision.so"
    if not output.exists() or output.stat().st_mtime < source.stat().st_mtime:
        nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc")
        subprocess.run(
            [nvcc, "-O3", "-std=c++17", "--fmad=false", "--shared", "-Xcompiler",
             "-fPIC", "-arch=sm_120a", str(source), "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.pi05_vision_layer_norm_launch.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int32, ctypes.c_void_p])
    lib.pi05_vision_layer_norm_launch.restype = ctypes.c_int32
    lib.pi05_vision_gelu_launch.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    lib.pi05_vision_gelu_launch.restype = ctypes.c_int32
    return lib


def _norm(x, weight, bias, scratch):
    rows = x.numel() // 1152
    normalized = scratch("pi05_vision_normalized", (rows, 1152), x.dtype, x.device)
    rc = _library().pi05_vision_layer_norm_launch(
        x.data_ptr(), weight.data_ptr(), bias.data_ptr(), normalized.data_ptr(),
        rows, torch.cuda.current_stream(x.device).cuda_stream)
    if rc:
        raise RuntimeError(f"pi05_vision_layer_norm rows={rows}, threads=256: CUDA error {rc}")
    return normalized


def vision_encoder_norm_qkv(x, norm_w, norm_b, qkv_w, qkv_b, out, *, scratch):
    """Normalize in FP32 then write the existing BF16 bias-fused QKV projection."""
    normalized = _norm(x, norm_w, norm_b, scratch)
    torch.addmm(qkv_b, normalized, qkv_w, out=out.view(-1, 3456))
    return out


def vision_encoder_norm_ffn_up(x, norm_w, norm_b, weight, bias, out, *, scratch):
    """Normalize and project before in-place GELU of the BF16-rounded projection."""
    normalized = _norm(x, norm_w, norm_b, scratch)
    torch.addmm(bias, normalized, weight, out=out.view(-1, 4304))
    rc = _library().pi05_vision_gelu_launch(
        out.data_ptr(), out.numel(), torch.cuda.current_stream(out.device).cuda_stream)
    if rc:
        raise RuntimeError(f"pi05_vision_gelu elements={out.numel()}, threads=256: CUDA error {rc}")
    return out


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind only the selected vision wrappers to the runner's scratch."""
    names = NAMES if selected_names is None else selected_names
    wrappers = {"vision_encoder_norm_qkv": vision_encoder_norm_qkv,
                "vision_encoder_norm_ffn_up": vision_encoder_norm_ffn_up}
    return {name: partial(wrappers[name], scratch=scratch) for name in names}


__all__ = ["NAMES", "make_wrappers"]
