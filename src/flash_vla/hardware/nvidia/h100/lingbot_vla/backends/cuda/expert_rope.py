"""Host side of the action expert's hand-written pointwise stages.

`build()` compiles `kernels/expert_rope.cu` into a plain shared library under
the repo's `.cache` (shared filesystem, so a login-node build is visible to a
compute node) and loads it through ctypes: the library has a C ABI and takes
raw device pointers, so no torch extension machinery is involved and every
launch is safe inside a CUDA-graph capture.

The toolkit must match the environment's torch build, not the newest module on
the machine: this Target runs torch 2.9.1+cu126 on a partition that mixes a
570 (CUDA 12.8) and a 610 (CUDA 13.x) driver, and a cubin from CUDA 13 will not
load on the 570 driver once the forward-compat directory has been dropped.
`LINGBOT_NVCC` overrides the default when a job needs another toolkit.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "expert_rope.cu"
_REPO = _HERE.parents[7]
_DEFAULT_NVCC = "/data/apps/cuda/12.6/bin/nvcc"

_LIB = None


def _nvcc() -> str:
    explicit = os.environ.get("LINGBOT_NVCC")
    if explicit:
        return explicit
    return _DEFAULT_NVCC if Path(_DEFAULT_NVCC).is_file() else "nvcc"


def build(verbose: bool = False) -> Path:
    """Compile the shared library if this source and toolchain have not been built."""
    nvcc = _nvcc()
    tag = hashlib.sha256(_SRC.read_bytes() + nvcc.encode()).hexdigest()[:16]
    directory = _REPO / ".cache" / "cuda_ext" / f"lingbot_expert_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "libexpert.so"
    if out.exists():
        return out
    command = [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-arch=sm_90a", "-o", str(out), str(_SRC)]
    if verbose:
        print("[expert build]", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{result.stdout}\n{result.stderr}")
    return out


def library(verbose: bool = False):
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build(verbose=verbose)))
        lib.expert_rope_launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 8 + [
            ctypes.c_void_p]
        lib.expert_rope_launch.restype = ctypes.c_int
        lib.expert_softmax_launch.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [
            ctypes.c_float, ctypes.c_void_p]
        lib.expert_softmax_launch.restype = ctypes.c_int
        lib.expert_epilogue_launch.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [
            ctypes.c_void_p]
        lib.expert_epilogue_launch.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _stream() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def rope_project(packed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 query: torch.Tensor, key_slot: torch.Tensor, value_slot: torch.Tensor) -> None:
    """Widen, rotate and scatter one layer-step's packed q/k/v projection.

    `packed` is `[rows, q_heads * head_dim + 2 * kv_heads * head_dim]` bf16 and
    `cos`/`sin` are `[rows, head_dim // 2]` float32, both contiguous. `query` is
    `[q_heads, rows, head_dim]` and `key_slot`/`value_slot` are
    `[kv_heads, rows, head_dim]` float32 views whose head and row strides may be
    those of a larger cache; their last dimension must be contiguous. All three
    outputs are written in full. One launch, capture-safe.
    """
    lib = library()
    q_heads, rows, head_dim = query.shape
    kv_heads = key_slot.shape[0]
    code = lib.expert_rope_launch(
        ctypes.c_void_p(packed.data_ptr()), ctypes.c_void_p(cos.data_ptr()),
        ctypes.c_void_p(sin.data_ptr()), ctypes.c_void_p(query.data_ptr()),
        ctypes.c_void_p(key_slot.data_ptr()), ctypes.c_void_p(value_slot.data_ptr()),
        rows, q_heads, kv_heads, head_dim,
        query.stride(0), query.stride(1), key_slot.stride(0), key_slot.stride(1),
        _stream())
    if code != 0:
        raise RuntimeError(f"expert_rope_launch failed: {code}")


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, scale: float) -> None:
    """Scale, apply `mask` and softmax `scores` in place along its last dimension.

    `scores` is `[heads, q_rows, keys]` float32 contiguous and `mask` is
    `[q_rows, keys]` bool; masked logits take upstream's finite sentinel before
    the reduction. One launch, capture-safe.
    """
    lib = library()
    heads, q_rows, keys = scores.shape
    code = lib.expert_softmax_launch(
        ctypes.c_void_p(scores.data_ptr()), ctypes.c_void_p(mask.data_ptr()),
        heads, q_rows, keys, ctypes.c_float(scale), _stream())
    if code != 0:
        raise RuntimeError(f"expert_softmax_launch failed: {code}")


def attention_epilogue(source: torch.Tensor, target: torch.Tensor) -> None:
    """`[heads, rows, head_dim]` float32 to `[rows, heads * head_dim]` bf16.

    Both contiguous on CUDA; `target` is written in full. One launch, capture-safe.
    """
    lib = library()
    heads, rows, head_dim = source.shape
    code = lib.expert_epilogue_launch(
        ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(target.data_ptr()),
        rows, heads, head_dim, _stream())
    if code != 0:
        raise RuntimeError(f"expert_epilogue_launch failed: {code}")


__all__ = ["attention_epilogue", "build", "library", "masked_softmax", "rope_project"]
