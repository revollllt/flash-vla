"""Host side of this Target's hand-written pointwise stages.

`build()` compiles `kernels/pointwise.cu` into a plain shared library under
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
_SRC = _HERE / "kernels" / "pointwise.cu"
_ATTENTION_SRC = _HERE / "kernels" / "attention.cu"
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
    tag = hashlib.sha256(_SRC.read_bytes() + _ATTENTION_SRC.read_bytes()
                         + nvcc.encode()).hexdigest()[:16]
    directory = _REPO / ".cache" / "cuda_ext" / f"lingbot_pointwise_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "libpointwise.so"
    if out.exists():
        return out
    command = [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-arch=sm_90a", "-o", str(out), str(_SRC), str(_ATTENTION_SRC)]
    if verbose:
        print("[lingbot pointwise build]", " ".join(command), flush=True)
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
        lib.rms_norm_launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [
            ctypes.c_float, ctypes.c_void_p]
        lib.rms_norm_launch.restype = ctypes.c_int
        lib.ada_rms_add_launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 2 + [
            ctypes.c_float, ctypes.c_void_p]
        lib.ada_rms_add_launch.restype = ctypes.c_int
        lib.rms_norm_add_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [
            ctypes.c_float, ctypes.c_void_p]
        lib.rms_norm_add_launch.restype = ctypes.c_int
        lib.expert_attention_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [
            ctypes.c_longlong] * 2 + [ctypes.c_float, ctypes.c_void_p]
        lib.expert_attention_launch.restype = ctypes.c_int
        lib.silu_multiply_launch.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [
            ctypes.c_void_p]
        lib.silu_multiply_launch.restype = ctypes.c_int
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
    the reduction. The row is held in registers, so `keys <= 512`. One launch,
    capture-safe.
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


def rms_norm(source: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
             epsilon: float) -> torch.Tensor:
    """Upstream's `Qwen2RMSNorm` over the last dimension, in one launch.

    `source` and `target` are `[rows, width]` bf16 contiguous on CUDA and
    `weight` is `[width]` bf16; `target` is written in full and returned.
    Capture-safe.
    """
    lib = library()
    rows, width = source.shape
    code = lib.rms_norm_launch(
        ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(target.data_ptr()), rows, width,
        ctypes.c_float(epsilon), _stream())
    if code != 0:
        raise RuntimeError(f"rms_norm_launch failed: {code}")
    return target


def ada_rms_add(source: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                gamma: torch.Tensor, beta: torch.Tensor, total: torch.Tensor,
                target: torch.Tensor, epsilon: float) -> None:
    """`total = source + residual`, `target = AdaRMS(total)`, in one launch.

    All tensors are bf16 on CUDA: `source`, `residual`, `total` and `target` are
    `[rows, width]` contiguous, and `weight`, `gamma` and `beta` are `[width]`.
    `total` and `target` are written in full. Capture-safe.
    """
    lib = library()
    rows, width = source.shape
    code = lib.ada_rms_add_launch(
        ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(gamma.data_ptr()),
        ctypes.c_void_p(beta.data_ptr()), ctypes.c_void_p(total.data_ptr()),
        ctypes.c_void_p(target.data_ptr()), rows, width, ctypes.c_float(epsilon), _stream())
    if code != 0:
        raise RuntimeError(f"ada_rms_add_launch failed: {code}")


def fused_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                    mask: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    """Masked grouped-query attention over a resident cache, in one launch.

    `query` is `[heads, rows, dim]` contiguous and `keys`/`values` are
    `[kv_heads, key_count, dim]` float32 views sharing one head and key stride,
    contiguous in their last axis; `mask` is `[rows, key_count]` bool. `target` is `[rows, heads * dim]` bf16 and is
    written in full, transposed and rounded for the output projection.
    Capture-safe.
    """
    lib = library()
    heads, rows, dim = query.shape
    kv_heads, key_count, _ = keys.shape
    if keys.stride() != values.stride():
        raise ValueError("the key and value caches must share one stride")
    code = lib.expert_attention_launch(
        ctypes.c_void_p(query.data_ptr()), ctypes.c_void_p(keys.data_ptr()),
        ctypes.c_void_p(values.data_ptr()), ctypes.c_void_p(mask.data_ptr()),
        ctypes.c_void_p(target.data_ptr()), rows, heads, kv_heads, key_count, dim,
        keys.stride(0), keys.stride(1), ctypes.c_float(scale), _stream())
    if code != 0:
        raise RuntimeError(f"expert_attention_launch failed: {code}")
    return target


def rms_norm_add(source: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                 total: torch.Tensor, target: torch.Tensor, epsilon: float) -> None:
    """`total = source + residual`, `target = RMSNorm(total)`, in one launch.

    All tensors are bf16 on CUDA: `source`, `residual`, `total` and `target` are
    `[rows, width]` contiguous and `weight` is `[width]`. `total` and `target`
    are written in full. Capture-safe.
    """
    lib = library()
    rows, width = source.shape
    code = lib.rms_norm_add_launch(
        ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(total.data_ptr()),
        ctypes.c_void_p(target.data_ptr()), rows, width, ctypes.c_float(epsilon), _stream())
    if code != 0:
        raise RuntimeError(f"rms_norm_add_launch failed: {code}")


def silu_multiply(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """`silu(gate) * up` over a packed `[rows, 2 * width]` bf16 gated projection.

    `target` is `[rows, width]` bf16 contiguous on CUDA and is written in full.
    One launch, capture-safe.
    """
    lib = library()
    rows, width = target.shape
    code = lib.silu_multiply_launch(
        ctypes.c_void_p(source.data_ptr()), ctypes.c_void_p(target.data_ptr()),
        rows, width, _stream())
    if code != 0:
        raise RuntimeError(f"silu_multiply_launch failed: {code}")
    return target


__all__ = ["ada_rms_add", "attention_epilogue", "build", "fused_attention", "library",
           "masked_softmax", "rms_norm", "rms_norm_add", "rope_project", "silu_multiply"]
