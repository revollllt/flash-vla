"""Host side of the split-key fused attention kernel.

`kernels/split_attention.cu` is compiled on its own into a plain shared library
under the repo's `.cache` and loaded through ctypes with raw device pointers,
the same self-contained pattern as `pointwise.py`: a C ABI and no torch
extension machinery, so every launch is safe inside a CUDA-graph capture.

The toolkit must match the environment's torch build. This Target runs torch
2.9.1+cu126 on a partition that mixes a 570 (CUDA 12.8) and a 610 (CUDA 13.x)
driver, and a cubin from CUDA 13 will not load on the 570 driver once the
forward-compat directory has been dropped. `LINGBOT_NVCC` overrides the default.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "split_attention.cu"
_REPO = _HERE.parents[7]
_DEFAULT_NVCC = "/data/apps/cuda/12.6/bin/nvcc"

# The grid shape, measured on an H100 at the Target's shape (job 614698,
# medians under graph replay, us per layer-step). 315 keys / 40 = 8 slices x 16
# heads = 128 CTAs, one wave on the 132 SMs, with all 51 rows in one CTA whose
# eight warps each cover every row:
#     keys 40, unroll 4, pieces 2   12.56      keys 48   13.67
#     keys 40, unroll 2, pieces 2   12.62      keys 64   15.62
#     keys 40, unroll 2, pieces 1   12.76      32-row CTA (256 CTAs)   13.57
# The kernel also instantiates 24 and 32 keys and a two-row-group warp grid;
# both were slower, and both stay reachable so the result can be re-measured.
DEFAULT_KEY_TILE = 40
DEFAULT_ROW_TILE = 64
DEFAULT_ROW_GROUPS = 1
DEFAULT_UNROLL = 4
DEFAULT_PIECES = 2
# The float32 mainloops are the default. `tensor=True` selects TF32 tensor-core
# mainloops instead, which is the precision the upstream reference itself runs
# at -- its QK is sm80_xmma_gemm_f32f32_tf32f32_f32 and its PV
# cutlass_80_tensorop_s1688gemm (job 614222) -- so it is not a step below the
# oracle, but it is an approximation relative to this kernel's float32 path and
# belongs in a parity run. The TF32 path fixes 64 query rows per CTA and takes
# key_tile 32, 48 or 64.
#
# Measured, it is a wash: 12.57 us at key_tile 32 against the float32 path's
# 12.53 at 40 (job 615519), because the mainloops it accelerates are only a
# third of the kernel. It is kept selectable, not recommended -- the float32
# path is the same speed and strictly closer to a float64 reference.
DEFAULT_TENSOR = False
TENSOR_KEY_TILE = 32

_LIB = None
_WORKSPACE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _nvcc() -> str:
    explicit = os.environ.get("LINGBOT_NVCC")
    if explicit:
        return explicit
    return _DEFAULT_NVCC if Path(_DEFAULT_NVCC).is_file() else "nvcc"


def build(verbose: bool = False) -> Path:
    """Compile the shared library if this source and toolchain have not been built."""
    nvcc = _nvcc()
    tag = hashlib.sha256(_SRC.read_bytes() + nvcc.encode()).hexdigest()[:16]
    directory = _REPO / ".cache" / "cuda_ext" / f"lingbot_split_attention_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "libsplitattention.so"
    if out.exists():
        return out
    command = [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-arch=sm_90a", "-lineinfo", "-o", str(out), str(_SRC)]
    if verbose:
        print("[lingbot split attention build]", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{result.stdout}\n{result.stderr}")
    return out


def library(verbose: bool = False):
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build(verbose=verbose)))
        lib.split_attention_launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [
            ctypes.c_longlong] * 2 + [ctypes.c_float] + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        lib.split_attention_launch.restype = ctypes.c_int
        lib.split_attention_partials.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [
            ctypes.c_longlong] * 2 + [ctypes.c_float] + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.split_attention_partials.restype = ctypes.c_int
        lib.split_attention_combine.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [
            ctypes.c_void_p]
        lib.split_attention_combine.restype = ctypes.c_int
        lib.split_attention_timed.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [
            ctypes.c_longlong] * 2 + [ctypes.c_float] + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        lib.split_attention_timed.restype = ctypes.c_int
        lib.split_attention_splits.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.split_attention_splits.restype = ctypes.c_int
        lib.split_attention_noop.argtypes = [ctypes.c_void_p]
        lib.split_attention_noop.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _stream() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def splits_for(key_count: int, key_tile: int = DEFAULT_KEY_TILE) -> int:
    """Number of key slices the grid will carry for this key count."""
    return int(library().split_attention_splits(key_count, key_tile))


def workspace(heads: int, rows: int, dim: int, key_count: int, device,
              key_tile: int = DEFAULT_KEY_TILE):
    """Allocate (and memoize) the per-slice partials the two launches share.

    Returns `(accumulator, maximum, denominator)`: float32 tensors of
    `[heads, rows, splits, dim]`, `[heads, rows, splits]` and
    `[heads, rows, splits]`. The slice axis is innermost so that the combine
    reads one row's slices contiguously. Allocate before a CUDA-graph capture; the launches
    only write and read them, so the same buffers may back every layer-step.
    """
    splits = splits_for(key_count, key_tile)
    key = (splits, heads, rows, dim, str(device))
    found = _WORKSPACE.get(key)
    if found is None:
        accumulator = torch.empty((heads, rows, splits, dim), dtype=torch.float32,
                                  device=device)
        maximum = torch.empty((heads, rows, splits), dtype=torch.float32, device=device)
        denominator = torch.empty_like(maximum)
        found = (accumulator, maximum, denominator)
        _WORKSPACE[key] = found
    return found


def fused_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                    mask: torch.Tensor, target: torch.Tensor, scale: float,
                    buffers=None, key_tile: int = DEFAULT_KEY_TILE,
                    row_tile: int = DEFAULT_ROW_TILE,
                    row_groups: int = DEFAULT_ROW_GROUPS,
                    unroll: int = DEFAULT_UNROLL,
                    pieces: int = DEFAULT_PIECES,
                    tensor: bool = DEFAULT_TENSOR) -> torch.Tensor:
    """Masked grouped-query attention over a resident cache, in two launches.

    `query` is `[heads, rows, dim]` float32 contiguous and `keys`/`values` are
    `[kv_heads, key_count, dim]` float32 views sharing one head and key stride,
    contiguous in their last axis; `mask` is `[rows, key_count]` bool. `target`
    is `[rows, heads * dim]` bf16 and is written in full, transposed and rounded
    for the output projection. `dim` must be 128, and
    `(key_tile, row_tile, row_groups, unroll, pieces)` must name an
    instantiated grid shape.

    `buffers` is the `workspace()` triple; it is allocated on demand when
    omitted, which is not capture-safe, so a captured caller passes its own.
    Both launches are capture-safe. Returns `target`.
    """
    lib = library()
    heads, rows, dim = query.shape
    kv_heads, key_count, _ = keys.shape
    if keys.stride() != values.stride():
        raise ValueError("the key and value caches must share one stride")
    if buffers is None:
        buffers = workspace(heads, rows, dim, key_count, query.device, key_tile)
    accumulator, maximum, denominator = buffers
    code = lib.split_attention_launch(
        ctypes.c_void_p(query.data_ptr()), ctypes.c_void_p(keys.data_ptr()),
        ctypes.c_void_p(values.data_ptr()), ctypes.c_void_p(mask.data_ptr()),
        ctypes.c_void_p(accumulator.data_ptr()), ctypes.c_void_p(maximum.data_ptr()),
        ctypes.c_void_p(denominator.data_ptr()), ctypes.c_void_p(target.data_ptr()),
        rows, heads, kv_heads, key_count, dim,
        keys.stride(0), keys.stride(1), ctypes.c_float(scale), key_tile, row_tile, row_groups, unroll, pieces, int(tensor),
        _stream())
    if code != 0:
        raise RuntimeError(f"split_attention_launch failed: {code}")
    return target


def partials(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
             mask: torch.Tensor, scale: float, buffers,
             key_tile: int = DEFAULT_KEY_TILE, row_tile: int = DEFAULT_ROW_TILE,
             row_groups: int = DEFAULT_ROW_GROUPS,
             unroll: int = DEFAULT_UNROLL, tensor: bool = DEFAULT_TENSOR) -> None:
    """The first of the two launches on its own, for attributing its latency."""
    lib = library()
    heads, rows, dim = query.shape
    kv_heads, key_count, _ = keys.shape
    accumulator, maximum, denominator = buffers
    code = lib.split_attention_partials(
        ctypes.c_void_p(query.data_ptr()), ctypes.c_void_p(keys.data_ptr()),
        ctypes.c_void_p(values.data_ptr()), ctypes.c_void_p(mask.data_ptr()),
        ctypes.c_void_p(accumulator.data_ptr()), ctypes.c_void_p(maximum.data_ptr()),
        ctypes.c_void_p(denominator.data_ptr()),
        rows, heads, kv_heads, key_count, dim,
        keys.stride(0), keys.stride(1), ctypes.c_float(scale), key_tile, row_tile, row_groups, unroll, int(tensor),
        _stream())
    if code != 0:
        raise RuntimeError(f"split_attention_partials failed: {code}")


def combine(buffers, target: torch.Tensor, pieces: int = DEFAULT_PIECES) -> None:
    """The second of the two launches on its own, for attributing its latency."""
    lib = library()
    accumulator, maximum, denominator = buffers
    heads, rows, splits, _ = accumulator.shape
    code = lib.split_attention_combine(
        ctypes.c_void_p(accumulator.data_ptr()), ctypes.c_void_p(maximum.data_ptr()),
        ctypes.c_void_p(denominator.data_ptr()), ctypes.c_void_p(target.data_ptr()),
        rows, heads, splits, pieces, _stream())
    if code != 0:
        raise RuntimeError(f"split_attention_combine failed: {code}")


def timed(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
          mask: torch.Tensor, scale: float, buffers,
          row_groups: int = DEFAULT_ROW_GROUPS,
          tensor: bool = DEFAULT_TENSOR) -> torch.Tensor:
    """Run one configuration instrumented, returning per-CTA SM cycles.

    The result is `[splits, heads, 4]` int64: staging, QK, PV and whole-CTA
    cycle counts for each CTA of the first row tile. The float32 path is
    instrumented at key_tile 40 with a 64-row tile, the tensor path at 48. Not for use in a captured graph.
    """
    lib = library()
    heads, rows, dim = query.shape
    kv_heads, key_count, _ = keys.shape
    accumulator, maximum, denominator = buffers
    splits = accumulator.shape[2]
    record = torch.zeros(splits, heads, 4, dtype=torch.int64, device=query.device)
    code = lib.split_attention_timed(
        ctypes.c_void_p(query.data_ptr()), ctypes.c_void_p(keys.data_ptr()),
        ctypes.c_void_p(values.data_ptr()), ctypes.c_void_p(mask.data_ptr()),
        ctypes.c_void_p(accumulator.data_ptr()), ctypes.c_void_p(maximum.data_ptr()),
        ctypes.c_void_p(denominator.data_ptr()), ctypes.c_void_p(record.data_ptr()),
        rows, heads, kv_heads, key_count, dim,
        keys.stride(0), keys.stride(1), ctypes.c_float(scale), row_groups, int(tensor), _stream())
    if code != 0:
        raise RuntimeError(f"split_attention_timed failed: {code}")
    return record


def noop() -> None:
    """An empty launch, so a benchmark can price one graph node."""
    library().split_attention_noop(_stream())


__all__ = ["DEFAULT_KEY_TILE", "DEFAULT_TENSOR", "TENSOR_KEY_TILE", "DEFAULT_ROW_GROUPS", "DEFAULT_PIECES", "DEFAULT_ROW_TILE",
           "DEFAULT_UNROLL", "build", "combine", "fused_attention",
           "library", "noop", "partials", "splits_for", "timed", "workspace"]
