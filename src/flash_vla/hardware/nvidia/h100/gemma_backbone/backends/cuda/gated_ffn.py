"""Host side of the persistent gated feed-forward kernel and its RMSNorm.

Mirrors `enc_attn.py`: `build()` compiles `kernels/gated_ffn.cu` into a plain
shared library under the repo's `.cache` (shared filesystem, so a login-node
build is visible to compute nodes) and loads it via ctypes; the library has a
C ABI and takes raw device pointers.

The four TMA tensor maps are encoded once per buffer set and cached, because
`cuTensorMapEncodeTiled` is a driver call and is not safe inside a CUDA-graph
capture. Production buffers are static, so the cache is hit on every step
after the first, which warmup makes.

Tensor contracts (all CUDA, contiguous, bf16): `x` and `x_norm` (rows, K) with
K a multiple of 128 and of 512; `gate_w` and `up_w` (K, F) with F a multiple of
128; `out` (rows, F), written in full. `rows` need not be a multiple of the
128-row tile -- the tensor map's out-of-bounds handling covers the ragged last
M tile on the load side, and the store map clips it.

`rows`, `K` and `F` are runtime arguments, so one shared object serves both
Targets: Pi0.5 runs 968 prefix rows and Pi0 768, at the same K=2048, F=16384.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from flash_vla.hardware.nvidia.native import (
    CUTLASS_DIR,
    CUTLASS_VERSION_HEADER,
    SM90_TILE_HEADERS,
    TILE_INCLUDE_DIR,
    NativeLibrary,
)

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "gated_ffn.cu"

_LIB = None
_MAPS: dict = {}


#: The kernel library: its sources, the sm_90a target and the headers it depends on.
LIBRARY = NativeLibrary(
    name="gated_ffn",
    sources=(_SRC,),
    arch=("-arch=sm_90a",),
    flags=("--expt-relaxed-constexpr", "-Xptxas", "-v"),
    include_dirs=(CUTLASS_DIR / "include", TILE_INCLUDE_DIR),
    headers=(*SM90_TILE_HEADERS, CUTLASS_VERSION_HEADER),
    link_driver=True,
    flags_env="GATED_FFN_NVCC_DEFINES")


def build(verbose: bool = False) -> Path:
    """Compile the library unless this exact build exists; return its path."""
    return LIBRARY.build(verbose=verbose)


def library(verbose: bool = False):
    """The loaded shared library, with its C ABI declared."""
    global _LIB
    if _LIB is None:
        lib = LIBRARY.load(verbose=verbose)
        lib.gated_ffn_geometry.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7
        lib.gated_ffn_geometry.restype = ctypes.c_int
        lib.gated_ffn_make_maps.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [
            ctypes.c_void_p]
        lib.gated_ffn_make_maps.restype = ctypes.c_int
        lib.gated_ffn_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [
            ctypes.c_int] * 4 + [ctypes.c_void_p]
        lib.gated_ffn_launch.restype = ctypes.c_int
        lib.rms_norm_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.rms_norm_launch.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def geometry(verbose: bool = False) -> dict[str, int]:
    """Compiled-in thread count, shared memory, ring depth and tile shape."""
    lib = library(verbose=verbose)
    vals = [ctypes.c_int(0) for _ in range(7)]
    lib.gated_ffn_geometry(*[ctypes.byref(v) for v in vals])
    keys = ("threads", "smem_b", "stages", "bm", "bn", "bk", "map_bytes")
    return dict(zip(keys, (v.value for v in vals)))


def _maps_for(x_norm: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor,
              out: torch.Tensor, verbose: bool):
    """Encoded tensor maps for one buffer set, cached on the pointers.

    Encoding is a driver call, so it must not run inside a graph capture; the
    production buffers are static and this runs once per set, during warmup.
    """
    key = (x_norm.data_ptr(), gate_w.data_ptr(), up_w.data_ptr(), out.data_ptr(),
           x_norm.shape[0], gate_w.shape[0], gate_w.shape[1])
    blob = _MAPS.get(key)
    if blob is None:
        lib = library(verbose=verbose)
        blob = ctypes.create_string_buffer(geometry()["map_bytes"])
        rc = lib.gated_ffn_make_maps(
            ctypes.c_void_p(x_norm.data_ptr()), ctypes.c_void_p(gate_w.data_ptr()),
            ctypes.c_void_p(up_w.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            x_norm.shape[0], gate_w.shape[0], gate_w.shape[1], blob)
        if rc != 0:
            raise RuntimeError(f"gated_ffn_make_maps failed: {rc}")
        _MAPS[key] = blob
    return blob


def rms_norm(x: torch.Tensor, out: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """out = x * rsqrt(mean_K(x^2) + 1e-6), rounded to bf16. One launch, capture-safe."""
    lib = library(verbose=verbose)
    rc = lib.rms_norm_launch(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                             x.shape[0], x.shape[1],
                             ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"rms_norm_launch failed: {rc}")
    return out


def gated_ffn(x_norm: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor,
              out: torch.Tensor, sm_count: int = 0, verbose: bool = False) -> torch.Tensor:
    """out = gelu_tanh(x_norm @ gate_w) * (x_norm @ up_w), one persistent launch.

    Safe during CUDA-graph capture: no allocation and no driver call once the
    maps for these buffers exist.
    """
    lib = library(verbose=verbose)
    blob = _maps_for(x_norm, gate_w, up_w, out, verbose)
    rc = lib.gated_ffn_launch(blob, ctypes.c_void_p(out.data_ptr()),
                              x_norm.shape[0], gate_w.shape[0], gate_w.shape[1], sm_count,
                              ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"gated_ffn_launch failed: {rc}")
    return out
