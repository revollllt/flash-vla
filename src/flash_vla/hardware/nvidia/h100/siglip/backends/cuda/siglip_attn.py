"""Build, load and launch the fused SigLIP vision attention kernel.

The kernel is compiled on first use into `<repo>/.cache/cuda_ext/`, keyed on
the source, every tile-library header and the CUTLASS pin, so an edit to any
of them never reuses a stale `.so`. `attention()` is safe during CUDA-graph
capture: it allocates nothing and, because the kernel needs no tensor map,
calls no driver entry point on the launch path.
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

from ... import geometry

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "siglip_attn.cu"

_LIB = None


#: The kernel library: its sources, the sm_90a target and the headers it depends on.
LIBRARY = NativeLibrary(
    name="siglip_attn",
    sources=(_SRC,),
    arch=("-arch=sm_90a",),
    flags=("--expt-relaxed-constexpr", "-Xptxas", "-v"),
    include_dirs=(CUTLASS_DIR / "include", TILE_INCLUDE_DIR),
    headers=(*SM90_TILE_HEADERS, CUTLASS_VERSION_HEADER),
    link_driver=True,
    flags_env="SIGLIP_ATTN_NVCC_DEFINES")


def build(verbose: bool = False) -> Path:
    """Compile the library unless this exact build exists; return its path."""
    return LIBRARY.build(verbose=verbose)


def library(verbose: bool = False):
    """The loaded .so, with the C ABI declared. Memoized per process."""
    global _LIB
    if _LIB is None:
        lib = LIBRARY.load(verbose=verbose)
        lib.siglip_attn_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_float,
                                           ctypes.c_void_p]
        lib.siglip_attn_launch.restype = ctypes.c_int
        lib.siglip_attn_geometry.argtypes = [ctypes.POINTER(ctypes.c_int)] * 9
        lib.siglip_attn_geometry.restype = None
        _check_geometry(lib)
        _LIB = lib
    return _LIB


def geometry_of(lib) -> dict[str, int]:
    """The constants the kernel was compiled with."""
    names = ("tokens", "heads", "dh", "dh_pad", "bm", "bkk", "stages", "threads",
             "smem_b")
    values = [ctypes.c_int(0) for _ in names]
    lib.siglip_attn_geometry(*[ctypes.byref(v) for v in values])
    return {n: v.value for n, v in zip(names, values)}


def _check_geometry(lib) -> None:
    """The kernel's compiled-in shape must be the model's.

    Checked once at load rather than per call: the geometry is frozen at
    compile time, so a mismatch is a build error, not a runtime condition, and
    the captured hot path must not carry the test.
    """
    compiled = geometry_of(lib)
    expected = {"tokens": geometry.TOKENS, "heads": geometry.HEADS,
                "dh": geometry.HEAD_DIM}
    wrong = {k: (compiled[k], v) for k, v in expected.items() if compiled[k] != v}
    if wrong:
        raise RuntimeError(
            f"siglip_attn.cu was compiled for a different vision tower than "
            f"siglip/geometry.py declares (compiled, expected): {wrong}")


def attention(qkv: torch.Tensor, out: torch.Tensor, verbose: bool = False) -> None:
    """Fused multi-head self-attention over the packed QKV buffer.

    qkv  (VIEWS, TOKENS, 3*DIM) bf16, contiguous, read
    out  (VIEWS, TOKENS, DIM)   bf16, contiguous, written in place

    The scale is DH**-0.5, which is what `scaled_dot_product_attention`
    defaults to and therefore what the production route computes.
    Safe during CUDA-graph capture: no allocation and no driver call.
    """
    lib = library(verbose=verbose)
    views = qkv.shape[0]
    assert qkv.shape == (views, geometry.TOKENS, geometry.QKV_DIM), qkv.shape
    assert out.shape == (views, geometry.TOKENS, geometry.DIM), out.shape
    assert qkv.dtype is torch.bfloat16 and out.dtype is torch.bfloat16
    assert qkv.is_contiguous() and out.is_contiguous()
    status = lib.siglip_attn_launch(
        ctypes.c_void_p(qkv.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(views), ctypes.c_float(geometry.HEAD_DIM ** -0.5),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status != 0:
        raise RuntimeError(f"siglip_attn_launch failed with {status}")
