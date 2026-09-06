"""Build, load and launch the fused SigLIP vision attention kernel.

The kernel is compiled on first use into `<repo>/.cache/cuda_ext/`, keyed on
the source, every tile-library header and the CUTLASS pin, so an edit to any
of them never reuses a stale `.so`. `attention()` is safe during CUDA-graph
capture: it allocates nothing and, because the kernel needs no tensor map,
calls no driver entry point on the launch path.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

from ... import geometry

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "siglip_attn.cu"
_REPO = _HERE.parents[7]
_CUTLASS = Path(os.environ.get("CUTLASS_DIR", _REPO / "third_party" / "cutlass"))
_TILE_ROOT = _REPO / "src" / "flash_vla" / "hardware" / "nvidia" / "cuda"

_LIB = None


def _extra_flags() -> list[str]:
    """SIGLIP_ATTN_NVCC_DEFINES: space-separated extra nvcc flags (variants)."""
    return os.environ.get("SIGLIP_ATTN_NVCC_DEFINES", "").split()


def _cutlass_identity() -> bytes:
    """Which CUTLASS this build compiles against, for the cache key.

    `CUTLASS_DIR` is a supported knob, so the path alone is not enough:
    pointing it at another tree must not reuse the previous `.so`.
    """
    version = _CUTLASS / "include" / "cutlass" / "version.h"
    payload = version.read_bytes() if version.is_file() else b"missing"
    return str(_CUTLASS.resolve()).encode() + payload


def _build_dir() -> Path:
    # The tile primitive headers are part of the kernel; hash them and the
    # flags so an edit there never reuses a stale .so.
    headers = b"".join(h.read_bytes()
                       for h in sorted((_TILE_ROOT / "tile" / "sm90").glob("*.cuh")))
    tag = hashlib.sha256(_SRC.read_bytes() + headers + _cutlass_identity()
                         + " ".join(_extra_flags()).encode()).hexdigest()[:16]
    d = _REPO / ".cache" / "cuda_ext" / f"siglip_attn_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(verbose: bool = False) -> Path:
    """Compile the .so if this source hash has not been built yet."""
    out = _build_dir() / "libsiglip_attn.so"
    if out.exists():
        return out
    cuda_home = os.environ.get("CUDA_HOME", "/data/apps/cuda/13.1")
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
        "-arch=sm_90a", "--expt-relaxed-constexpr", "-Xptxas", "-v",
        *_extra_flags(),
        f"-I{_CUTLASS}/include", f"-I{_TILE_ROOT}",
        "-o", str(out), str(_SRC),
        f"-L{cuda_home}/lib64/stubs", "-lcuda",
    ]
    if verbose:
        print("[siglip_attn]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if verbose or proc.returncode != 0:
        print(proc.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed ({proc.returncode}) building {_SRC}")
    return out


def library(verbose: bool = False):
    """The loaded .so, with the C ABI declared. Memoized per process."""
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build(verbose=verbose)))
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
