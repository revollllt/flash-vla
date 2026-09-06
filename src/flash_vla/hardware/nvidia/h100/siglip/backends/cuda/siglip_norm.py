"""Build, load and launch the SigLIP vision LayerNorm kernel.

The kernel is compiled on first use into `<repo>/.cache/cuda_ext/`, keyed on
the source, every tile-library header and the CUTLASS pin, so an edit to any
of them never reuses a stale `.so`. `layer_norm()` is safe during CUDA-graph
capture: it writes a caller-provided buffer and calls no driver entry point.
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
_SRC = _HERE / "kernels" / "siglip_norm.cu"
_REPO = _HERE.parents[7]
_CUTLASS = Path(os.environ.get("CUTLASS_DIR", _REPO / "third_party" / "cutlass"))
_TILE_ROOT = _REPO / "src" / "flash_vla" / "hardware" / "nvidia" / "cuda"

_LIB = None


def _extra_flags() -> list[str]:
    """SIGLIP_NORM_NVCC_DEFINES: space-separated extra nvcc flags (variants)."""
    return os.environ.get("SIGLIP_NORM_NVCC_DEFINES", "").split()


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
    d = _REPO / ".cache" / "cuda_ext" / f"siglip_norm_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(verbose: bool = False) -> Path:
    """Compile the .so if this source hash has not been built yet."""
    out = _build_dir() / "libsiglip_norm.so"
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
        print("[siglip_norm]", " ".join(cmd), flush=True)
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
        lib.siglip_norm_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_float,
                                           ctypes.c_void_p]
        lib.siglip_norm_launch.restype = ctypes.c_int
        lib.siglip_norm_geometry.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2
        lib.siglip_norm_geometry.restype = None
        _check_geometry(lib)
        _LIB = lib
    return _LIB


def geometry_of(lib) -> dict[str, int]:
    """The constants the kernel was compiled with."""
    names = ("dim", "threads")
    values = [ctypes.c_int(0) for _ in names]
    lib.siglip_norm_geometry(*[ctypes.byref(v) for v in values])
    return {n: v.value for n, v in zip(names, values)}


def _check_geometry(lib) -> None:
    """The kernel's compiled-in shape must be the model's.

    Checked once at load rather than per call: the geometry is frozen at
    compile time, so a mismatch is a build error, not a runtime condition, and
    the captured hot path must not carry the test.
    """
    compiled = geometry_of(lib)
    expected = {"dim": geometry.DIM}
    wrong = {k: (compiled[k], v) for k, v in expected.items() if compiled[k] != v}
    if wrong:
        raise RuntimeError(
            f"siglip_norm.cu was compiled for a different vision tower than "
            f"siglip/geometry.py declares (compiled, expected): {wrong}")



def layer_norm(x2, weight, bias, out2, verbose: bool = False):
    """LayerNorm over the feature axis, into `out2`.

    x2    (M, DIM) bf16, contiguous, read
    weight, bias  (DIM,) bf16, contiguous, read
    out2  (M, DIM) bf16, contiguous, written in place; may not alias x2,
          because the kernel reads the whole row before writing any of it only
          within a CTA, and rows are independent -- aliasing is safe in fact,
          but the callers here always pass a separate workspace, so the
          contract stays the stricter one.

    Safe during CUDA-graph capture: no allocation and no driver call.
    """
    lib = library(verbose=verbose)
    rows, dim = x2.shape
    assert dim == geometry.DIM, x2.shape
    assert out2.shape == x2.shape and weight.shape == (dim,) and bias.shape == (dim,)
    assert x2.dtype is torch.bfloat16 and out2.dtype is torch.bfloat16
    assert x2.is_contiguous() and out2.is_contiguous()
    status = lib.siglip_norm_launch(
        ctypes.c_void_p(x2.data_ptr()), ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()), ctypes.c_void_p(out2.data_ptr()),
        ctypes.c_int(rows), ctypes.c_float(geometry.NORM_EPS),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status != 0:
        raise RuntimeError(f"siglip_norm_launch failed with {status}")
    return out2
