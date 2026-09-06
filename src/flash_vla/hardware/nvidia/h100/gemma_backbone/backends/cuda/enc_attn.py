"""Host side of the fused prefix (backbone) attention kernel.

`build()` compiles `kernels/enc_attn.cu` into a plain shared library under the
repo's `.cache` (shared filesystem, so a login-node build is visible to
compute nodes) and loads it via ctypes; the library has a C ABI and takes raw
device pointers.

The kernel source is a byte-identical copy of the Pi0.5 Target's
`backends/cuda/kernels/enc_attn.cu`, moved here because both H100 Targets run
the same backbone at the same head geometry. Its cache prefix differs from the
Target copy's on purpose, so the two build separately and a bit-identity run
compares two builds rather than one shared object. The Target copy is retired
by the coordinator at integration; nothing here edits it.

The three TMA tensor maps are encoded once per (Q, K, V) buffer triple and
cached, because `cuTensorMapEncodeTiled` is a driver call and is not safe
inside a CUDA-graph capture. Production buffers are static, so the cache is
hit on every step after the first.

Tensor contracts (all CUDA, contiguous, bf16): Q (m_rows, 256) with
m_rows a multiple of 64 and row = token * heads + head; K, V (keys, 256);
key_mask (keys,) additive, 0 on valid keys and a large finite negative on
padding; out (m_rows, 256), written in full.

`m_rows` and `keys` are runtime arguments, so one shared object serves both
Targets: Pi0.5 runs 7744 query rows against 968 keys (a ragged last key block,
covered by the tensor map's out-of-bounds zero fill and the kernel's own
padding of the mask into shared memory) and Pi0 runs 6144 against 768 (twelve
whole blocks, no ragged tail).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "enc_attn.cu"
_REPO = _HERE.parents[7]
_CUTLASS = Path(os.environ.get("CUTLASS_DIR", _REPO / "third_party" / "cutlass"))
_TILE_ROOT = _REPO / "src" / "flash_vla" / "hardware" / "nvidia" / "cuda"

_LIB = None
_MAPS: dict = {}


def _extra_flags() -> list[str]:
    """GEMMA_ENC_ATTN_NVCC_DEFINES: space-separated extra nvcc flags (kernel variants)."""
    return os.environ.get("GEMMA_ENC_ATTN_NVCC_DEFINES", "").split()


def _cutlass_identity() -> bytes:
    """What CUTLASS this build compiles against, for the cache key.

    `CUTLASS_DIR` is a supported knob (`third_party/README.md`), so the path
    alone is not enough: pointing it at another tree, or unsetting it after a
    custom build, must not reuse the previous `.so`. The version header is
    read rather than the whole tree because it is what changes when the pin
    moves and it costs one small read.
    """
    version = _CUTLASS / "include" / "cutlass" / "version.h"
    payload = version.read_bytes() if version.is_file() else b"missing"
    return str(_CUTLASS.resolve()).encode() + payload


def _build_dir() -> Path:
    # The tile primitive headers are part of the kernel; hash them and the
    # flags so an edit there never reuses a stale .so.
    tile_headers = b"".join(h.read_bytes() for h in
                            sorted((_TILE_ROOT / "tile" / "sm90").glob("*.cuh")))
    tag = hashlib.sha256(_SRC.read_bytes() + tile_headers + _cutlass_identity()
                         + " ".join(_extra_flags()).encode()).hexdigest()[:16]
    d = _REPO / ".cache" / "cuda_ext" / f"gemma_enc_attn_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(verbose: bool = False) -> Path:
    """Compile the .so if this source hash has not been built yet."""
    out = _build_dir() / "libgemma_enc_attn.so"
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
        print("[enc_attn build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    if verbose and r.stderr:
        print(r.stderr, flush=True)
    return out


def library(verbose: bool = False):
    """The loaded shared library, with its C ABI declared."""
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build(verbose=verbose)))
        lib.enc_attn_make_maps.argtypes = [ctypes.c_void_p] * 3 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.enc_attn_make_maps.restype = ctypes.c_int
        lib.enc_attn_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                        ctypes.c_void_p]
        lib.enc_attn_launch.restype = ctypes.c_int
        lib.enc_attn_geometry.argtypes = [ctypes.POINTER(ctypes.c_int)] * 6
        lib.enc_attn_geometry.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def geometry(verbose: bool = False) -> dict[str, int]:
    """Compiled-in thread count, shared-memory size, warpgroups and ring depth."""
    lib = library(verbose=verbose)
    vals = [ctypes.c_int(0) for _ in range(6)]
    lib.enc_attn_geometry(*[ctypes.byref(v) for v in vals])
    keys = ("threads", "smem_b", "nwg", "kdepth", "vdepth", "map_bytes")
    return dict(zip(keys, (v.value for v in vals)))


def _maps_for(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keys: int, verbose: bool):
    """Encoded tensor maps for one buffer triple, cached on the pointers.

    Encoding is a driver call, so it must not run inside a graph capture; the
    production buffers are static and this is called once per triple.
    """
    key = (q.data_ptr(), k.data_ptr(), v.data_ptr(), q.shape[0], keys)
    blob = _MAPS.get(key)
    if blob is None:
        lib = library(verbose=verbose)
        blob = ctypes.create_string_buffer(geometry()["map_bytes"])
        rc = lib.enc_attn_make_maps(ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                                    ctypes.c_void_p(v.data_ptr()), q.shape[0], keys, blob)
        if rc != 0:
            raise RuntimeError(f"enc_attn_make_maps failed: {rc}")
        _MAPS[key] = blob
    return blob


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
              mask: torch.Tensor, out: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """out = softmax(q @ k.T * scale + mask) @ v, one launch.

    `q` is (m_rows, 256) with m_rows a multiple of 64, `k`/`v` are (keys, 256),
    `mask` is (keys,) additive bf16, `out` is (m_rows, 256) and is fully
    written. Safe during CUDA-graph capture: no allocation and no driver call
    on this path once the maps for these buffers exist.
    """
    lib = library(verbose=verbose)
    blob = _maps_for(q, k, v, k.shape[0], verbose)
    rc = lib.enc_attn_launch(blob, ctypes.c_void_p(mask.data_ptr()),
                             ctypes.c_void_p(out.data_ptr()), q.shape[0], k.shape[0],
                             ctypes.c_float(scale),
                             ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"enc_attn_launch failed: {rc}")
    return out
