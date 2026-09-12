"""Host side of this Target's hand-written pointwise stages.

`build()` compiles `kernels/expert_pointwise.cu` into a plain shared library
under the repo's `.cache` and loads it through ctypes: the library has a C ABI
and takes raw device pointers, so no torch extension machinery is involved and
every launch is safe inside a CUDA-graph capture.

The toolkit comes from the environment (`CUDA_HOME`, else `nvcc` on `PATH`,
else `FLASH_VLA_NVCC`) rather than a baked path -- machine paths belong in
user-local configuration, not in the repository.

Target is `sm_120f`, the family-specific form: it is identical to `sm_120a`
across all 51 instructions this project probed and covers the whole GB20x
family rather than one die [isa.target.a_required].
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "expert_pointwise.cu"
_ATTN_SRC = _HERE / "kernels" / "expert_attention.cu"
_REPO = _HERE.parents[7]
_ARCH = "sm_120f"

_LIB = None


def _nvcc() -> str:
    explicit = os.environ.get("FLASH_VLA_NVCC")
    if explicit:
        return explicit
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / "nvcc").is_file():
        return str(Path(home) / "bin" / "nvcc")
    found = shutil.which("nvcc")
    if found:
        return found
    raise RuntimeError(
        "no nvcc: set CUDA_HOME to a CUDA toolkit, put nvcc on PATH, or set "
        "FLASH_VLA_NVCC")


def build(verbose: bool = False) -> Path:
    """Compile the shared library if this source and toolchain have not been built."""
    nvcc = _nvcc()
    tag = hashlib.sha256(_SRC.read_bytes() + _ATTN_SRC.read_bytes()
                         + nvcc.encode() + _ARCH.encode()).hexdigest()[:16]
    directory = _REPO / ".cache" / "cuda_ext" / f"rtx5090_pi0_pointwise_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "libexpert_pointwise.so"
    if out.exists():
        return out
    command = [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-gencode", f"arch=compute_{_ARCH[3:]},code={_ARCH}",
               "-o", str(out), str(_SRC), str(_ATTN_SRC)]
    if verbose:
        print("[rtx5090 pi0 pointwise build]", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{result.stdout}\n{result.stderr}")
    return out


def library(verbose: bool = False):
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build(verbose=verbose)))
        lib.rms_norm_launch.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [
            ctypes.c_void_p]
        lib.rms_norm_launch.restype = ctypes.c_int
        lib.rope_scatter_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [
            ctypes.c_void_p]
        lib.rope_scatter_launch.restype = ctypes.c_int
        lib.gelu_mul_launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_longlong] + [
            ctypes.c_void_p]
        lib.gelu_mul_launch.restype = ctypes.c_int
        lib.expert_attention_launch.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        lib.expert_attention_launch.restype = ctypes.c_int
        lib.expert_masked_softmax_launch.argtypes = [ctypes.c_void_p] * 2 + [
            ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        lib.expert_masked_softmax_launch.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _check(rc: int, name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{name} failed: cudaError {rc}")


def rms_norm(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[r] = x[r] * rsqrt(mean(x[r]^2) + 1e-6), bf16 in and out, fp32 inside.

    `x` and `out` are (rows, cols) contiguous bf16 on CUDA; cols must be a
    multiple of 8 so the row is read in 128-bit chunks. Writes `out` in place;
    safe during CUDA-graph capture.
    """
    rows, cols = x.shape
    _check(library().rms_norm_launch(x.data_ptr(), out.data_ptr(), rows, cols,
                                     _stream()), "rms_norm")
    return out


def rope_scatter(packed: torch.Tensor, rope: torch.Tensor, q: torch.Tensor,
                 k: torch.Tensor, v: torch.Tensor) -> None:
    """Split a packed projection into Q/K/V, rotating Q and K.

    `packed` is (rows, q_dim + 2 * head_dim) bf16; `rope` is (rows, head_dim)
    holding interleaved [cos, sin] pairs. `q` is (rows * heads, head_dim) or any
    view of (rows, q_dim); `k` and `v` are (rows, head_dim). All contiguous bf16
    on CUDA, written in place; safe during CUDA-graph capture.
    """
    rows, head_dim = k.shape
    q_dim = packed.shape[1] - 2 * head_dim
    _check(library().rope_scatter_launch(
        packed.data_ptr(), rope.data_ptr(), q.data_ptr(), k.data_ptr(),
        v.data_ptr(), rows, q_dim, head_dim, _stream()), "rope_scatter")


def gelu_mul(gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out = gelu_tanh(gate) * up, element-wise.

    All three are contiguous bf16 CUDA tensors of the same shape, with a total
    element count that is a multiple of 8. Writes `out` in place; safe during
    CUDA-graph capture.
    """
    _check(library().gelu_mul_launch(gate.data_ptr(), up.data_ptr(),
                                     out.data_ptr(), gate.numel(), _stream()),
           "gelu_mul")
    return out


#: Query rows per CTA. Swept by `lab/sm120/pi0_attention_bench.py`; 16 was the
#: best of 8/16/32 at Pi0's 408 x 819 x 256 shape.
ATTENTION_BLOCK_M = 16


def expert_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     out: torch.Tensor, *, heads: int, prefix: int,
                     block_m: int = 0) -> torch.Tensor:
    """out = softmax(mask(q @ k^T * scale)) @ v, multi-query, one launch.

    `q` and `out` are (queries, head_dim), `k` and `v` are (keys, head_dim), all
    contiguous bf16 on CUDA. The mask keeps key j for flat row r when
    `r >= heads or j <= prefix`. `out` may alias `q`. Written in place; safe
    during CUDA-graph capture.
    """
    queries, head_dim = q.shape
    keys = k.shape[0]
    _check(library().expert_attention_launch(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), queries, keys,
        head_dim, heads, prefix, float(head_dim ** -0.5),
        block_m or ATTENTION_BLOCK_M, _stream()), "expert_attention")
    return out


def expert_masked_softmax(scores: torch.Tensor, probs: torch.Tensor, *,
                          heads: int, prefix: int, scale: float) -> torch.Tensor:
    """probs[r] = softmax(mask(r, :) ? scores[r] * scale : -inf), one launch.

    `scores` and `probs` are (queries, keys) contiguous bf16 on CUDA and may be
    the same tensor. The mask keeps key j for flat row r when `r >= heads or
    j <= prefix`. Written in place; safe during CUDA-graph capture.
    """
    queries, keys = scores.shape
    _check(library().expert_masked_softmax_launch(
        scores.data_ptr(), probs.data_ptr(), queries, keys, heads, prefix,
        float(scale), _stream()), "expert_masked_softmax")
    return probs


__all__ = ["ATTENTION_BLOCK_M", "build", "expert_attention",
           "expert_masked_softmax", "gelu_mul",
           "library", "rms_norm", "rope_scatter"]
