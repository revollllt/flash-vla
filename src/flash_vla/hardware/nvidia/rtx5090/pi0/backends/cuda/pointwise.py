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
_MMA_SRC = _HERE / "kernels" / "expert_attention_mma.cu"
_QKV_SRC = _HERE / "kernels" / "expert_qkv.cu"
_GEMM_SRC = _HERE / "kernels" / "tiled_gemm.cu"
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
                         + _MMA_SRC.read_bytes() + _QKV_SRC.read_bytes() + _GEMM_SRC.read_bytes() + nvcc.encode()
                         + _ARCH.encode()).hexdigest()[:16]
    directory = _REPO / ".cache" / "cuda_ext" / f"rtx5090_pi0_pointwise_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "libexpert_pointwise.so"
    if out.exists():
        return out
    command = [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-gencode", f"arch=compute_{_ARCH[3:]},code={_ARCH}",
               "-o", str(out), str(_SRC), str(_ATTN_SRC), str(_MMA_SRC), str(_QKV_SRC), str(_GEMM_SRC)]
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
        lib.tiled_gemm_launch.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.tiled_gemm_launch.restype = ctypes.c_int
        lib.expert_qkv_launch.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.expert_qkv_launch.restype = ctypes.c_int
        lib.expert_attention_launch.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        lib.expert_attention_launch.restype = ctypes.c_int
        lib.expert_masked_softmax_launch.argtypes = [ctypes.c_void_p] * 2 + [
            ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        lib.expert_masked_softmax_launch.restype = ctypes.c_int
        lib.expert_attention_mma_workspace.argtypes = [
            ctypes.c_int] * 2 + [ctypes.POINTER(ctypes.c_longlong)]
        lib.expert_attention_mma_workspace.restype = ctypes.c_int
        lib.expert_attention_mma_launch.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_void_p]
        lib.expert_attention_mma_launch.restype = ctypes.c_int
        lib.layer_norm_launch.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int] * 2 + [ctypes.c_void_p]
        lib.layer_norm_launch.restype = ctypes.c_int
        lib.gelu_launch.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                    ctypes.c_void_p]
        lib.gelu_launch.restype = ctypes.c_int
        lib.gelu_mul_packed_launch.argtypes = [ctypes.c_void_p] * 2 + [
            ctypes.c_int] * 2 + [ctypes.c_void_p]
        lib.gelu_mul_packed_launch.restype = ctypes.c_int
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


#: Shapes `tiled_gemm` accepts: its tile is 64 x 64 x 32 and it has no tail
#: path, by design -- these are fixed model shapes.
TILE_M, TILE_N, TILE_K = 64, 64, 32


def tiles(m: int, k: int, n: int) -> bool:
    """Whether `tiled_gemm` can take this shape."""
    return m % TILE_M == 0 and n % TILE_N == 0 and k % TILE_K == 0


def tiled_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
               res: torch.Tensor | None = None) -> torch.Tensor:
    """c = (res +) a @ b, bf16 in and out, fp32 accumulate.

    `a` is (m, k), `b` is (k, n), `c` is (m, n), all contiguous bf16 on CUDA.
    `res` is optional and MAY ALIAS `c`. Writes `c` in place; safe during
    CUDA-graph capture. Call `tiles()` first: a shape that does not tile is
    rejected rather than handled slowly.
    """
    m, k = a.shape
    n = b.shape[1]
    _check(library().tiled_gemm_launch(
        a.data_ptr(), b.data_ptr(), 0 if res is None else res.data_ptr(),
        c.data_ptr(), m, k, n, _stream()), "tiled_gemm")
    return c


def expert_qkv(x: torch.Tensor, weight: torch.Tensor, rope: torch.Tensor,
               q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """RMS-scale x, project to QKV, rotate and scatter -- one launch.

    `x` is (rows, dim) bf16 and `weight` is (dim, q_dim + 2 * head_dim) bf16;
    `rope` is (rows, head_dim) holding interleaved [cos, sin] pairs. `q` is any
    contiguous view of (rows, q_dim), `k` and `v` are (rows, head_dim). All on
    CUDA, written in place; safe during CUDA-graph capture.

    The normalized activations are never materialized: Pi0's expert RMSNorm has
    no learnable gain, so it is a per-row scalar that commutes with the
    projection and is applied in the kernel's epilogue.
    """
    rows, dim = x.shape
    n = weight.shape[1]
    head_dim = k.shape[-1]
    _check(library().expert_qkv_launch(
        x.data_ptr(), weight.data_ptr(), rope.data_ptr(), q.data_ptr(),
        k.data_ptr(), v.data_ptr(), rows, dim, n, n - 2 * head_dim, head_dim,
        _stream()), "expert_qkv")


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


def gelu_mul_packed(packed: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[r, c] = gelu_tanh(packed[r, c]) * packed[r, half + c].

    `packed` is (rows, 2 * half) contiguous bf16 on CUDA -- the gate and up
    projections as the two halves of one row -- and `out` is (rows, half).
    Written in place; safe during CUDA-graph capture.
    """
    rows, width = packed.shape
    _check(library().gelu_mul_packed_launch(packed.data_ptr(), out.data_ptr(),
                                            rows, width // 2, _stream()),
           "gelu_mul_packed")
    return out


def gelu_(x: torch.Tensor) -> torch.Tensor:
    """x = gelu_tanh(x), in place. Contiguous bf16 CUDA, numel a multiple of 8.

    Safe during CUDA-graph capture.
    """
    _check(library().gelu_launch(x.data_ptr(), x.numel(), _stream()), "gelu")
    return x


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
               out: torch.Tensor) -> torch.Tensor:
    """out[r] = (x[r] - mean) * rsqrt(var + 1e-5) * w + b, bf16 in and out.

    `x` and `out` are (rows, cols) contiguous bf16 on CUDA and may be the same
    tensor; `w` and `b` are (cols,) bf16. cols must be a multiple of 8. Written
    in place; safe during CUDA-graph capture.
    """
    rows, cols = x.shape
    _check(library().layer_norm_launch(x.data_ptr(), w.data_ptr(), b.data_ptr(),
                                       out.data_ptr(), rows, cols, _stream()),
           "layer_norm")
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


#: Split-KV partials, per (queries, splits). 408 queries is 26 mma M tiles and
#: therefore 26 CTAs; splitting the key axis is the only place more parallelism
#: can come from, and the partials it costs are fp32 and stay in L2.
_ATTENTION_WORKSPACE: dict[tuple[int, int], torch.Tensor] = {}


def _attention_workspace(queries: int, splits: int, device):
    """The fp32 partial buffer for this shape, or None inside a capture."""
    if splits <= 1:
        return torch.empty(0)
    got = _ATTENTION_WORKSPACE.get((queries, splits))
    if got is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        floats = ctypes.c_longlong(0)
        _check(library().expert_attention_mma_workspace(
            queries, splits, ctypes.byref(floats)), "attention_workspace")
        got = torch.empty(floats.value, dtype=torch.float32, device=device)
        _ATTENTION_WORKSPACE[(queries, splits)] = got
    return got


def expert_attention_mma(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         out: torch.Tensor, *, heads: int, prefix: int,
                         splits: int = 1) -> bool:
    """out = softmax(mask(q @ k^T * scale)) @ v on the tensor core.

    `q` and `out` are (queries, 256), `k` and `v` are (keys, 256), contiguous
    bf16 on CUDA. `out` must NOT alias `q`: the accumulator is written per tile.
    The mask keeps key j for flat row r when `r >= heads or j <= prefix`.

    `splits` divides the KEY axis across that many extra CTAs, each producing an
    unnormalized partial that a second kernel merges. Safe during CUDA-graph
    capture once the workspace exists; returns False without touching `out` if
    it is missing inside a capture, so the caller can fall back.
    """
    queries, head_dim = q.shape
    ws = _attention_workspace(queries, splits, q.device)
    if ws is None:
        return False
    _check(library().expert_attention_mma_launch(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), queries,
        k.shape[0], head_dim, heads, prefix, float(head_dim ** -0.5),
        ws.data_ptr() if splits > 1 else 0, splits, _stream()),
        "expert_attention_mma")
    return True


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
           "expert_attention_mma", "expert_masked_softmax", "gelu_", "gelu_mul", "gelu_mul_packed",
           "layer_norm",
           "library", "rms_norm", "rope_scatter"]
