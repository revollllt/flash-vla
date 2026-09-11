"""Host side of the action expert's weight-stationary skinny GEMM.

`build()` compiles `kernels/skinny_gemm.cu` into a plain shared library under
the repo's `.cache` and loads it through ctypes: the library has a C ABI and
takes raw device pointers, so no torch extension machinery is involved and
every launch is safe inside a CUDA-graph capture.

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
_SRC = _HERE / "kernels" / "skinny_gemm.cu"
_REPO = _HERE.parents[7]
_TILE_ROOT = _REPO / "src" / "flash_vla" / "hardware" / "nvidia" / "cuda"
_DEFAULT_NVCC = "/data/apps/cuda/12.6/bin/nvcc"

_LIB = None


def _nvcc() -> str:
    explicit = os.environ.get("LINGBOT_NVCC")
    if explicit:
        return explicit
    return _DEFAULT_NVCC if Path(_DEFAULT_NVCC).is_file() else "nvcc"


def _defines() -> list[str]:
    """LINGBOT_SKINNY_DEFINES: extra nvcc flags, for the probe variants."""
    return os.environ.get("LINGBOT_SKINNY_DEFINES", "").split()


def _cutlass() -> Path:
    """The CUTLASS tree to compile against.

    A git worktree does not check out submodules, so fall back to the primary
    checkout's tree rather than failing the build in an experiment worktree.
    """
    explicit = os.environ.get("CUTLASS_DIR")
    if explicit:
        return Path(explicit)
    local = _REPO / "third_party" / "cutlass"
    if (local / "include" / "cutlass" / "version.h").is_file():
        return local
    return Path("/data/user/jzou521/codes/cuda/flash-vla/third_party/cutlass")


def _identity() -> bytes:
    """What this build depends on, so an edit never reuses a stale .so."""
    headers = b"".join(h.read_bytes()
                       for h in sorted((_TILE_ROOT / "tile" / "sm90").glob("*.cuh")))
    version = _cutlass() / "include" / "cutlass" / "version.h"
    cutlass = version.read_bytes() if version.is_file() else b"missing"
    return (_SRC.read_bytes() + headers + cutlass + _nvcc().encode()
            + " ".join(_defines()).encode())


def build(verbose: bool = False) -> Path:
    """Compile the shared library if this source and toolchain have not been built."""
    tag = hashlib.sha256(_identity()).hexdigest()[:16]
    directory = _REPO / ".cache" / "cuda_ext" / f"lingbot_skinny_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "libskinny.so"
    if out.exists():
        return out
    cuda_home = str(Path(_nvcc()).parents[1])
    command = [_nvcc(), "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-arch=sm_90a", "--expt-relaxed-constexpr", "-Xptxas", "-v",
               *_defines(),
               f"-I{_cutlass()}/include", f"-I{_TILE_ROOT}",
               "-o", str(out), str(_SRC),
               f"-L{cuda_home}/lib64/stubs", "-lcuda"]
    if verbose:
        print("[skinny build]", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if verbose or result.returncode != 0:
        print(result.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed ({result.returncode}) building {_SRC}")
    return out


def library(verbose: bool = False):
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build(verbose=verbose)))
        lib.skinny_gemm_launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 7 + [
            ctypes.c_void_p]
        lib.skinny_gemm_launch.restype = ctypes.c_int
        lib.skinny_gemm_tiles.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.skinny_gemm_tiles.restype = ctypes.c_int
        lib.skinny_gemm_workspace_floats.argtypes = [ctypes.c_int] * 4
        lib.skinny_gemm_workspace_floats.restype = ctypes.c_longlong
        lib.skinny_gemm_residual_launch.argtypes = [ctypes.c_void_p] * 7 + [
            ctypes.c_int] * 7 + [ctypes.c_void_p]
        lib.skinny_gemm_residual_launch.restype = ctypes.c_int
        lib.skinny_gemm_silu_launch.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int] * 6 + [ctypes.c_void_p]
        lib.skinny_gemm_silu_launch.restype = ctypes.c_int
        lib.skinny_gemm_adarms_launch.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_int] * 6 + [ctypes.c_float, ctypes.c_void_p]
        lib.skinny_gemm_adarms_launch.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def n_tiles(n: int, tile_n: int) -> int:
    """How many N tiles a `tile_n` tiling of width `n` produces."""
    return library().skinny_gemm_tiles(n, tile_n)


def make_workspace(m: int, n: int, tile_n: int, k_split: int,
                   device) -> tuple[torch.Tensor, torch.Tensor]:
    """The split-K partial buffer and the arrival counters.

    The partials are rewritten in full every launch so the buffer is left
    uninitialised; only the counters start at zero, and the kernel restores
    them, so one allocation serves every replay of a captured graph.
    """
    floats = library().skinny_gemm_workspace_floats(m, n, tile_n, k_split)
    partials = torch.empty(floats, dtype=torch.float32, device=device)
    # Two counters per N tile: publications, then completions.
    counters = torch.zeros(2 * n_tiles(n, tile_n), dtype=torch.int32, device=device)
    return partials, counters


def linear(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
           bias: torch.Tensor | None = None, tile_n: int = 64, depth: int = 4,
           k_split: int = 1, producer: int = 1,
           workspace: torch.Tensor | None = None,
           counters: torch.Tensor | None = None) -> torch.Tensor:
    """`out = x @ weight.T (+ bias)` for a tall-thin `x`, bf16 with fp32 accumulation.

    `x` is `[m, k]` with `m <= 64`, `weight` is `[n, k]` (torch Linear's
    layout) and `out` is `[m, n]`; all three are bf16, contiguous and on the
    same CUDA device, and `out` is written in full. `bias` is `[n]` bf16 or
    None. `k` must be a multiple of 64. `producer` selects the ring's feeder
    and the split-K reduction: 0 is cp.async from the math warps, 1 is TMA, 2
    is TMA with a cluster of two sharing the activation tile (`k_split` must be
    1), and 3 makes the splits of an N tile a cluster that reduces through
    distributed shared memory (`k_split` must be 2, 4 or 8). `k_split` > 1
    needs the pair from `make_workspace` -- zero on entry, left zero again --
    for every producer except 3, which keeps its partials in shared memory and
    takes neither. One launch; safe during CUDA-graph capture once this
    `x`/`weight` pair has been passed once outside capture, which builds the
    TMA tensor maps.
    """
    lib = library()
    m, k = x.shape
    n = weight.shape[0]
    code = lib.skinny_gemm_launch(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr() if bias is not None else None),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr() if workspace is not None else None),
        ctypes.c_void_p(counters.data_ptr() if counters is not None else None),
        m, n, k, tile_n, depth, k_split, producer,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if code != 0:
        raise RuntimeError(f"skinny_gemm_launch failed: {code}")
    return out


def linear_residual(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor,
                    out: torch.Tensor, bias: torch.Tensor | None = None,
                    tile_n: int = 64, depth: int = 6, k_split: int = 8) -> torch.Tensor:
    """`out = x @ weight.T (+ bias) + residual`, bf16 with fp32 accumulation.

    Shapes and dtypes are `linear`'s; `residual` is `[m, n]` bf16, is read and
    never written, and `out` may alias it. The GEMM result is rounded to bf16
    before the residual is added, which is what the separate GEMM and
    `ada_rms_add` pair did, so the fused and unfused chains agree bit for bit.
    Only the cluster reduction (producer 3) carries this, so `k_split` must be
    2, 4 or 8. One launch, capture safe on the same terms as `linear`.
    """
    lib = library()
    m, k = x.shape
    n = weight.shape[0]
    code = lib.skinny_gemm_residual_launch(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr() if bias is not None else None),
        ctypes.c_void_p(residual.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        None, None, m, n, k, tile_n, depth, k_split, 3,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if code != 0:
        raise RuntimeError(f"skinny_gemm_residual_launch failed: {code}")
    return out


def adarms_linear(x: torch.Tensor, weight: torch.Tensor, norm_weight: torch.Tensor,
                  gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor,
                  bias: torch.Tensor | None = None, epsilon: float = 1e-6,
                  tile_n: int = 32, depth: int = 6, threads: int = 256,
                  normed_out: torch.Tensor | None = None) -> torch.Tensor:
    """`out = adarms(x) @ weight.T (+ bias)`, with the normalization fused in.

    `x` is `[m, k]` bf16 with `m <= 64` and `k == 768`: the already-summed
    activation, since the residual add belongs to the producing GEMM's epilogue
    (`linear_residual`). `norm_weight`, `gamma` and `beta` are `[k]` bf16 and
    `epsilon` matches the reference. `weight` is `[n, k]`, `out` is `[m, n]`,
    all bf16, contiguous, same device; `out` is written in full. The
    normalization is `ada_rms_add_kernel`'s arithmetic, reduction order
    included. Passing `normed_out` (`[m, k]` bf16) runs the prologue only and
    writes the normalized activation there instead of the GEMM result, which is
    how the fused arithmetic is compared against the kernel it replaces. One
    launch, capture safe on the same terms as `linear`.
    """
    lib = library()
    m, k = x.shape
    n = weight.shape[0]
    code = lib.skinny_gemm_adarms_launch(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(norm_weight.data_ptr()), ctypes.c_void_p(gamma.data_ptr()),
        ctypes.c_void_p(beta.data_ptr()),
        ctypes.c_void_p(bias.data_ptr() if bias is not None else None),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(normed_out.data_ptr() if normed_out is not None else None),
        m, n, k, tile_n, depth, threads,
        ctypes.c_float(epsilon),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if code != 0:
        raise RuntimeError(f"skinny_gemm_adarms_launch failed: {code}")
    return out


def silu_linear(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
                bias: torch.Tensor | None = None, tile_n: int = 64,
                depth: int = 6, k_split: int = 2) -> torch.Tensor:
    """`out = silu(gate) * up` where `[gate | up] = x @ weight.T (+ bias)`.

    `x` is `[m, k]` bf16 with `m <= 64`, `weight` is `[n, k]` with `n` even --
    the packed gate/up weight -- and `out` is `[m, n // 2]` bf16, written in
    full. `bias` is `[n]` bf16 or None and applies to both halves before the
    activation. The activation is rounded to bf16 before multiplying `up`, as
    `silu_multiply_kernel` does, so the fused and unfused chains agree bit for
    bit. `tile_n` is the paired tile: a CTA owns `tile_n // 2` gate columns and
    their `tile_n // 2` up partners. One launch, capture safe on the same terms
    as `linear`.
    """
    lib = library()
    m, k = x.shape
    n = weight.shape[0]
    code = lib.skinny_gemm_silu_launch(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr() if bias is not None else None),
        ctypes.c_void_p(out.data_ptr()), m, n, k, tile_n, depth, k_split,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if code != 0:
        raise RuntimeError(f"skinny_gemm_silu_launch failed: {code}")
    return out


__all__ = ["adarms_linear", "build", "library", "linear", "linear_residual",
           "make_workspace", "n_tiles", "silu_linear"]
