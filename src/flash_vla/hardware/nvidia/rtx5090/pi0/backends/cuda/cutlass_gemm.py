"""Host side of the route's CUTLASS stream-K GEMMs.

`kernels/cutlass_gemm.cu` compiles into its own shared library, separate from
the pointwise one, because CUTLASS templates are slow to compile and the
pointwise kernels are edited far more often; sharing a translation unit would
put a multi-minute rebuild behind every one-line kernel change.

Planning is split from launching. Stream-K carries a barrier workspace, so a
`Plan` does the host-side setup once -- during warmup, before capture -- and
holds the device buffers it was planned against. `run()` is a bare launch and
is safe inside a CUDA-graph capture; a replay needs no re-initialization
because the kernel resets its own barriers.

A plan is therefore bound to the POINTERS it was built with. The runner's
buffers are stable for the life of a graph, which is what makes this sound; the
cache key includes every pointer so a different buffer builds a different plan
rather than silently writing to the wrong place.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "cutlass_gemm.cu"
_REPO = _HERE.parents[7]
_ARCH = "sm_120a"

_LIB = None


def _nvcc() -> str:
    """Same resolution order as the pointwise build: CUDA_HOME, PATH, override."""
    explicit = os.environ.get("FLASH_VLA_NVCC")
    if explicit:
        return explicit
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin" / "nvcc").is_file():
        return str(Path(home) / "bin" / "nvcc")
    import shutil
    found = shutil.which("nvcc")
    if found:
        return found
    raise RuntimeError(
        "no nvcc: set CUDA_HOME to a CUDA toolkit, put nvcc on PATH, or set "
        "FLASH_VLA_NVCC")


def _cutlass() -> Path:
    """The CUTLASS tree to compile against.

    `CUTLASS_DIR` first, matching the H100 target's skinny GEMM, then the
    repo's own submodule, then the skills checkout this machine carries.
    """
    explicit = os.environ.get("CUTLASS_DIR")
    if explicit:
        return Path(explicit)
    for candidate in (_REPO / "third_party" / "cutlass",
                      Path("/home/ubuntu/agent-gpu-skills/third_party/cutlass")):
        if (candidate / "include" / "cutlass" / "version.h").is_file():
            return candidate
    raise RuntimeError("no CUTLASS tree: set CUTLASS_DIR")


def _identity() -> bytes:
    """What this build depends on, so an edit never reuses a stale .so."""
    version = _cutlass() / "include" / "cutlass" / "version.h"
    return (_SRC.read_bytes() + version.read_bytes() + _nvcc().encode()
            + _ARCH.encode())


def build(verbose: bool = False) -> Path:
    """Compile the shared library if this source and toolchain have not been built."""
    tag = hashlib.sha256(_identity()).hexdigest()[:16]
    directory = _REPO / ".cache" / "cuda_ext" / f"rtx5090_pi0_cutlass_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "libcutlass_gemm.so"
    if out.exists():
        return out
    command = [_nvcc(), "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "--expt-relaxed-constexpr",
               "-gencode", f"arch=compute_{_ARCH[3:]},code={_ARCH}",
               f"-I{_cutlass()}/include",
               "-o", str(out), str(_SRC)]
    if verbose or os.environ.get("FLASH_VLA_BUILD_VERBOSE"):
        print("[rtx5090 pi0 cutlass build]", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{result.stdout}\n{result.stderr}")
    return out


def library(verbose: bool = False):
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build(verbose=verbose)))
        lib.cutlass_gemm_config_count.argtypes = []
        lib.cutlass_gemm_config_count.restype = ctypes.c_int
        lib.cutlass_gemm_workspace.argtypes = [ctypes.c_int] * 4 + [
            ctypes.POINTER(ctypes.c_longlong)]
        lib.cutlass_gemm_workspace.restype = ctypes.c_int
        lib.cutlass_gemm_plan.argtypes = ([ctypes.c_int] * 5 + [ctypes.c_float]
                                          + [ctypes.c_void_p] * 5
                                          + [ctypes.POINTER(ctypes.c_void_p)])
        lib.cutlass_gemm_plan.restype = ctypes.c_int
        lib.cutlass_gemm_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cutlass_gemm_run.restype = ctypes.c_int
        lib.cutlass_gemm_destroy.argtypes = [ctypes.c_void_p]
        lib.cutlass_gemm_destroy.restype = ctypes.c_int
        _LIB = lib
    return _LIB


#: The kernel's answer for "this tile does not apply to this shape".
CANNOT_IMPLEMENT = 1000


class Unsupported(RuntimeError):
    """This tile cannot serve this shape. An answer while choosing, not a fault."""


def _check(status: int, what: str) -> None:
    if status >= CANNOT_IMPLEMENT:
        raise Unsupported(f"{what}: tile does not apply to this shape "
                          f"(cutlass status {status - CANNOT_IMPLEMENT})")
    if status != 0:
        raise RuntimeError(f"{what} failed: cuda error {status}")


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


#: Planned operators, keyed by everything that changes what the launch does.
_PLANS: dict[tuple, tuple[int, torch.Tensor]] = {}


def plan(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, *, config: int,
         c: torch.Tensor | None = None, beta: float = 0.0,
         broadcast_c: bool = False):
    """A planned stream-K operator for these buffers, or None inside a capture.

    `a` is (m, k) and `b` is (k, n), both contiguous bf16; `d` is (m, n). `c`
    is the epilogue's source: a full (m, n) residual, which MAY ALIAS `d`, or a
    length-n bias when `broadcast_c` is set, which is passed with a zero
    leading dimension so every row reads the same values.

    Building a plan allocates, so a miss during a capture returns None and the
    caller falls back. The runner warms every call site before capturing.
    """
    m, k = a.shape
    n = b.shape[1]
    ldc = 0 if broadcast_c else n
    key = (config, m, k, n, ldc, beta, a.data_ptr(), b.data_ptr(),
           0 if c is None else c.data_ptr(), d.data_ptr())
    got = _PLANS.get(key)
    if got is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        lib = library()
        nbytes = ctypes.c_longlong(0)
        _check(lib.cutlass_gemm_workspace(config, m, k, n,
                                          ctypes.byref(nbytes)),
               "cutlass_gemm_workspace")
        # The workspace is held by the plan: stream-K reads and resets it every
        # launch, so it must outlive the graph, not the call that built it.
        ws = torch.zeros(max(int(nbytes.value), 1), dtype=torch.uint8,
                         device=a.device)
        handle = ctypes.c_void_p()
        _check(lib.cutlass_gemm_plan(
            config, m, k, n, ldc, ctypes.c_float(beta), a.data_ptr(),
            b.data_ptr(), 0 if c is None else c.data_ptr(), d.data_ptr(),
            ws.data_ptr(), ctypes.byref(handle)), "cutlass_gemm_plan")
        got = (handle.value, ws)
        _PLANS[key] = got
    return got


def run(planned) -> None:
    """Launch a planned operator on the current stream. Capture-safe."""
    _check(library().cutlass_gemm_run(planned[0], _stream()), "cutlass_gemm_run")


def gemm(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, *, config: int,
         c: torch.Tensor | None = None, beta: float = 0.0,
         broadcast_c: bool = False) -> bool:
    """Plan-and-run in one call. False if the plan is missing inside a capture."""
    planned = plan(a, b, d, config=config, c=c, beta=beta,
                   broadcast_c=broadcast_c)
    if planned is None:
        return False
    run(planned)
    return True


__all__ = ["build", "library", "plan", "run", "gemm"]
