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
from pathlib import Path

import torch

from flash_vla.hardware.nvidia.native import CUTLASS_DIR, CUTLASS_VERSION_HEADER, NativeLibrary

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "kernels" / "cutlass_gemm.cu"
#: One arch-specific target: the kernels use sm_120a instructions.
ARCH = "sm_120a"

_LIB = None


#: The CUTLASS GEMM configurations of this Target.
LIBRARY = NativeLibrary(
    name="rtx5090_pi0_cutlass",
    sources=(_SRC,),
    arch=("-gencode", f"arch=compute_{ARCH[3:]},code={ARCH}"),
    flags=("--expt-relaxed-constexpr",),
    include_dirs=(CUTLASS_DIR / "include",),
    headers=(CUTLASS_VERSION_HEADER,))


def build(verbose: bool = False) -> Path:
    """Compile the library unless this exact build exists; return its path."""
    return LIBRARY.build(verbose=verbose)


def library(verbose: bool = False):
    global _LIB
    if _LIB is None:
        lib = LIBRARY.load(verbose=verbose)
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
        lib.cutlass_gemm_set_pdl.argtypes = [ctypes.c_int]
        lib.cutlass_gemm_set_pdl.restype = ctypes.c_int
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


def set_pdl(on: bool) -> None:
    """Turn programmatic dependent launch on or off for these GEMMs.

    Only the launch changes: the entry point carries the wait and the trigger
    either way, and both are no-ops on a grid launched without programmatic
    serialization.
    """
    _check(library().cutlass_gemm_set_pdl(1 if on else 0), "cutlass_gemm_set_pdl")


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
