"""Pi0.5 expert attention with FP32 score output and fused masked softmax.

BF16 CUDA Q(400,256), K/V(1018,256), and additive mask(1018) produce
BF16 out(400,256). All rows are contiguous; out may alias Q. This Target's
<=1024-key softmax consumes the runtime mask, with FP32 scale/reductions and
BF16 probability rounding before the unchanged P@V matrix multiplication.
Scratch belongs to the runner; warmup compiles/allocates before graph capture.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache, partial
from pathlib import Path

import torch

from flash_vla.hardware.nvidia.native import NativeLibrary
from flash_vla.runtime.registry import Backend

NAMES = frozenset({"action_expert_attention"})


SOURCE = Path(__file__).with_suffix(".cu")

#: The expert attention's softmax and PV kernels.
LIBRARY = NativeLibrary(
    name="rtx5090_pi05_fused_attention",
    sources=(SOURCE,),
    arch=("-arch=sm_120a",),
    flags=("--fmad=false",))


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The loaded library with its C ABI declared; build it before graph capture."""
    kernels = LIBRARY.load()
    kernels.pi05_attention_softmax_launch.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int32] * 2 + [ctypes.c_float, ctypes.c_void_p])
    kernels.pi05_attention_softmax_launch.restype = ctypes.c_int32
    return kernels


def action_expert_attention(Q, K, V, mask, out, prefix_len=None, *, scratch):
    """Write BF16 attention output on the current stream, including when out is Q."""
    queries, head_dim = Q.shape
    keys = K.shape[0]
    logits = scratch("pi05_attention_logits", (queries, keys), torch.float32, Q.device)
    probabilities = scratch("pi05_attention_probabilities", (queries, keys), Q.dtype, Q.device)
    lib = library()
    torch.mm(Q, K.T, out_dtype=torch.float32, out=logits)
    rc = lib.pi05_attention_softmax_launch(
        logits.data_ptr(), mask.data_ptr(), probabilities.data_ptr(), queries, keys,
        float(head_dim ** -0.5), torch.cuda.current_stream(Q.device).cuda_stream)
    if rc:
        raise RuntimeError(
            f"pi05_attention_softmax queries={queries}, keys={keys}, threads=256: CUDA error {rc}")
    torch.mm(probabilities, V, out=out)
    return out


def make_wrappers(scratch, selected_names=None) -> dict:
    """Bind the selected expert attention wrapper to this runner's scratch."""
    names = NAMES if selected_names is None else selected_names
    wrappers = {"action_expert_attention": action_expert_attention}
    return {name: partial(wrappers[name], scratch=scratch) for name in names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers)


__all__ = ["BACKEND", "NAMES", "make_wrappers"]
