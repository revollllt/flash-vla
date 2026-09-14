"""Pi0.5 expert attention with FP32 score output and fused masked softmax.

BF16 CUDA Q(400,256), K/V(1018,256), and additive mask(1018) produce
BF16 out(400,256). All rows are contiguous; out may alias Q. This Target's
<=1024-key softmax consumes the runtime mask, with FP32 scale/reductions and
BF16 probability rounding before the unchanged P@V matrix multiplication.
Scratch belongs to the runner; warmup compiles/allocates before graph capture.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache, partial
from pathlib import Path

import torch

NAMES = frozenset({"action_expert_attention"})


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_suffix(".cu")
    directory = source.parents[7] / ".cache" / "cuda_ext" / "rtx5090_pi05_fused_attention"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "fused_attention.so"
    if not output.exists() or output.stat().st_mtime < source.stat().st_mtime:
        nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc")
        subprocess.run(
            [nvcc, "-O3", "-std=c++17", "--fmad=false", "--shared", "-Xcompiler",
             "-fPIC", "-arch=sm_120a", str(source), "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.pi05_attention_softmax_launch.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int32] * 2 + [ctypes.c_float, ctypes.c_void_p])
    lib.pi05_attention_softmax_launch.restype = ctypes.c_int32
    return lib


def action_expert_attention(Q, K, V, mask, out, prefix_len=None, *, scratch):
    """Write BF16 attention output on the current stream, including when out is Q."""
    queries, head_dim = Q.shape
    keys = K.shape[0]
    logits = scratch("pi05_attention_logits", (queries, keys), torch.float32, Q.device)
    probabilities = scratch("pi05_attention_probabilities", (queries, keys), Q.dtype, Q.device)
    lib = _library()
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


__all__ = ["NAMES", "make_wrappers"]
