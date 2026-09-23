"""Pi0.5 backbone FFN under the mxfp8-llm-ffn recipe: quant_ops producers
around CUTLASS block-scaled GEMMs (mxfp8_backbone.cu).

Per layer, on the M = 968 prefix rows:

    x --quant_ops.rms_norm--> MXFP8 [M, 2048]
      --GEMM with gate|up [32768, 2048]--> BF16 [M, 32768], gate and up halves
      --quant_ops.gated_act--> MXFP8 hidden [M, 16384]
      --GEMM with down [2048, 16384], beta = 1--> residual [M, 2048], in place

These are the fake-quant reference's rounding points (fake_quant_ffn.py): the
activation is quantized from the BF16 RMSNorm output, gate and up round to
BF16, GELU-tanh(gate) * up rounds once before it is quantized, and the down
product joins the residual in FP32 with one rounding. The RMSNorm weight is
folded into gate and up.

The MXFP8 hidden stays in this backend's scratch and the down call site reads
it instead of its `x` argument, so the two call sites route here together;
neither `out` of the gated call site nor its `x_norm` is written. Weights are
quantized once, on a layer's first call (warmup, before capture), into scratch
as [N, K] with K contiguous; the runner's BF16 weights stay allocated but are
not read.

Each GEMM is planned for both prefix row buckets, M = 896 and M = 968, and
both plans run on every call: the kernel the prefix mask does not select exits
at once (mxfp8_backbone.cu, RowBucket). Under the short bucket, rows 896..967
of the residual keep their values and those of the hidden are stale; they are
padding either way. The two producers always cover all 968 rows.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable
import weakref

import torch

from flash_vla.hardware.nvidia.quant_ops import ops
from flash_vla.runtime.binding import RouteConstraint
from flash_vla.runtime.runner import Scratch

from .torch_ops import RMS_EPS

NAMES = frozenset({"llm_backbone_norm_gated_ffn_masked", "llm_backbone_ffn_down_residual_masked"})
ROUTE_CONSTRAINTS = (RouteConstraint.atomic(
    NAMES, "the down GEMM reads the MXFP8 hidden the gated FFN leaves in scratch"),)
SOURCE = Path(__file__).with_suffix(".cu")
BUCKET_ROWS = (896, 968)   # the short and full prefix; mxfp8_backbone.cu: kShortRows
# Tile configurations of mxfp8_backbone.cu (the index its extern "C" entries take).
GATE_UP_CONFIG = 8   # 128x128x128, pingpong
DOWN_CONFIG = 0      # 128x128x128, persistent
CUTLASS_ERROR = 1000  # mxfp8_backbone.cu: status values at or above are CUTLASS's


def check_status(status: int, operation: str) -> None:
    if status >= CUTLASS_ERROR:
        raise RuntimeError(f"{operation}: CUTLASS status {status - CUTLASS_ERROR}")
    if status:
        raise RuntimeError(f"{operation}: cudaError {status}")


@lru_cache(maxsize=None)
def library() -> ctypes.CDLL:
    """Build mxfp8_backbone.cu for sm_120a unless the cached build is current; load it."""
    repo = SOURCE.parents[7]
    output = repo / ".cache/cuda_ext/rtx5090_pi05_mxfp8_backbone/libmxfp8_backbone.so"
    if not output.exists() or output.stat().st_mtime_ns < SOURCE.stat().st_mtime_ns:
        cuda_home = os.environ.get("CUDA_HOME")
        nvcc = os.environ.get("FLASH_VLA_NVCC") or (
            str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc"))
        if nvcc is None:
            raise RuntimeError("set CUDA_HOME or FLASH_VLA_NVCC, or put nvcc on PATH")
        cutlass = Path(os.environ.get("CUTLASS_DIR", repo / "third_party/cutlass"))
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
             "--expt-relaxed-constexpr", "-DCUTLASS_ENABLE_GDC_FOR_SM100",
             "-diag-suppress", "20012", "-gencode", "arch=compute_120a,code=sm_120a",
             f"-I{cutlass}/include", f"-I{cutlass}/tools/util/include",
             str(SOURCE), "-o", str(output)],
            check=True)
    kernels = ctypes.CDLL(str(output))
    pointer, int32 = ctypes.c_void_p, ctypes.c_int32
    kernels.mxfp8_gemm_workspace.argtypes = [int32] * 4
    kernels.mxfp8_gemm_workspace.restype = ctypes.c_int64
    kernels.mxfp8_gemm_plan.argtypes = ([int32] * 4 + [ctypes.c_float] + [pointer] * 8
                                        + [ctypes.POINTER(pointer)])
    kernels.mxfp8_gemm_plan.restype = int32
    kernels.mxfp8_gemm_run.argtypes = [pointer, pointer]
    kernels.mxfp8_gemm_run.restype = int32
    kernels.mxfp8_gemm_destroy.argtypes = [pointer]
    kernels.mxfp8_gemm_destroy.restype = None
    return kernels


class GemmPlan:
    """output[:rows] bf16 = activation[:rows] * weight^T + beta * output[:rows], with
    activation [M, K] and weight [N, K] in MXFP8, bound to fixed addresses and one
    tile configuration. With a prefix `mask`, `rows` is a bucket of BUCKET_ROWS and
    the kernel runs only when the mask selects it; without one it always runs."""

    def __init__(self, config: int, activation: ops.Activation, weight: ops.Activation,
                 output: torch.Tensor, beta: float, scratch: Scratch, *, rows: int,
                 mask: torch.Tensor | None) -> None:
        depth, cols = activation.cols, weight.rows
        if (weight.cols != depth or tuple(output.shape) != (activation.rows, cols)
                or output.stride() != (cols, 1) or output.dtype != torch.bfloat16
                or rows > activation.rows or (mask is not None and rows not in BUCKET_ROWS)):
            raise ValueError(f"GEMM [{rows} of {activation.rows}, {depth}] x [{weight.rows}, "
                             f"{weight.cols}]^T cannot write {output.dtype} "
                             f"{tuple(output.shape)} {output.stride()}")
        kernels = library()
        size = kernels.mxfp8_gemm_workspace(config, rows, cols, depth)
        if size < 0:
            raise RuntimeError(f"mxfp8_gemm_workspace config={config} M={rows} N={cols} "
                               f"K={depth}: CUTLASS status {-size - CUTLASS_ERROR}")
        # Launches are serial on one stream, so plans of one size share a workspace.
        self.workspace = scratch(f"mxfp8_backbone_workspace_{config}", (max(size, 1),),
                                 torch.uint8, output.device)
        self.tensors = (activation, weight, output, mask)
        self.handle = ctypes.c_void_p()
        check_status(kernels.mxfp8_gemm_plan(
            config, rows, cols, depth, beta, activation.values.data_ptr(),
            activation.scale.data_ptr(), weight.values.data_ptr(), weight.scale.data_ptr(),
            output.data_ptr(), None if mask is None else mask.data_ptr(),
            self.workspace.data_ptr(), torch.cuda.current_stream(output.device).cuda_stream,
            ctypes.byref(self.handle)),
            f"mxfp8_gemm_plan config={config} M={rows} N={cols} K={depth} beta={beta}")
        self.run_native = kernels.mxfp8_gemm_run
        self.destroy = weakref.finalize(self, kernels.mxfp8_gemm_destroy, self.handle)

    def run(self) -> None:
        check_status(self.run_native(self.handle, torch.cuda.current_stream().cuda_stream),
                     "mxfp8_gemm_run")


def make_wrappers(scratch: Scratch, selected_names: set[str] | None = None
                  ) -> dict[str, Callable[..., torch.Tensor]]:
    """Wrappers for both FFN call sites; a layer's plans and quantized weights are
    built on its first call, which the runner's warmup makes before capture."""
    gate_up_plans: dict[int, list[GemmPlan]] = {}   # gate weight address -> bucket plans
    down_plans: dict[int, list[GemmPlan]] = {}      # down weight address -> bucket plans

    def _allocator(role: str, device: torch.device) -> ops.Allocator:
        return lambda name, shape, dtype: scratch(role + name, shape, dtype, device)

    def _activation(role: str, rows: int, cols: int, device: torch.device) -> ops.Activation:
        """MXFP8 [rows, cols] in scratch; every layer gets the same buffers."""
        return ops.empty(rows, cols, "mxfp8", device, allocate=_allocator(role, device))

    def _quantized_weight(weight_kn: torch.Tensor, role: str) -> ops.Activation:
        """weight [K, N] bf16 as the GEMM's B operand: MXFP8 [N, K], blocks along K."""
        depth, cols = weight_kn.shape
        return ops.quantize(weight_kn.t().contiguous(), ops.empty(
            cols, depth, "mxfp8", weight_kn.device, allocate=_allocator(role, weight_kn.device)))

    def _layer_buckets(plans: dict[int, list[GemmPlan]], weight: torch.Tensor,
                       quantize: Callable[[], ops.Activation],
                       build: Callable[[ops.Activation, int], GemmPlan]) -> list[GemmPlan]:
        """The row-bucket plans of the layer `weight` belongs to; on the layer's
        first call its weight is quantized and both plans are built."""
        cached = plans.get(weight.data_ptr())
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("an MXFP8 backbone layer was not warmed before capture")
        quantized = quantize()
        return plans.setdefault(weight.data_ptr(),
                                [build(quantized, rows) for rows in BUCKET_ROWS])

    def llm_backbone_norm_gated_ffn_masked(x: torch.Tensor, gate_w: torch.Tensor,
                                           up_w: torch.Tensor, out: torch.Tensor,
                                           x_norm: torch.Tensor, mask: torch.Tensor
                                           ) -> torch.Tensor:
        rows, width = x.shape                      # 968, 2048
        hidden_width = gate_w.shape[1]             # 16384
        normed = _activation("mxfp8_backbone_normed", rows, width, x.device)
        hidden = _activation("mxfp8_backbone_hidden", rows, hidden_width, x.device)
        # bf16 [M, 2 * 16384]: gate | up.
        projected = scratch("mxfp8_backbone_gate_up", (rows, 2 * hidden_width), torch.bfloat16,
                            x.device)
        buckets = _layer_buckets(
            gate_up_plans, gate_w,
            lambda: _quantized_weight(torch.cat((gate_w, up_w), dim=1),
                                      f"mxfp8_backbone_gate_up_{gate_w.data_ptr()}"),
            lambda gate_up, rows: GemmPlan(GATE_UP_CONFIG, normed, gate_up, projected, 0.0,
                                           scratch, rows=rows, mask=mask))
        ops.rms_norm(x, normed, eps=RMS_EPS)
        for plan in buckets:
            plan.run()
        ops.gated_act(projected[:, :hidden_width], projected[:, hidden_width:], hidden)
        return out

    def llm_backbone_ffn_down_residual_masked(x: torch.Tensor, weight: torch.Tensor,
                                              out: torch.Tensor, mask: torch.Tensor
                                              ) -> torch.Tensor:
        rows, hidden_width = x.shape               # 968, 16384; x itself is not read
        hidden = _activation("mxfp8_backbone_hidden", rows, hidden_width, x.device)
        for plan in _layer_buckets(
                down_plans, weight,
                lambda: _quantized_weight(weight, f"mxfp8_backbone_down_{weight.data_ptr()}"),
                lambda down, rows: GemmPlan(DOWN_CONFIG, hidden, down, out, 1.0, scratch,
                                            rows=rows, mask=mask)):
            plan.run()
        return out

    wrappers = {"llm_backbone_norm_gated_ffn_masked": llm_backbone_norm_gated_ffn_masked,
                "llm_backbone_ffn_down_residual_masked": llm_backbone_ffn_down_residual_masked}
    return {name: wrapper for name, wrapper in wrappers.items()
            if selected_names is None or name in selected_names}
