"""Pi0.5 backbone FFN under the mxfp8-llm-ffn recipe: quant_ops producers
around CUTLASS block-scaled GEMMs (mxfp8_backbone.cu).

Per layer, on the M rows the backbone runs (a replay-time bucket's,
`runtime/replay.py`):

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
as [N, K] with K contiguous, and serve every bucket; the runner's BF16 weights
stay allocated but are not read. A bucket's rows need a screened down
configuration (`DOWN_CONFIGS`; `supports`).
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping
import weakref

import torch

from flash_vla.hardware.nvidia.native import CUTLASS_DIR, CUTLASS_VERSION_HEADER, NativeLibrary
from flash_vla.hardware.nvidia.quant_ops import ops
from flash_vla.runtime.binding import RouteConstraint
from flash_vla.runtime.registry import Backend, Wrapper
from flash_vla.runtime.workspace import Scratch

from .torch_ops import RMS_EPS

NAMES = frozenset({"llm_backbone_norm_gated_ffn", "llm_backbone_ffn_down_residual"})
ROUTE_CONSTRAINTS = (RouteConstraint.atomic(
    NAMES, "the down GEMM reads the MXFP8 hidden the gated FFN leaves in scratch"),)
SOURCE = Path(__file__).with_suffix(".cu")
# Tile configuration of mxfp8_backbone.cu (the index its extern "C" entries take)
# per row count, from `lab/pi05/mxfp8_gemm_screen.py --rows ...`; a new bucket's
# rows are screened before they run.
GATE_UP_CONFIG = 8          # 128x128x128 pingpong, at every row count
# Down, by rows (results/pi05-rtx5090/replay-extent/v0/mxfp8-gemm-screen.json):
# 128x128 persistent fills the SMs at 968 rows (128 output tiles for 170 SMs);
# 712-896 rows (96-112 tiles) split K in three; at 576-640 (72-80 tiles)
# stream-K beats that split by 18%.
# 640 is libero's short bucket at a 128-row granularity, kept with its screen.
DOWN_CONFIGS = {968: 0, 896: 6, 832: 6, 712: 6, 640: 3, 576: 3}


def supports(shape: Mapping[str, int]) -> bool:
    """Whether the down GEMM has a screened configuration for the rows the
    backbone runs at `shape` (a replay bucket's prompt tokens after the image tokens)."""
    return shape["visual_tokens"] + shape["prompt_tokens"] in DOWN_CONFIGS


CUTLASS_ERROR = 1000  # mxfp8_backbone.cu: status values at or above are CUTLASS's


def check_status(status: int, operation: str) -> None:
    if status >= CUTLASS_ERROR:
        raise RuntimeError(f"{operation}: CUTLASS status {status - CUTLASS_ERROR}")
    if status:
        raise RuntimeError(f"{operation}: cudaError {status}")


#: The MXFP8 block-scaled GEMMs, built against CUTLASS for sm_120a.
LIBRARY = NativeLibrary(
    name="rtx5090_pi05_mxfp8_backbone",
    sources=(SOURCE,),
    arch=("-gencode", "arch=compute_120a,code=sm_120a"),
    flags=("--expt-relaxed-constexpr", "-DCUTLASS_ENABLE_GDC_FOR_SM100", "-diag-suppress", "20012"),
    include_dirs=(CUTLASS_DIR / "include", CUTLASS_DIR / "tools" / "util" / "include"),
    headers=(CUTLASS_VERSION_HEADER,))


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    """The loaded GEMM library with its C ABI declared; build it before graph capture."""
    kernels = LIBRARY.load()
    pointer, int32 = ctypes.c_void_p, ctypes.c_int32
    kernels.mxfp8_gemm_workspace.argtypes = [int32] * 4
    kernels.mxfp8_gemm_workspace.restype = ctypes.c_int64
    kernels.mxfp8_gemm_plan.argtypes = ([int32] * 4 + [ctypes.c_float] + [pointer] * 7
                                        + [ctypes.POINTER(pointer)])
    kernels.mxfp8_gemm_plan.restype = int32
    kernels.mxfp8_gemm_run.argtypes = [pointer, pointer]
    kernels.mxfp8_gemm_run.restype = int32
    kernels.mxfp8_gemm_destroy.argtypes = [pointer]
    kernels.mxfp8_gemm_destroy.restype = None
    return kernels


class GemmPlan:
    """output bf16 = activation * weight^T + beta * output, with activation [M, K]
    and weight [N, K] in MXFP8, bound to fixed addresses and one tile configuration."""

    def __init__(self, config: int, activation: ops.Activation, weight: ops.Activation,
                 output: torch.Tensor, beta: float, scratch: Scratch) -> None:
        rows, depth, cols = activation.rows, activation.cols, weight.rows
        if (weight.cols != depth or tuple(output.shape) != (rows, cols)
                or output.stride() != (cols, 1) or output.dtype != torch.bfloat16):
            raise ValueError(f"GEMM [{rows}, {depth}] x [{weight.rows}, {weight.cols}]^T "
                             f"cannot write {output.dtype} {tuple(output.shape)} {output.stride()}")
        kernels = library()
        size = kernels.mxfp8_gemm_workspace(config, rows, cols, depth)
        if size < 0:
            raise RuntimeError(f"mxfp8_gemm_workspace config={config} M={rows} N={cols} "
                               f"K={depth}: CUTLASS status {-size - CUTLASS_ERROR}")
        # Launches are serial on one stream, so plans of one size share a workspace.
        self.workspace = scratch(f"mxfp8_backbone_workspace_{config}", (max(size, 1),),
                                 torch.uint8, output.device)
        self.tensors = (activation, weight, output)
        self.handle = ctypes.c_void_p()
        check_status(kernels.mxfp8_gemm_plan(
            config, rows, cols, depth, beta, activation.values.data_ptr(),
            activation.scale.data_ptr(), weight.values.data_ptr(), weight.scale.data_ptr(),
            output.data_ptr(), self.workspace.data_ptr(),
            torch.cuda.current_stream(output.device).cuda_stream, ctypes.byref(self.handle)),
            f"mxfp8_gemm_plan config={config} M={rows} N={cols} K={depth} beta={beta}")
        self.run_native = kernels.mxfp8_gemm_run
        self.destroy = weakref.finalize(self, kernels.mxfp8_gemm_destroy, self.handle)

    def run(self) -> None:
        check_status(self.run_native(self.handle, torch.cuda.current_stream().cuda_stream),
                     "mxfp8_gemm_run")


def make_wrappers(scratch: Scratch, selected_names: frozenset[str] | None = None
                  ) -> dict[str, Wrapper]:
    """Wrappers for both FFN call sites. A layer's weights are quantized on its
    first call and its plans built on the first call at each bucket's rows, all
    in the runner's warmup, before capture."""
    quantized: dict[int, ops.Activation] = {}        # bf16 weight address -> MXFP8 [N, K]
    plans: dict[tuple[int, int], GemmPlan] = {}      # (weight address, rows) -> plan

    def allocator(role: str, device: torch.device) -> ops.Allocator:
        return lambda name, shape, dtype: scratch(role + name, shape, dtype, device)

    def activation(role: str, rows: int, cols: int, device: torch.device) -> ops.Activation:
        """MXFP8 [rows, cols] in scratch; every layer at one bucket gets the same buffers."""
        return ops.empty(rows, cols, "mxfp8", device, allocate=allocator(role, device))

    def operand(weight: torch.Tensor, weight_kn: Callable[[], torch.Tensor]) -> ops.Activation:
        """The MXFP8 B operand [N, K], blocks along K, of the layer `weight` belongs
        to, quantized from `weight_kn()` ([K, N] bf16) on the layer's first call."""
        address = weight.data_ptr()
        if address in quantized:
            return quantized[address]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("an MXFP8 backbone layer was not warmed before capture")
        kn = weight_kn()
        depth, cols = kn.shape
        return quantized.setdefault(address, ops.quantize(kn.t().contiguous(), ops.empty(
            cols, depth, "mxfp8", kn.device,
            allocate=allocator(f"mxfp8_backbone_weight_{address}", kn.device))))

    def llm_backbone_norm_gated_ffn(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor,
                                    out: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        rows, width = x.shape                      # a bucket's rows, 2048
        hidden_width = gate_w.shape[1]             # 16384
        normed = activation("mxfp8_backbone_normed", rows, width, x.device)
        hidden = activation("mxfp8_backbone_hidden", rows, hidden_width, x.device)
        # bf16 [M, 2 * 16384]: gate | up.
        projected = scratch("mxfp8_backbone_gate_up", (rows, 2 * hidden_width), torch.bfloat16,
                            x.device)
        key = (gate_w.data_ptr(), rows)
        plan = plans[key] if key in plans else plans.setdefault(key, GemmPlan(
            GATE_UP_CONFIG, normed, operand(gate_w, lambda: torch.cat((gate_w, up_w), dim=1)),
            projected, 0.0, scratch))
        ops.rms_norm(x, normed, eps=RMS_EPS)
        plan.run()
        ops.gated_act(projected[:, :hidden_width], projected[:, hidden_width:], hidden)
        return out

    def llm_backbone_ffn_down_residual(x: torch.Tensor, weight: torch.Tensor,
                                       out: torch.Tensor) -> torch.Tensor:
        rows, hidden_width = x.shape               # a bucket's rows, 16384; x itself is not read
        hidden = activation("mxfp8_backbone_hidden", rows, hidden_width, x.device)
        key = (weight.data_ptr(), rows)
        plan = plans[key] if key in plans else plans.setdefault(key, GemmPlan(
            DOWN_CONFIGS[rows], hidden, operand(weight, lambda: weight), out, 1.0, scratch))
        plan.run()
        return out

    wrappers = {"llm_backbone_norm_gated_ffn": llm_backbone_norm_gated_ffn,
                "llm_backbone_ffn_down_residual": llm_backbone_ffn_down_residual}
    return {name: wrapper for name, wrapper in wrappers.items()
            if selected_names is None or name in selected_names}


#: What the Target's registry routes to (`flash_vla.runtime.registry`).
BACKEND = Backend(names=frozenset(NAMES), make_wrappers=make_wrappers,
                  route_constraints=ROUTE_CONSTRAINTS, supports=supports)
