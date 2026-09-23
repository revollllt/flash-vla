"""Producers that write MXFP8 or NVFP4 activations for Blackwell block-scaled GEMMs.

A quantized GEMM's A operand is produced by a norm or an activation; running
that producer in BF16 and quantizing in a second kernel writes and reads the
activation twice. These ops quantize in the producer's epilogue instead and
write E4M3/E2M1 values plus the 128x4 swizzled scales CUTLASS, cuDNN and
FlashInfer's block-scaled GEMMs read on sm_100 and later. ``quantize`` covers
activations whose producer is not here (attention output).

Every op writes into an ``Activation`` allocated once with ``empty``, so the ops
are CUDA-graph safe; the output format is the ``Activation``'s. Format
``"bf16"`` runs the same kernel stopped at its rounding point, which is the
fusion contract: ``quantize(op(x) -> bf16)`` equals ``op(x) -> mxfp8/nvfp4``
bit for bit. Quantization follows FlashInfer 0.7.0's default fast path (see
``cuda/quant/block_quant.cuh``), so outputs are interchangeable with
``flashinfer.mxfp8_quantize`` / ``fp4_quantize`` and feed ``mm_mxfp8`` /
``mm_fp4``. References: ``flash_vla.quantization.reference``.

Every op launches with programmatic dependent launch (PDL) unless ``pdl=False``:
it may start while the previous kernel drains, and reads nothing an earlier
kernel writes before that kernel completes. The norm weight and bias are read
before that point, so no kernel of a captured chain may write them.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Callable, Literal

import torch

from flash_vla.hardware.nvidia.native import NativeLibrary
from flash_vla.quantization import formats

Format = Literal["bf16", "mxfp8", "nvfp4"]
Allocator = Callable[[str, tuple[int, ...], torch.dtype], torch.Tensor]

PACKAGE_DIR = Path(__file__).resolve().parent
CUDA_INCLUDE_DIR = PACKAGE_DIR.parent / "cuda"
SOURCE = PACKAGE_DIR / "fused_quant.cu"
# Blackwell families by compute capability major. A family target (sm_100f
# runs on 10.0 and 10.3, sm_120f on 12.0 and 12.1) has the E2M1/UE8M0
# conversions; Hopper and earlier lack cvt.e2m1x2 and block-scaled MMA alike.
BLACKWELL_FAMILIES = {10: "100f", 11: "110f", 12: "120f"}
OUTPUT_KIND = {"bf16": 0, "mxfp8": 1, "nvfp4": 2}           # fused_quant.cu: Out
RMS_WEIGHT_MODE = {"none": 0, "mul": 1, "one_plus": 2}      # fused_quant.cu: weight_mode
GATED_ACTIVATION = {"gelu_tanh": 0, "silu": 1}              # fused_quant.cu: act
ELEMENTWISE_KIND = {"copy": 0, "gelu_tanh": 1, "gated": 2}  # fused_quant.cu: kind
MAX_ROW_COLS = 8192        # row kernels: one CTA of at most 1024 threads x 8 elements
MAX_ELEMENTWISE_ROWS = 65535  # elementwise kernels: one grid row per matrix row
# A build knob for sweeps: FLASH_VLA_QUANT_PDL_TRIGGER=<point> builds the kernels
# with another PDL trigger point than fused_quant.cu's measured default.
PDL_TRIGGER_OVERRIDE = os.environ.get("FLASH_VLA_QUANT_PDL_TRIGGER")


@lru_cache(maxsize=None)
def library(device: torch.device) -> ctypes.CDLL:
    """The kernels for this device's Blackwell family; call before graph capture."""
    major, minor = torch.cuda.get_device_capability(device)
    if major not in BLACKWELL_FAMILIES:
        raise RuntimeError(f"quant_ops needs a Blackwell GPU (sm_100 or later); "
                           f"{torch.cuda.get_device_name(device)} is sm_{major}{minor}")
    family = BLACKWELL_FAMILIES[major]
    trigger = () if PDL_TRIGGER_OVERRIDE is None else (f"-DFVQ_PDL_TRIGGER={int(PDL_TRIGGER_OVERRIDE)}",)
    kernels = NativeLibrary(
        name=f"fused_quant_sm_{family}",
        sources=(SOURCE,),
        arch=("-gencode", f"arch=compute_{family},code=sm_{family}"),
        flags=("--fmad=false", *trigger),
        include_dirs=(CUDA_INCLUDE_DIR,),
        headers=(CUDA_INCLUDE_DIR / "quant" / "block_quant.cuh",)).load()
    pointer, int32, int64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64
    kernels.fvq_row_norm.argtypes = (
        [pointer] * 11   # x, residual, residual_out, weight, bias, mod_scale, mod_shift,
                         # factor_out, values, scale, global_scale
        + [int64]        # mod_row_stride
        + [int32] * 5    # rows, cols, weight_mode, round_factor, mod_group_rows
        + [ctypes.c_float, int32, int32, int32, pointer])  # eps, layer, out_kind, pdl, stream
    kernels.fvq_row_norm.restype = int32
    kernels.fvq_elementwise.argtypes = (
        [pointer, int64, pointer, int64]  # x, x_row_stride, up, up_row_stride
        + [pointer] * 3                   # values, scale, global_scale
        + [int32] * 7                     # rows, cols, kind, act, round_act, out_kind, pdl
        + [pointer])                      # stream
    kernels.fvq_elementwise.restype = int32
    return kernels


@dataclass(frozen=True)
class Activation:
    """A producer output: BF16 values, or block-quantized values with their scales.

    values: bf16 [M, K], float8_e4m3fn [M, K] (mxfp8) or uint8 [M, K/2] (nvfp4,
    even element in the low nibble). scale: uint8, flat 128x4 swizzled, padding
    zero. global_scale: float32 (1,), NVFP4's encode scale ``448 * 6 / amax``,
    read on the device so it can change between graph replays.
    """
    fmt: Format
    values: torch.Tensor
    scale: torch.Tensor | None = None
    global_scale: torch.Tensor | None = None

    @property
    def rows(self) -> int:
        return self.values.shape[0]

    @property
    def cols(self) -> int:
        return self.values.shape[1] * (2 if self.fmt == "nvfp4" else 1)


def empty(rows: int, cols: int, fmt: Format, device: torch.device | str = "cuda", *,
          global_scale: torch.Tensor | None = None,
          allocate: Allocator | None = None) -> Activation:
    """Allocate an output once, before capture. `allocate(name, shape, dtype)` must
    return zeroed memory, as a runner's Scratch does:
    ``lambda name, shape, dtype: scratch(role + name, shape, dtype, device)``."""
    def _zeros(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    allocate_zeroed = allocate if allocate is not None else _zeros
    if cols % 8:
        raise ValueError(f"rows must be whole 16-byte vectors: K divisible by 8, got {cols}")
    if fmt == "bf16":
        return Activation(fmt, allocate_zeroed("values", (rows, cols), torch.bfloat16))
    if fmt == "nvfp4" and global_scale is None:
        raise ValueError("nvfp4 needs a float32 (1,) global_scale tensor")
    padded_rows, padded_blocks = formats.scale_shape(rows, cols, fmt)
    values = (allocate_zeroed("values", (rows, cols), torch.float8_e4m3fn) if fmt == "mxfp8"
              else allocate_zeroed("values", (rows, cols // 2), torch.uint8))
    scale = allocate_zeroed("scale", (padded_rows * padded_blocks,), torch.uint8)
    return Activation(fmt, values, scale, global_scale)


def data_pointer(tensor: torch.Tensor | None) -> int | None:
    return None if tensor is None else tensor.data_ptr()


def check_bf16_matrix(name: str, matrix: torch.Tensor, rows: int, cols: int,
                      contiguous: bool) -> None:
    if (matrix.dtype != torch.bfloat16 or not matrix.is_cuda or matrix.dim() != 2
            or tuple(matrix.shape) != (rows, cols)):
        raise ValueError(f"{name} must be a CUDA bf16 [{rows}, {cols}] tensor, got "
                         f"{matrix.dtype} {tuple(matrix.shape)}")
    if (matrix.stride(1) != 1 or matrix.data_ptr() % 16 or matrix.stride(0) % 8
            or (contiguous and matrix.stride(0) != cols)):
        raise ValueError(f"{name} needs unit column stride and 16-byte aligned rows"
                         + (", contiguous" if contiguous else ""))


def launch_row_norm(op_name: str, x: torch.Tensor, out: Activation, *,
                    weight: torch.Tensor | None, bias: torch.Tensor | None,
                    modulation: tuple[torch.Tensor, torch.Tensor] | None, mod_group_rows: int,
                    residual: torch.Tensor | None, residual_out: torch.Tensor | None,
                    factor_out: torch.Tensor | None, weight_mode: int, round_factor: bool,
                    eps: float, layer: bool, pdl: bool) -> Activation:
    """Validate what RMSNorm and LayerNorm share, then run fvq_row_norm."""
    rows, cols = x.shape
    if (out.rows, out.cols) != (rows, cols):
        raise ValueError(f"{op_name}: output is {out.rows}x{out.cols}, input {rows}x{cols}")
    if cols > MAX_ROW_COLS:
        raise ValueError(f"{op_name} keeps a row in registers: K <= {MAX_ROW_COLS}, got {cols}")
    matrices = [("x", x), ("residual", residual), ("residual_out", residual_out)]
    for name, matrix in [(name, matrix) for name, matrix in matrices if matrix is not None]:
        check_bf16_matrix(name, matrix, rows, cols, contiguous=True)
    vectors = [("weight", weight), ("bias", bias)]
    for name, vector in [(name, vector) for name, vector in vectors if vector is not None]:
        if (vector.dtype != torch.bfloat16 or vector.numel() != cols
                or not vector.is_contiguous() or vector.data_ptr() % 16):
            raise ValueError(f"{name} must be a contiguous, 16-byte aligned bf16 [{cols}]")
    mod_scale, mod_shift = modulation if modulation is not None else (None, None)
    status = library(x.device).fvq_row_norm(
        data_pointer(x), data_pointer(residual), data_pointer(residual_out),
        data_pointer(weight), data_pointer(bias), data_pointer(mod_scale),
        data_pointer(mod_shift), data_pointer(factor_out), data_pointer(out.values),
        data_pointer(out.scale), data_pointer(out.global_scale),
        cols if mod_scale is None else mod_scale.stride(0),
        rows, cols, weight_mode, int(round_factor), mod_group_rows, eps, int(layer),
        OUTPUT_KIND[out.fmt], int(pdl), torch.cuda.current_stream(x.device).cuda_stream)
    if status != 0:
        raise RuntimeError(f"{op_name}({rows}x{cols}, {out.fmt}) failed: cudaError {status}")
    return out


def rms_norm(x: torch.Tensor, out: Activation, *, weight: torch.Tensor | None = None,
             weight_mode: Literal["none", "mul", "one_plus"] | None = None, eps: float = 1e-6,
             round_factor: bool = False, factor_out: torch.Tensor | None = None,
             residual: torch.Tensor | None = None, residual_out: torch.Tensor | None = None,
             pdl: bool = True) -> Activation:
    """out = rms_norm(x [+ residual]); see reference.rms_norm for the rounding.

    weight_mode "mul" (the default with a weight): bf16(bf16(x*r) * w);
    "one_plus": bf16(x*r*(1+w)); "none". round_factor rounds r to bf16 first
    (Pi0.5 decoder) and factor_out, bf16 [M], receives it. residual_out receives
    bf16(x + residual).
    """
    mode = weight_mode if weight_mode is not None else ("none" if weight is None else "mul")
    if (weight is None) != (mode == "none"):
        raise ValueError(f"weight_mode {mode!r} and weight disagree")
    if factor_out is not None and (factor_out.dtype != torch.bfloat16
                                   or factor_out.numel() != x.shape[0]
                                   or not factor_out.is_contiguous()):
        raise ValueError(f"factor_out must be a contiguous bf16 [{x.shape[0]}]")
    return launch_row_norm("rms_norm", x, out, weight=weight, bias=None, modulation=None,
                           mod_group_rows=0, residual=residual, residual_out=residual_out,
                           factor_out=factor_out, weight_mode=RMS_WEIGHT_MODE[mode],
                           round_factor=round_factor, eps=eps, layer=False, pdl=pdl)


def layer_norm(x: torch.Tensor, out: Activation, *, weight: torch.Tensor | None = None,
               bias: torch.Tensor | None = None, eps: float = 1e-5,
               modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
               mod_group_rows: int = 0, residual: torch.Tensor | None = None,
               residual_out: torch.Tensor | None = None, pdl: bool = True) -> Activation:
    """out = layer_norm(x [+ residual]) in fp32 with one rounding, then optionally
    AdaLN's bf16 ``y * (1 + scale) + shift``. modulation = (scale, shift), each bf16
    [groups, K], one row per mod_group_rows rows (0: one row for all). The rows
    may be strided, as the halves of AdaLN's ``linear(...).chunk(2, dim=1)`` are.
    """
    if (weight is None) != (bias is None):
        raise ValueError("layer_norm's affine needs both weight and bias")
    rows, cols = x.shape
    groups = -(-rows // mod_group_rows) if mod_group_rows else 1
    present_modulation = list(modulation) if modulation is not None else []
    for vectors in present_modulation:
        if (vectors.dtype != torch.bfloat16 or tuple(vectors.shape) != (groups, cols)
                or vectors.stride(1) != 1 or vectors.stride(0) % 8 or vectors.data_ptr() % 16
                or vectors.stride(0) != present_modulation[0].stride(0)):
            raise ValueError(f"modulation must be two bf16 [{groups}, {cols}] tensors with unit "
                             "column stride and the same 16-byte aligned row stride")
    return launch_row_norm("layer_norm", x, out, weight=weight, bias=bias, modulation=modulation,
                           mod_group_rows=mod_group_rows, residual=residual,
                           residual_out=residual_out, factor_out=None,
                           weight_mode=int(weight is not None),  # LayerNorm: 1 = affine
                           round_factor=False, eps=eps, layer=True, pdl=pdl)


def launch_elementwise(op_name: str, x: torch.Tensor, up: torch.Tensor | None, out: Activation,
                       kind: int, act: int, round_act: bool, pdl: bool) -> Activation:
    """Validate and run fvq_elementwise (kind: ELEMENTWISE_KIND; act: GATED_ACTIVATION)."""
    rows, cols = x.shape
    if (out.rows, out.cols) != (rows, cols):
        raise ValueError(f"{op_name}: output is {out.rows}x{out.cols}, input {rows}x{cols}")
    if rows > MAX_ELEMENTWISE_ROWS:
        raise ValueError(f"{op_name}: M <= {MAX_ELEMENTWISE_ROWS}, got {rows}")
    matrices = [("x", x), ("up", up)]
    for name, matrix in [(name, matrix) for name, matrix in matrices if matrix is not None]:
        check_bf16_matrix(name, matrix, rows, cols, contiguous=False)
    status = library(x.device).fvq_elementwise(
        data_pointer(x), x.stride(0), data_pointer(up), 0 if up is None else up.stride(0),
        data_pointer(out.values), data_pointer(out.scale), data_pointer(out.global_scale),
        rows, cols, kind, act, int(round_act), OUTPUT_KIND[out.fmt], int(pdl),
        torch.cuda.current_stream(x.device).cuda_stream)
    if status != 0:
        raise RuntimeError(f"{op_name}({rows}x{cols}, {out.fmt}) failed: cudaError {status}")
    return out


def quantize(x: torch.Tensor, out: Activation, *, pdl: bool = True) -> Activation:
    """out = x, in out's format. x may be a row-strided view."""
    return launch_elementwise("quantize", x, None, out, kind=ELEMENTWISE_KIND["copy"], act=0,
                              round_act=False, pdl=pdl)


def gelu_tanh(x: torch.Tensor, out: Activation, *, pdl: bool = True) -> Activation:
    """out = gelu(x, approximate="tanh"), fp32 with one rounding."""
    return launch_elementwise("gelu_tanh", x, None, out, kind=ELEMENTWISE_KIND["gelu_tanh"],
                              act=0, round_act=False, pdl=pdl)


def gated_act(gate: torch.Tensor, up: torch.Tensor, out: Activation, *,
              act: Literal["gelu_tanh", "silu"] = "gelu_tanh", round_act: bool = False,
              pdl: bool = True) -> Activation:
    """out = act(gate) * up. gate and up may be the column halves of one [M, 2N] GEMM
    output. round_act rounds act(gate) to bf16 before the product (HF SwiGLU)."""
    return launch_elementwise("gated_act", gate, up, out, kind=ELEMENTWISE_KIND["gated"],
                              act=GATED_ACTIVATION[act], round_act=round_act, pdl=pdl)
