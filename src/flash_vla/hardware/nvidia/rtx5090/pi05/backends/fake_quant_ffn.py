"""Pi0.5 backbone FFN under a fake-quantized recipe, for quality measurement.

The prefix FFN's three GEMMs run on operands quantized by the real kernels
(`hardware/nvidia/quant_ops`, byte-identical to FlashInfer's) and multiplied as
a block-scaled GEMM multiplies them (`flash_vla.quantization.fake_quant`): the
gate and up GEMMs read the RMSNorm output, the down GEMM reads
GELU-tanh(gate) * up, and every weight is quantized along K. The BF16 route's
rounding points are kept: gate and up round to BF16, the gated product rounds
once, and the down product joins the residual in FP32 with one rounding. A layer
whose format is "bf16" runs the BF16 route unchanged.

The recipe gives one format per layer for each GEMM group,
{"gate_up": [...], "down": [...]}, from the runner asset `quantization_recipe`
(a JSON file) or, without that asset, the registry entry's single format. NVFP4
activations use the dynamic per-tensor encode scale 448 * 6 / amax, an upper
bound on what a calibrated static scale reaches; weights use their own. Layers
are numbered in the order the forward first reaches their weights, which is
layer order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from flash_vla.hardware.nvidia.quant_ops import ops
from flash_vla.models.pi05.spec import ENCODER_LAYERS
from flash_vla.quantization import formats
from flash_vla.quantization.fake_quant import (FakeQuantized, dequantized_operand,
                                               dynamic_encode_scale, fake_quant_matmul)
from flash_vla.runtime.registry import Backend
from flash_vla.runtime.workspace import Scratch

from .torch_ops import RMS_EPS

NAMES = frozenset({"llm_backbone_norm_gated_ffn_masked", "llm_backbone_ffn_down_residual_masked"})
LayerFormat = Literal["bf16", "mxfp8", "nvfp4"]
LAYER_FORMATS = ("bf16", "mxfp8", "nvfp4")


def kernel_fake_quantize(rows: torch.Tensor, fmt: Literal["mxfp8", "nvfp4"]) -> FakeQuantized:
    """bf16 [M, K] quantized by quant_ops, blocks along K; NVFP4 with the dynamic
    per-tensor encode scale."""
    row_count, cols = rows.shape
    encode_scale = dynamic_encode_scale(rows) if fmt == "nvfp4" else None
    quantized = ops.quantize(rows, ops.empty(row_count, cols, fmt, rows.device,
                                             global_scale=encode_scale))
    scale_codes = formats.unswizzle(quantized.scale, row_count, cols // formats.FORMATS[fmt].block)
    return dequantized_operand(quantized.values, scale_codes, fmt, encode_scale)


class FakeQuantFFN:
    """The fake-quantized FFN: every layer in `fmt` unless the runner supplies a recipe."""

    def __init__(self, fmt: LayerFormat) -> None:
        self.fmt = fmt

    def make_wrappers(self, scratch: Scratch, selected_names: frozenset[str] | None = None
                      ) -> dict[str, Callable[..., torch.Tensor]]:
        recipe_path = scratch.assets.get("quantization_recipe")
        recipe = (json.loads(Path(recipe_path).read_text()) if recipe_path is not None
                  else {"gate_up": [self.fmt] * ENCODER_LAYERS, "down": [self.fmt] * ENCODER_LAYERS})
        for group in ("gate_up", "down"):
            if len(recipe[group]) != ENCODER_LAYERS or not set(recipe[group]) <= set(LAYER_FORMATS):
                raise ValueError(f"recipe {group!r} needs {ENCODER_LAYERS} formats from "
                                 f"{LAYER_FORMATS}, got {recipe[group]!r}")
        layer_of: dict[tuple[str, int], int] = {}          # (group, weight address) -> layer
        weights: dict[int, FakeQuantized] = {}             # weight address -> fake-quantized
        role = f"pi05_fake_quant_ffn_{id(weights)}"

        def _layer(group: str, weight: torch.Tensor) -> int:
            key = (group, weight.data_ptr())
            return layer_of.setdefault(key, sum(1 for seen, _ in layer_of if seen == group))

        def _weight(weight_kn: torch.Tensor, fmt: LayerFormat) -> FakeQuantized:
            """The weight [K, N] fake-quantized along K, kept in scratch as [N, K]."""
            cached = weights.get(weight_kn.data_ptr())
            if cached is not None:
                return cached
            prepared = kernel_fake_quantize(weight_kn.t().contiguous(), fmt)
            name = f"{role}_{weight_kn.data_ptr()}"
            values = scratch(name + "_values", tuple(prepared.values.shape), torch.bfloat16,
                             weight_kn.device)
            decode_scale = scratch(name + "_decode", (), torch.float32, weight_kn.device)
            values.copy_(prepared.values)
            decode_scale.copy_(prepared.decode_scale)
            weights[weight_kn.data_ptr()] = FakeQuantized(values, decode_scale)
            return weights[weight_kn.data_ptr()]

        def llm_backbone_norm_gated_ffn_masked(x: torch.Tensor, gate_w: torch.Tensor,
                                               up_w: torch.Tensor, out: torch.Tensor,
                                               x_norm: torch.Tensor, mask: torch.Tensor
                                               ) -> torch.Tensor:
            rows = x.shape[0]
            fmt = recipe["gate_up"][_layer("gate_up", gate_w)]
            x_fp32 = x.float()
            x_norm[:rows].copy_(
                (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(x.dtype))
            normalized = x_norm[:rows]
            if fmt == "bf16":
                gate, up = normalized @ gate_w, normalized @ up_w
            else:
                activation = kernel_fake_quantize(normalized, fmt)
                gate = fake_quant_matmul(activation, _weight(gate_w, fmt)).to(torch.bfloat16)
                up = fake_quant_matmul(activation, _weight(up_w, fmt)).to(torch.bfloat16)
            out[:rows].copy_((F.gelu(gate.float(), approximate="tanh") * up.float()).to(out.dtype))
            return out

        def llm_backbone_ffn_down_residual_masked(x: torch.Tensor, weight: torch.Tensor,
                                                  out: torch.Tensor, mask: torch.Tensor
                                                  ) -> torch.Tensor:
            fmt = recipe["down"][_layer("down", weight)]
            if fmt == "bf16":
                return out.addmm_(x, weight)
            product = fake_quant_matmul(kernel_fake_quantize(x, fmt), _weight(weight, fmt))
            out.copy_((out.float() + product).to(out.dtype))
            return out

        wrappers = {"llm_backbone_norm_gated_ffn_masked": llm_backbone_norm_gated_ffn_masked,
                    "llm_backbone_ffn_down_residual_masked": llm_backbone_ffn_down_residual_masked}
        return {name: wrapper for name, wrapper in wrappers.items()
                if selected_names is None or name in selected_names}


def backend(fmt: LayerFormat) -> Backend:
    """The registry entry of the FFN fake-quantized to `fmt`."""
    return Backend(names=NAMES, make_wrappers=FakeQuantFFN(fmt).make_wrappers)
