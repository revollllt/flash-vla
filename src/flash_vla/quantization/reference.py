"""PyTorch reference for block-quantized activations and the producers that emit them.

Quantization follows FlashInfer 0.7.0's default fast path, which the kernels in
``hardware/nvidia/quant_ops`` implement (rcp is ``rcp.approx.ftz``):

- MXFP8: ``s = ue8m0_round_up(amax * rcp(448))``,
  ``q = e4m3_rn(clamp(x * rcp(s), -448, 448))``.
- NVFP4: ``s = e4m3_rn(S * (amax * rcp(6)))``, ``q = e2m1_rn(x * rcp(s * rcp(S)))``,
  where ``S = 448 * 6 / global_amax`` is the per-tensor encode scale.

Products flush subnormals to zero, as FlashInfer's ``-use_fast_math`` build
does: a block whose MXFP8 scale is code 0 (2**-127, subnormal), which includes
every all-zero block, quantizes to zeros. This reference uses correctly rounded
reciprocals. MXFP8 scales are powers of two, whose reciprocals are exact, so
MXFP8 agrees with the kernels bit for bit. NVFP4's ``rcp(s * rcp(S))`` is not
exact: a value on an E2M1 rounding boundary can move by one step. The kernels
are held to FlashInfer bit for bit (``lab/quantization/check_quant_ops.py``)
and to this reference within that step.

The producers are the BF16 numerical contracts the fused kernels stop at before
quantizing; each rounds where the model's reference implementation rounds.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from .formats import MXFP8, NVFP4

E4M3_MAX = 448.0
E2M1_MAX = 6.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # magnitude of codes 0-7
# The kernels multiply by float32 reciprocals of the format maxima.
E4M3_MAX_RECIPROCAL = torch.tensor(1.0 / E4M3_MAX, dtype=torch.float32)
E2M1_MAX_RECIPROCAL = torch.tensor(1.0 / E2M1_MAX, dtype=torch.float32)
FLOAT32_MIN_NORMAL = torch.finfo(torch.float32).tiny


def flush_subnormals(values: torch.Tensor) -> torch.Tensor:
    """Flush-to-zero: subnormals become a zero of the same sign."""
    return torch.where(values.abs() < FLOAT32_MIN_NORMAL, values * 0, values)


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, K] -> (float8_e4m3fn [M, K], UE8M0 codes uint8 [M, K/32] row-major)."""
    rows, cols = x.shape
    blocks = x.float().view(rows, cols // MXFP8.block, MXFP8.block)
    block_amax = blocks.abs().amax(-1)                                  # fp32 [M, K/32]
    # cvt.rp.satfinite.ue8m0: the smallest 2**(code - 127) >= amax / 448, code in [0, 254].
    scale_target = flush_subnormals(block_amax * E4M3_MAX_RECIPROCAL.to(x.device))
    mantissa, exponent = torch.frexp(scale_target)
    ceil_log2 = torch.where(mantissa == 0.5, exponent - 1, exponent)
    scale_code = torch.where(scale_target > 0, (ceil_log2 + 127).clamp(0, 254), 0).to(torch.uint8)
    # Code 0 is 2**-127, a subnormal, so the fast-math zero test gives it a zero multiplier.
    inverse_scale = torch.where(scale_code > 0, torch.exp2(127.0 - scale_code.float()), 0.0)
    scaled = flush_subnormals(flush_subnormals(blocks) * inverse_scale[..., None])
    values = scaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(rows, cols)
    return values, scale_code


def quantize_nvfp4(x: torch.Tensor, global_scale: float | torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, K] -> (E2M1 pairs uint8 [M, K/2], even element in the low nibble;
    UE4M3 scales uint8 [M, K/16] row-major). global_scale is the encode scale S."""
    rows, cols = x.shape
    encode_scale = torch.as_tensor(global_scale, dtype=torch.float32, device=x.device).reshape(())
    blocks = flush_subnormals(x.float().view(rows, cols // NVFP4.block, NVFP4.block))
    block_amax = blocks.abs().amax(-1)                                  # fp32 [M, K/16]
    scale_target = flush_subnormals(
        encode_scale * flush_subnormals(block_amax * E2M1_MAX_RECIPROCAL.to(x.device)))
    scale_e4m3 = scale_target.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    inverse_scale = torch.where(
        block_amax != 0, 1.0 / flush_subnormals(scale_e4m3.float() * (1.0 / encode_scale)), 0.0)
    scaled = flush_subnormals(blocks * inverse_scale[..., None]).view(rows, cols)
    # cvt.rn.satfinite.e2m1: magnitude code 0-7 with ties to the even code, sign in
    # bit 3; NaN (0 * inf in a block whose scale underflowed) saturates to +6, code 7.
    magnitude = scaled.abs()
    magnitude_code = ((magnitude > 0.25).int() + (magnitude >= 0.75).int()
                      + (magnitude > 1.25).int() + (magnitude >= 1.75).int()
                      + (magnitude > 2.5).int() + (magnitude >= 3.5).int()
                      + (magnitude > 5.0).int())
    code = torch.where(torch.isnan(scaled), 7,
                       magnitude_code | (torch.signbit(scaled).int() << 3)).to(torch.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4), scale_e4m3.view(torch.uint8)


def dequantize_mxfp8(values: torch.Tensor, scale_code: torch.Tensor) -> torch.Tensor:
    """(float8_e4m3fn [M, K], UE8M0 uint8 [M, K/32]) -> fp32 [M, K]."""
    rows, cols = values.shape
    block_scale = torch.exp2(scale_code.float() - 127.0)
    return (values.float().view(rows, -1, MXFP8.block) * block_scale[..., None]).view(rows, cols)


def dequantize_nvfp4(packed: torch.Tensor, scale_e4m3: torch.Tensor,
                     global_scale: float | torch.Tensor) -> torch.Tensor:
    """(E2M1 pairs uint8 [M, K/2], UE4M3 uint8 [M, K/16], encode scale S) -> fp32 [M, K]."""
    rows = packed.shape[0]
    code_values = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), device=packed.device)
    code = torch.stack((packed & 0xF, packed >> 4), dim=-1).view(rows, -1).long()
    encode_scale = torch.as_tensor(global_scale, dtype=torch.float32, device=packed.device)
    block_scale = scale_e4m3.view(torch.float8_e4m3fn).float() / encode_scale
    return (code_values[code].view(rows, -1, NVFP4.block) * block_scale[..., None]).view(rows, -1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, *,
             weight_mode: Literal["none", "mul", "one_plus"] = "none", eps: float = 1e-6,
             round_factor: bool = False, residual: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """(bf16 [M, K] output, bf16 [M] factor r). "mul": bf16(bf16(x*r) * w), as
    Llama/Qwen and Pi0.5 AdaRMS; "one_plus": bf16(x*r*(1+w)) in fp32, as Gemma.
    `residual` is added first, in bf16; round_factor rounds r to bf16 before use."""
    summed = x if residual is None else x + residual
    summed_fp32 = summed.float()
    factor = torch.rsqrt(summed_fp32.pow(2).mean(-1, keepdim=True) + eps)
    factor = factor.to(torch.bfloat16).float() if round_factor else factor
    normalized = summed_fp32 * factor
    if weight_mode == "one_plus":
        output = (normalized * (1.0 + weight.float())).to(torch.bfloat16)
    elif weight_mode == "mul":
        output = normalized.to(torch.bfloat16) * weight
    else:
        output = normalized.to(torch.bfloat16)
    return output, factor.squeeze(-1).to(torch.bfloat16)


def layer_norm(x: torch.Tensor, weight: torch.Tensor | None = None,
               bias: torch.Tensor | None = None, *, eps: float = 1e-5,
               modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
               mod_group_rows: int = 0, residual: torch.Tensor | None = None) -> torch.Tensor:
    """LayerNorm in fp32 with one rounding, then optionally AdaLN's bf16
    ``y * (1 + scale) + shift``, modulation = (scale, shift) each [groups, K],
    one row per mod_group_rows rows (0: one row for all)."""
    summed = x if residual is None else x + residual
    rows, cols = summed.shape
    weight_fp32 = None if weight is None else weight.float()
    bias_fp32 = None if bias is None else bias.float()
    normalized = F.layer_norm(summed.float(), (cols,), weight_fp32, bias_fp32, eps)
    normalized = normalized.to(torch.bfloat16)
    if modulation is None:
        return normalized
    rows_per_vector = mod_group_rows or rows
    scale, shift = (vectors.reshape(-1, cols).repeat_interleave(rows_per_vector, 0)[:rows]
                    for vectors in modulation)
    return normalized * (1 + scale) + shift


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def gated_act(gate: torch.Tensor, up: torch.Tensor, *,
              act: Literal["gelu_tanh", "silu"] = "gelu_tanh",
              round_act: bool = False) -> torch.Tensor:
    """act(gate) * up. round_act=False: one rounding (Pi0.5 GeGLU); True: act
    rounds to bf16 first (HF SwiGLU, bf16(bf16(silu(g)) * u))."""
    activation_fn = {"gelu_tanh": lambda value: F.gelu(value, approximate="tanh"),
                     "silu": F.silu}[act]
    if round_act:
        return activation_fn(gate) * up
    return (activation_fn(gate.float()) * up.float()).to(torch.bfloat16)
