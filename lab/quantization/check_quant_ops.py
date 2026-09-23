"""Hold the fused-quant ops to FlashInfer 0.7.0 bit for bit, and feed its GEMMs.

For every activation shape (M, K) of the survey, in MXFP8 and NVFP4:

1. ``quant_ops.quantize`` against ``flashinfer.mxfp8_quantize`` /
   ``fp4_quantize`` (128x4 swizzled scales): the values and the whole scale
   buffer, padding included, must be byte-identical.
2. The same bytes against ``flash_vla.quantization.reference``: MXFP8
   identical, NVFP4 scales identical and values within one E2M1 step on under
   1% of values (the reference's reciprocals are exact, the kernel's approximate).
3. ``mm_mxfp8`` / ``mm_fp4`` (b12x) on our operands against the BF16 matmul,
   and against ``flash_vla.quantization.fake_quant``, which must reproduce the
   kernel's arithmetic up to FP32 summation order and the output rounding.

Inputs mix row magnitudes from 1e-3 to 1e2, zero blocks, outliers, tiny values
and negative zeros. Run from the repository root:

    PYTHONPATH=src CUDA_HOME=$HOME/cuda-13.1 flock /tmp/flash-vla-rtx5090.lock \\
        ~/quant-survey/venv-flashinfer/bin/python lab/quantization/check_quant_ops.py
"""
import math
from pathlib import Path
import sys
from typing import Literal

from flashinfer import fp4_quantize, mm_fp4, mm_mxfp8, mxfp8_quantize
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import cosine
from flash_vla.hardware.nvidia.quant_ops import ops
from flash_vla.quantization import formats, reference
from flash_vla.quantization.fake_quant import dequantized_operand, fake_quant_matmul
from vla_shapes import SHAPES

# A single row, a two-block row, and rows running past one 128-row scale tile.
EXTRA_SHAPES = {(1, 2048), (3, 64), (130, 96)}


def adversarial_activation(rows: int, cols: int, seed: int) -> torch.Tensor:
    """bf16 [rows, cols] with row magnitudes from 1e-3 to 1e2 and the edge cases
    of both formats' scale selection."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    row_magnitude = torch.exp(torch.empty(rows, 1, device="cuda").uniform_(-7, 4.6,
                                                                             generator=generator))
    x = torch.randn(rows, cols, device="cuda", generator=generator) * row_magnitude
    x[:, :32] = 0                              # a zero block in every row
    # Slices past a small shape are empty, so small shapes skip these cases.
    x[1:2, 32:64] = 1e-30                      # a block below the UE4M3/UE8M0 range
    x[2:3, 64:96] = -0.0
    x[3:4, 96:128] = 1e-39                     # bf16 subnormals
    x[4:5, 128:160] = -1e-39
    x[4:5, 130:131] = 1e-35                    # a tiny normal max over subnormals
    x[5:6] = 0                                 # a zero row
    outliers = torch.randint(0, rows * cols, (max(1, rows * cols // 4096),), device="cuda",
                             generator=generator)
    x.view(-1)[outliers] *= 300
    return x.to(torch.bfloat16)


def b12x_check(fmt: Literal["mxfp8", "nvfp4"], rows: int, cols: int, n: int
               ) -> tuple[float, float]:
    """b12x fed with our quantized activation: (cosine to the BF16 matmul,
    relative RMS difference from fake_quant's product of the same operand bytes)."""
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, cols, device="cuda", dtype=torch.bfloat16) * 0.05
    blocks = cols // formats.FORMATS[fmt].block
    if fmt == "mxfp8":
        activation = ops.quantize(x, ops.empty(rows, cols, "mxfp8"))
        weight_values, weight_scales = mxfp8_quantize(weight, True)
        product = mm_mxfp8(activation.values, weight_values.t(), activation.scale, weight_scales,
                           out_dtype=torch.bfloat16, backend="b12x")
        x_encode_scale, weight_encode_scale = None, None
    else:
        x_encode_scale = ((448 * 6) / x.float().abs().max()).reshape(1).float()
        weight_encode_scale = ((448 * 6) / weight.float().abs().max()).reshape(1).float()
        activation = ops.quantize(x, ops.empty(rows, cols, "nvfp4", global_scale=x_encode_scale))
        weight_values, weight_scales = fp4_quantize(weight, weight_encode_scale, 16, False, True)
        product = mm_fp4(activation.values, weight_values.t(), activation.scale, weight_scales.t(),
                         (1.0 / (x_encode_scale * weight_encode_scale)).float(), torch.bfloat16,
                         backend="b12x")
    fake_product = fake_quant_matmul(
        dequantized_operand(activation.values, formats.unswizzle(activation.scale, rows, blocks),
                            fmt, x_encode_scale),
        dequantized_operand(weight_values, formats.unswizzle(weight_scales.view(torch.uint8), n,
                                                             blocks),
                            fmt, weight_encode_scale)).bfloat16()
    difference = (product.float() - fake_product.float()).pow(2).mean().sqrt()
    return (cosine(product, x @ weight.t()),
            float(difference / fake_product.float().pow(2).mean().sqrt()))


def main() -> None:
    gemm_n = {(site.m, site.k): site.n for site in SHAPES}
    failures = 0
    print(f"{'M':>5} {'K':>6} {'fmt':>6}  flashinfer-bytes  scale-bytes  reference        gemm-cos"
          "  vs-fake-quant")
    for rows, cols in sorted(set(gemm_n) | EXTRA_SHAPES):
        x = adversarial_activation(rows, cols, seed=rows * 100003 + cols)
        encode_scale = ((448 * 6) / x.float().abs().max()).reshape(1).float()
        for fmt in [fmt for fmt in ("mxfp8", "nvfp4") if cols % formats.FORMATS[fmt].block == 0]:
            ours = ops.quantize(x, ops.empty(rows, cols, fmt, global_scale=encode_scale))
            flashinfer_values, flashinfer_scales = (
                mxfp8_quantize(x, True) if fmt == "mxfp8"
                else fp4_quantize(x, encode_scale, 16, False, True))
            our_bytes = ours.values.view(torch.uint8)
            values_equal = torch.equal(our_bytes, flashinfer_values.view(torch.uint8))
            scales_equal = torch.equal(ours.scale, flashinfer_scales.flatten())
            our_scales = formats.unswizzle(ours.scale, rows, cols // formats.FORMATS[fmt].block)
            if fmt == "mxfp8":
                expected_values, expected_scales = reference.quantize_mxfp8(x)
                reference_ok = (torch.equal(our_bytes, expected_values.view(torch.uint8))
                                and torch.equal(our_scales, expected_scales))
                reference_note = "identical" if reference_ok else "DIFFERS"
            else:
                expected_packed, expected_scales = reference.quantize_nvfp4(x, encode_scale)
                # E2M1 codes as signed steps (code 8 + c is -c, so +0 and -0 are both 0).
                signed_steps = [torch.where(codes >= 8, 8 - codes, codes) for codes in (
                    torch.stack((packed & 0xF, packed >> 4), -1).flatten().int()
                    for packed in (ours.values, expected_packed))]
                step_distance = (signed_steps[0] - signed_steps[1]).abs()
                mismatch_fraction = float((step_distance > 0).float().mean())
                scales_match_reference = torch.equal(our_scales, expected_scales)
                reference_ok = (scales_match_reference and int(step_distance.max()) <= 1
                                and mismatch_fraction < 1e-2)
                reference_note = (f"{mismatch_fraction:.1e} off, max {int(step_distance.max())}"
                                  + ("" if scales_match_reference else " SF!"))
            n = gemm_n.get((rows, cols))
            similarity, fake_quant_rel_rms = b12x_check(fmt, rows, cols, n) if n else (math.nan,
                                                                                        math.nan)
            ok = (values_equal and scales_equal and reference_ok
                  and (math.isnan(similarity) or (similarity > 0.98 and fake_quant_rel_rms < 2e-3)))
            failures += not ok
            print(f"{rows:>5} {cols:>6} {fmt:>6}  {'identical' if values_equal else 'DIFFERS':>16}"
                  f"  {'identical' if scales_equal else 'DIFFERS':>11}  {reference_note:<15}"
                  f"  {similarity:8.5f}  {fake_quant_rel_rms:13.2e}" + ("" if ok else "   <-- FAIL"))
    print("all checks passed" if not failures else f"{failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
